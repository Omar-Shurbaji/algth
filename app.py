from flask import Flask, request, jsonify, send_file ,render_template
import cv2
import numpy as np
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from flask_cors import CORS
import io
import os
import base64
from PIL import Image
import mysql.connector


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://healthsync-website.surge.sh", "https://omar-shurbaji.github.io/test/"]}})

epsilon = 1e-5
smooth = 1

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def focal_tversky(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

# Loading the segmentation model
new_model = load_model('./models/model-5.keras')
# model = load_model("seg_model.h5",custom_objects={"focal_tversky":focal_tversky,"tversky":tversky,"tversky_loss":tversky_loss})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'})

    try:
        img = Image.open(image_file)
        img = img.resize((224, 224))
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = new_model.predict(img)
        result = "Pneumonia" if prediction[0][0] < 0.5 else "Normal"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

def reconstruct_original_image(original_base64):
    original_image = cv2.imdecode(np.frombuffer(base64.b64decode(original_base64), np.uint8), cv2.IMREAD_COLOR)
    return original_image

@app.route('/')
def index():
    return "Hello, Flask App . . ."

# @app.route('/predict_vision', methods=['POST'])
# def predict_vision():
#     if request.method == 'POST':
#         file = request.files['image_file']
#         img = Image.open(io.BytesIO(file.read()))
#         original_image = np.array(img)  # الحفاظ على الصورة الأصلية
        
#         img = cv2.resize(original_image, (256, 256))
#         img = np.array(img, dtype=np.float64)
#         img -= img.mean()
#         img /= img.std()
#         img = np.expand_dims(img, axis=0)
        
#         prediction_result = model.predict(img)
#         predicted_image = (prediction_result.squeeze().round() * 255).astype(np.uint8)
        
#         # تحويل الصور إلى Base64
#         _, buffer_predicted = cv2.imencode('.jpg', predicted_image)
#         _, buffer_original = cv2.imencode('.jpg', original_image)

#         predicted_image_base64 = base64.b64encode(buffer_predicted).decode('utf-8')
#         original_image_base64 = base64.b64encode(buffer_original).decode('utf-8')
        
#         return jsonify({
#             'predicted_image': predicted_image_base64,
#             'original_image': original_image_base64
#         })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)

