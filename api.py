from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)
MODEL_PATH = './model/snake-EfficientNetV2.h5'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Banded_Krait',
 'Eastern_Russells_Viper',
 'Green_Pit_Viper',
 'Indo_Chinese_Rat_Snake',
 'King_Cobra',
 'Malayan_Krait',
 'Malayan_Pit_Viper',
 'Monocled_Cobra'] 

MODEL_PATH_2 = './model/final_model1_final.h5'
CLASS_NAMES2 = ['Non Venomous', 'Venomous'] 

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model2 = tf.keras.models.load_model(MODEL_PATH_2)
    print("model 1 loaded successfully")
    print(f"Input shape: {model.input_shape}")
    print("model 2 loaded successfully")
    print(f"Input shape: {model2.input_shape}")
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    model = None
    model2 = None

def preprocess_image(image, target_size):

    image = image.resize(target_size)
    
    img_array = np.array(image, dtype=np.float32)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    img_array = img_array / 127.5 - 1.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'EfficientNetV2 CNN API',
        'model_loaded': model is not None,
        'expected_input_size': IMG_SIZE,
        'num_classes': len(CLASS_NAMES)
    })

@app.route('/model1/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        image = None
        
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({
                    'error': 'Missing "image" field in request body'
                }), 400
            
            image_data = data['image']
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected'
                }), 400
            
            image = Image.open(file.stream)
        
        else:
            return jsonify({
                'error': 'No image provided. Send as JSON with "image" field or multipart form-data with "file"'
            }), 400
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_image = preprocess_image(image, IMG_SIZE)
        
        predictions = model.predict(processed_image, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        top_k = min(3, len(CLASS_NAMES))
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        top_predictions = [
            {
                'class': CLASS_NAMES[idx],
                'class_index': int(idx),
                'confidence': float(predictions[0][idx])
            }
            for idx in top_indices
        ]
        
        return jsonify({
            'success': True,
            'predicted_class': CLASS_NAMES[predicted_class_idx],
            'class_index': int(predicted_class_idx),
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_probabilities': predictions[0].tolist()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/model2/predict', methods=['POST'])
def predict2():
    try:
        if model2 is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        image = None
        
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({
                    'error': 'Missing "image" field in request body'
                }), 400
            
            image_data = data['image']
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected'
                }), 400
            
            image = Image.open(file.stream)
        
        else:
            return jsonify({
                'error': 'No image provided. Send as JSON with "image" field or multipart form-data with "file"'
            }), 400
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_image = preprocess_image(image, IMG_SIZE)
        
        predictions = model2.predict(processed_image, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        top_k = min(3, len(CLASS_NAMES2))
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        top_predictions = [
            {
                'class': CLASS_NAMES2[idx],
                'class_index': int(idx),
                'confidence': float(predictions[0][idx])
            }
            for idx in top_indices
        ]
        
        return jsonify({
            'success': True,
            'predicted_class': CLASS_NAMES2[predicted_class_idx],
            'class_index': int(predicted_class_idx),
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_probabilities': predictions[0].tolist()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        return jsonify({
            'model_type': 'EfficientNetV2 CNN',
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'expected_image_size': IMG_SIZE,
            'total_params': model.count_params(),
            'layers': len(model.layers),
            'classes': CLASS_NAMES
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Expected image size: {IMG_SIZE}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    app.run(host='0.0.0.0', port=3400, debug=True)
