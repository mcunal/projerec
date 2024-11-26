from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64
from flask_cors import CORS

# Flask application setup
app = Flask(__name__)
CORS(app)  # Enable CORS support for frontend

# Load pre-trained MNIST model
model = load_model('mnist_model.h4')

# Home route
@app.route('/')
def home():
    return 'MNIST Model API'

# Recognize endpoint to handle image input and make predictions
@app.route('/recognize', methods=['POST'])
def recognize():
    # Get the image in base64 format from the POST request
    img_data = request.json['image']
    
    # Decode the base64 image data
    img_data = img_data.split(',')[1]  # Ignore the initial part of the base64 string
    img = Image.open(io.BytesIO(base64.b64decode(img_data)))
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels (MNIST input size)
    img = np.array(img) / 255.0  # Normalize the image data
    img = img.reshape(1, 28, 28, 1)  # Reshape for the model input (28, 28, 1)

    # Make prediction
    prediction = model.predict(img)

    # Get the predicted class and its confidence level
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_confidence = np.max(prediction)

    # Return the prediction and confidence in JSON format
    return jsonify({'prediction': str(predicted_class), 'confidence': str(predicted_confidence)})

if __name__ == '__main__':
    app.run(debug=True)
