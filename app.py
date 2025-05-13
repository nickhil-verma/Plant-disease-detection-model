
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model and class labels
model = tf.keras.models.load_model("plant_disease_model.h5")
with open("class_labels.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

img_height, img_width = 128, 128

app = Flask(__name__)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((img_width, img_height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    img_bytes = image_file.read()
    img_array = preprocess_image(img_bytes)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_label = class_labels[class_index]
    confidence = float(np.max(prediction))

    return jsonify({
        'predicted_label': predicted_label,
        'confidence': round(confidence, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
