from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Config
model_path = "plant_disease_model.h5"
labels_path = "class_labels.txt"
img_height, img_width = 128, 128

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and labels
model = tf.keras.models.load_model(model_path)
with open(labels_path, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Image preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)
            label = class_labels[np.argmax(prediction)]
            confidence = round(float(np.max(prediction)) * 100, 2)
            return render_template('upload.html', prediction=label, confidence=confidence, filename=filename)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
