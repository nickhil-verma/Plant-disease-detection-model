from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__, static_url_path='/static')
model = load_model('model.h5')
app.config['UPLOAD_FOLDER'] = 'static/uploads'

class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']  # Update as needed

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            image = Image.open(path).resize((224, 224))
            image = np.expand_dims(np.array(image) / 255.0, axis=0)
            preds = model.predict(image)[0]
            prediction = class_names[np.argmax(preds)]
            confidence = f"{100 * np.max(preds):.2f}%"

    return render_template('upload.html', prediction=prediction, confidence=confidence, filename=filename)

# Required for Vercel
def handler(request, context):
    return app(request.environ, start_response=context['start_response'])
