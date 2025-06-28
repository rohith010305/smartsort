from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
model = load_model('fruit_classifier_model_v5.h5')

# Get class names from training folder
try:
    class_names = sorted(os.listdir('output_dataset/train'))
except FileNotFoundError:
    raise Exception(" ERROR: 'output_dataset/train' folder not found. Please create it and add class folders.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    file = request.files.get('file')
    if not file or file.filename == '':
        return 'No file uploaded', 400

    os.makedirs('static', exist_ok=True)
    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    # Preprocess image
    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    pred = model.predict(arr)[0]
    cls = class_names[np.argmax(pred)]

    return render_template('result.html', prediction=cls, image_path=img_path)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
