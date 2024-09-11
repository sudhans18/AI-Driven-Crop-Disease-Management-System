import os
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model(r"D:\Rithvik\potato-disease-classification-main\potato-disease-classification-main\Prediction\models/model1.h5")

# Define the class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Define disease actions
disease_actions = {
    "Potato___Late_blight": {
        "preventive": [
            "Plant disease-resistant crop varieties.",
            "Rotate crops to prevent pathogen buildup.",
            "Avoid overhead irrigation to reduce moisture on leaves."
        ],
        "treatment": [
            "Apply fungicides such as copper-based products.",
            "Remove infected plants to prevent the disease from spreading.",
            "Maintain good air circulation by proper plant spacing."
        ]
    },
    "Potato___Early_blight": {
        "preventive": [
            "Use disease-resistant varieties (e.g., ‘Russet Burbank’).",
            "Rotate crops (avoid planting potatoes in the same field for 2-3 years).",
            "Use drip irrigation to keep foliage dry and water early in the day.",
            "Ensure proper plant spacing to improve air circulation.",
            "Apply mulch to reduce soil splashing and retain moisture.",
            "Remove and destroy crop debris after harvest.",
            "Apply preventive fungicides (e.g., copper-based) as needed.",
            "Avoid over-fertilization with nitrogen; use balanced fertilizers"
        ],
        "treatment": [
            "Apply fungicides (e.g., copper-based, chlorothalonil, or mancozeb) at the first sign of infection.",
            "Remove infected plant parts to prevent further spread.",
            "Improve air circulation by pruning overcrowded plants.",
            "Rotate fungicides to prevent resistance buildup in the fungus.",
            "Maintain proper irrigation (avoid overhead watering) to minimize moisture on leaves"
        ]
    }
}

# Allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_image(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static/uploads', filename)
            file.save(filepath)

            predicted_class, confidence = predict_image(model, filepath)

            if predicted_class in disease_actions:
                preventive_measures = disease_actions[predicted_class]['preventive']
                treatment_recommendations = disease_actions[predicted_class]['treatment']
            else:
                preventive_measures = []
                treatment_recommendations = []

            return render_template(
                'result.html', 
                filename=filename, 
                predicted_class=predicted_class, 
                confidence=confidence, 
                preventive_measures=preventive_measures,
                treatment_recommendations=treatment_recommendations
            )

    return redirect(url_for('upload_file'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
