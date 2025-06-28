import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = load_model('fruit_classifier_model_v3.h5')

# Define the image path (CHANGE this to your test image path)
img_path = r"C:\Users\murik\OneDrive\Desktop\test1.jpeg"  # example image path

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
predictions = model.predict(img_array)[0]

# Get class names from training folder
class_names = sorted(os.listdir('C:/Users/murik/Downloads/fruit_sort/output_dataset/train'))

# Get top 3 predictions
top_indices = predictions.argsort()[-3:][::-1]
top_classes = [(class_names[i], predictions[i]) for i in top_indices]

# Print results
print("\n🔍 Top 3 Predictions:")
for cls, prob in top_classes:
    print(f"{cls:25s}: {prob:.4f}")
