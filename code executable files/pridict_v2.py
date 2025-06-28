import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load your new model
model = load_model("fruit_classifier_model_v2.h5")

# Path to the test image (update this!)
img_path = r"C:\Users\murik\OneDrive\Desktop\mangotest.jpeg"

# Preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
pred = model.predict(img_array)
predicted_class_index = np.argmax(pred[0])

# Match index to class name
train_dir = r"C:\Users\murik\Downloads\fruit_sort\output_dataset\train"
class_names = sorted(os.listdir(train_dir))
print("Predicted class:", class_names[predicted_class_index])
