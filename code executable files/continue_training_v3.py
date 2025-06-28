import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Paths
base_dir = "C:/Users/murik/Downloads/fruit_sort/output_dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load model and continue training
model = load_model("fruit_classifier_model_v2.h5")

# Compile again before training more
model.compile(optimizer=Adam(learning_rate=1e-4),  # Low LR for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training from 13th epoch
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=18,  # Continue up to epoch 18
    initial_epoch=12  # Resume from where you stopped
)

# Save updated model
model.save("fruit_classifier_model_v3.h5")
print("*Training complete. Model saved as fruit_classifier_model_v3.h5")
