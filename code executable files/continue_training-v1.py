import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# === STEP 1: Define paths ===
train_dir = "C:/Users/murik/Downloads/fruit_sort/output_dataset/train"
val_dir = "C:/Users/murik/Downloads/fruit_sort/output_dataset/val"
model_path = "C:/Users/murik/Downloads/fruit_sort/fruit_classifier_model.h5"

# === STEP 2: Image settings ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# === STEP 3: Load data ===
train_datagen = ImageDataGenerator(rescale=1./255)
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

# === STEP 4: Load previous model ===
model = load_model(model_path)

# === STEP 5: Compile model again ===
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === STEP 6: Continue training from epoch 6 ===
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=12,         # training until epoch 12-    initial_epoch=5    # resume from epoch 6
)

# === STEP 7: Save new model ===
model.save("fruit_classifier_model_v2.h5")
print("Model retrained and saved as 'fruit_classifier_model_v2.h5'")
