from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load previous model trained up to 20 epochs
model = load_model('fruit_classifier_model_v5.h5')

# Define image directories
train_dir = 'Fruit And Vegetable Diseases Dataset/output_dataset/train'
val_dir = 'Fruit And Vegetable Diseases Dataset/output_dataset/validation'

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train from epoch 20 to 22
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=22,           # Final epoch
    initial_epoch=20     # Continue from epoch 20
)

# Save the new model
model.save('fruit_classifier_model_v6.h5')
print("✅ Model retrained and saved as fruit_classifier_model_v6.h5")
