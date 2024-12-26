import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# Define dataset paths
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Check GPU availability
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# Data generators for loading and augmenting images
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

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

# Print out class indices for debugging
print(f"Train Class Indices: {train_generator.class_indices}")
print(f"Validation Class Indices: {val_generator.class_indices}")

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)  # Updated to 8 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save the trained model
model.save('color_classifier_8_classes.h5')
print("Model saved as 'color_classifier_8_classes.h5'.")

# Define a function for prediction
def predict_color(image_path, model, class_indices):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_labels = {v: k for k, v in class_indices.items()}
    return class_labels[predicted_class], predictions[0][predicted_class]

# Test the model on a new image
test_image_path = r'C:\Users\User\ImageClassifier\yellow.jpg'  # Replace with your image path
model = tf.keras.models.load_model('color_classifier_8_classes.h5')
predicted_class, confidence = predict_color(test_image_path, model, train_generator.class_indices)

print(f"Predicted Class: {predicted_class} with confidence {confidence:.2f}")
