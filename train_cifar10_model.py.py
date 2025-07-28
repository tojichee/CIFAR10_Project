# -*- coding: utf-8 -*-
"""
CIFAR-10 CNN Model Training, Evaluation, and Custom Image Prediction
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2  # Used for image preprocessing for custom images
import os   # For saving the model

# --- Part 1: Load and Preprocess CIFAR-10 Dataset ---
print("Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("CIFAR-10 dataset loaded and preprocessed.")

# --- Part 2: Build the CNN Model ---
print("\nBuilding the CNN model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# --- Requirement 1: Show all the layers in your model ---
print("\n--- Model Summary (Layers) ---")
model.summary()
print("----------------------------")

# --- Part 3: Compile and Train the Model ---
print("\nCompiling the model...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

print("Training the model (this may take a few minutes)...")
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

print("\nModel training complete.")

# --- Part 4: Evaluate the Model ---
print("\n--- Model Evaluation ---")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# --- Requirement 2: Show the accuracy score of your model ---
print(f'\nTest accuracy: {test_acc:.4f}')
print("------------------------")

# --- Requirement 3: Show the model's prediction on three images of your own ---
print("\n--- Model Predictions on Your Custom Images ---")

# --- !!! CRITICAL: CHANGE THESE PATHS to your actual image files !!! ---
# --- Use forward slashes: / ---
custom_image_paths = [
    'C:/Users/Brendan/path/to/your/deer_image.png',    # <-- EDIT THIS PATH
    'C:/Users/Brendan/path/to/your/car_image.jpg',     # <-- EDIT THIS PATH
    'C:/Users/Brendan/path/to/your/bird_image.jpeg'    # <-- EDIT THIS PATH
]

plt.figure(figsize=(15, 5))

for i, image_path in enumerate(custom_image_paths):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image at {image_path}. Please check the file path.")
        continue  # Skip to the next image if one fails to load

    # --- Preprocess the image for the model ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color
    img_resized = cv2.resize(img_rgb, (32, 32))      # Resize to 32x32 pixels
    img_normalized = img_resized / 255.0             # Normalize
    image_for_prediction = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # --- Make Prediction ---
    predictions = model.predict(image_for_prediction)
    predicted_class_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100

    # --- Display the Result ---
    plt.subplot(1, len(custom_image_paths), i + 1)
    plt.imshow(img_rgb)  # Show the original image for better quality
    plt.title(f"Prediction: {predicted_label}\n({confidence:.2f}%)")
    plt.axis('off')

print("Displaying predictions for custom images...")
plt.tight_layout()
plt.show()  # This will open a plot window. Save this image for your report.
print("--------------------------------------")

# --- Part 6: Save the trained model ---
# This saves the model in the same directory where the script is run.
model_save_path = 'cifar10_cnn_model.h5'
model.save(model_save_path)
print(f"\nModel saved to {model_save_path}")
print("This model can now be copied into your Webots controller folder.")