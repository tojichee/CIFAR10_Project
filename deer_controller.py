# Webots Controller for Object Classification

from controller import Robot
import numpy as np
import tensorflow as tf
import cv2

def preprocess_image(img_data, width, height):
    """Converts camera data to a format the CNN can use."""
    # Convert the raw image data from Webots to a numpy array
    img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 4))

    # Convert BGRA (Webots format) to RGB (TensorFlow format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # Resize image to 32x32 pixels, the size our model was trained on
    img_resized = cv2.resize(img_rgb, (32, 32))

    # Normalize pixel values to be between 0 and 1
    img_normalized = img_resized / 255.0

    # Add the "batch" dimension, as the model expects a list of images
    return np.expand_dims(img_normalized, axis=0)

# --- Main Controller Logic ---

# 1. Initialization
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get and enable the camera
# Make sure the camera device in Webots is named 'camera'
camera = robot.getDevice('camera')
camera.enable(timestep)
width = camera.getWidth()
height = camera.getHeight()

# 2. Load the pre-trained CNN model
# This .h5 file MUST be in the same folder as this script.
print("Loading the trained CNN model...")
try:
    model = tf.keras.models.load_model('cifar10_cnn_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'cifar10_cnn_model.h5' is in the controller directory.")
    # Stop the robot if the model can't be loaded
    while robot.step(timestep) != -1:
        pass

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 3. Main loop
print("\nStarting prediction loop...")
while robot.step(timestep) != -1:
    # 4. Get image from camera and preprocess it
    image_data = camera.getImage()
    if image_data:
        processed_image = preprocess_image(image_data, width, height)

        # 5. Make a prediction using the loaded model
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = np.max(predictions[0]) * 100

        # 6. Print the result to the Webots console
        # This fulfills Requirement 4
        print(f"Object Detected: {predicted_class} (Confidence: {confidence:.2f}%)")

    # Add a delay to avoid spamming the console and using too much CPU
    # This makes one prediction every 1 second
    robot.step(1000)