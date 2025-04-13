# Import necessary libraries
import os
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

# Check if TensorFlow can access the GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("Num GPUs Available: ", len(physical_devices))
else:
    print("No GPUs found. Using CPU.")

# Load the trained model inside a GPU context
with tf.device('/GPU:0'):  # Use GPU:0 or replace with appropriate GPU ID
    model = load_model(r"C:\Users\prane\Downloads\Glaucoma_Detection\GLAUCOMA_DETECTION.h5", compile=False)
    print("Model is loaded")
# Custom preprocessing function
def custom_preprocessing(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (65, 65), 0)

    # Find the pixel with the highest intensity value
    max_intensity_pixel = np.unravel_index(np.argmax(blurred_image), blurred_image.shape)

    # Define the radius for the circle
    radius = 200 // 2

    # Get the x and y coordinates for cropping the image
    x = max_intensity_pixel[1] - radius
    y = max_intensity_pixel[0] - radius

    # Create a mask for the circle
    mask = np.zeros_like(image)
    cv2.circle(mask, (x + radius, y + radius), radius, (255, 255, 255), -1)

    # Apply the mask to the original image
    roi_image = cv2.bitwise_and(image, mask)

    # Split the green channel and apply histogram equalization
    green_channel = roi_image[:, :, 1]
    clahe_op = cv2.createCLAHE(clipLimit=2)
    roi_image = clahe_op.apply(green_channel)

    # Convert back to BGR to maintain three channels
    roi_image = cv2.merge([roi_image, roi_image, roi_image])  # Create a 3-channel image

    return roi_image

# Function to make predictions on a single image
def predict_image(image_path):
    with tf.device('/GPU:0'):  # Use GPU:0 or replace with appropriate GPU ID
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))  # Resize to match the model input size
        processed_image = custom_preprocessing(image)

        # Normalize the image to match training preprocessing (if any)
        processed_image = processed_image / 255.0

        # Convert processed_image to uint8
        processed_image = (processed_image * 255).astype(np.uint8)

        # Expand dimensions to make it (1, 256, 256, 3)
        processed_image = np.expand_dims(processed_image, axis=0)

        # Make a prediction
        prediction = model.predict(processed_image)[0][0]
        print(prediction)

        # Interpret the prediction
        if prediction > 0.5:
            print("Prediction: Glaucoma detected")
        else:
            print("Prediction: No Glaucoma")

        # Show original and ROI images
        plt.figure(figsize=(12, 6))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        plt.title('Original Image')
        plt.axis('off')  # Hide axis

        # ROI image
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(processed_image[0], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        plt.title('Region of Interest (ROI)')
        plt.axis('off')  # Hide axis

        plt.tight_layout()
        plt.show()

# Example usage
predict_image(r"C:\Users\prane\Downloads\Glaucoma_Detection\noglaucoma5.jpg")
