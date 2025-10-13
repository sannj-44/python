# core/ml_image_classifier.py

import tensorflow as tf
import numpy as np
import cv2

class WasteImageClassifier:
    def __init__(self, model_path):
        # Load your trained Keras model (.h5)
        self.model = tf.keras.models.load_model(model_path) # type: ignore
        # Define class names in the same order as your training dataset
        self.class_names = ['Organic', 'Recyclable', 'Hazardous', 'Other']

    def classify(self, frame):
        """
        Takes an image frame (from OpenCV), preprocesses it, and predicts the waste class.
        Returns: (label, confidence)
        """
        # Resize to match model input shape
        img = cv2.resize(frame, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
        img = np.expand_dims(img, axis=0) / 255.0   # Normalize and add batch dimension

        # Model prediction
        preds = self.model.predict(img, verbose=0)[0]
        label_index = np.argmax(preds)
        confidence = float(preds[label_index])

        return self.class_names[label_index], confidence
