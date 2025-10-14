import tensorflow as tf
import numpy as np
import cv2

class WasteImageClassifier:
    def __init__(self, model_path):
        # Load the trained Keras model
        self.model = tf.keras.models.load_model(model_path) # type: ignore
        
        # 1. Define the 10 Trained Class Names (MUST match model's training order/output)
        # Assuming the order is alphabetical, matching the directory structure you showed:
        self.trained_class_names = [
            'battery', 'biological', 'cardboard', 'clothes', 
            'glass', 'metal', 'paper', 'plastic', 
            'shoes', 'trash'
        ]
        
        # 2. Define the 4 Target Categories (for user display)
        self.target_categories = ['Organic', 'Recyclable', 'Hazardous', 'Other']

    def _map_to_category(self, trained_label):
        """Maps one of the 10 trained labels to one of the 4 final categories."""
        
        # --- Mapping Logic ---
        
        # Organic
        if trained_label in ['biological']:
            return 'Organic'
        
        # Recyclable
        elif trained_label in ['cardboard', 'glass', 'metal', 'paper', 'plastic']:
            return 'Recyclable'
            
        # Hazardous
        elif trained_label in ['battery']:
            return 'Hazardous'
            
        # Other (General Waste / Non-classifiable)
        elif trained_label in ['clothes', 'shoes', 'trash']:
            return 'Other'
            
        # Fallback for safety (though all 10 are covered above)
        return 'Other'

    def classify(self, frame):
        """
        Takes an image frame, preprocesses it, and predicts the waste class.
        Returns: (final_category_label, confidence)
        """
        # Preprocessing
        img = cv2.resize(frame, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) / 255.0

        # Model prediction (Outputs 10 probabilities)
        preds = self.model.predict(img, verbose=0)[0]
        
        # Find the best prediction index
        label_index = np.argmax(preds)
        
        # Use the correct, 10-item list to get the trained label name
        # This fixes the 'IndexError: list index out of range'
        trained_label = self.trained_class_names[label_index]

        confidence = float(preds[label_index])
        
        # Map the specific trained label (e.g., 'plastic') to the final category (e.g., 'Recyclable')
        final_category = self._map_to_category(trained_label)

        # Return the final category and the confidence of the original prediction
        return final_category, confidence