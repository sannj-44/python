
class WasteImageClassifier:
    def __init__(self, model_path='models/waste_cnn_model.h5'):
        self.model = tf.keras.models.load_model(model_path) # pyright: ignore[reportAttributeAccessIssue]
        self.labels = ['Organic', 'Recyclable', 'Hazardous', 'Other']

    def predict_frame(self, frame):
        # Preprocess frame
        frame_resized = cv2.resize(frame, (128, 128)) / 255.0
        frame_input = np.expand_dims(frame_resized, axis=0)
        
        # Predict
        result = self.model.predict(frame_input)
        label = self.labels[np.argmax(result)]
        return label

