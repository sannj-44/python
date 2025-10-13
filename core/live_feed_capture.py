# core/live_feed_capture.py

import cv2
import time

def capture_live_feed(classifier, duration=10):
    """
    Opens webcam feed for a given duration (seconds), classifies each frame using
    WasteImageClassifier, and returns all predicted labels.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access the camera.")
        return []

    start_time = time.time()
    predictions = []

    print(f"\n🎥 Live feed started for {duration} seconds...\nPress 'q' to stop early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Could not read frame from camera.")
            break

        label, confidence = classifier.classify(frame)
        predictions.append(label)

        # Display prediction on frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Smart Waste Classifier - Live Feed', frame)

        # Print classification info to console
        print(f"Detected: {label} ({confidence:.2f})")

        # Stop after duration or when 'q' is pressed
        if time.time() - start_time > duration:
            print("\n🕓 Time limit reached. Closing camera...\n")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Stopped by user.\n")
            break

    cap.release()
    cv2.destroyAllWindows()
    return predictions
