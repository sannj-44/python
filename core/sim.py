import cv2
import time
import tkinter as tk
from tkinter import filedialog # <-- New Import

def select_video_file():
    """Opens a file dialog for the user to select a video file."""
    # Initialize Tkinter (hidden root window is created)
    root = tk.Tk()
    root.withdraw() 
    
    # Open the file selection dialog
    video_path = filedialog.askopenfilename(
        title="Select Video Dataset File",
        filetypes=(("Video files", "*.mp4;*.avi;*.mov"), ("All files", "*.*"))
    )
    
    return video_path

def process_video_file(classifier, video_path):
    """
    Processes a video file, classifies each frame, and returns all predicted labels.
    """
    # Check if a path was selected (user didn't press Cancel)
    if not video_path:
        print("🛑 Video selection cancelled by user.")
        return []
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at {video_path}")
        return []

    print(f"\n🎥 Starting simulation from video: {video_path}...\n")
    predictions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("🎬 End of video stream.")
            break

        # Classification
        label, confidence = classifier.classify(frame)
        predictions.append(label)

        # Display prediction on frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Smart Waste Classifier - Video Simulation', frame)
        
        # Add a delay for better viewing speed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("\n🛑 Stopped by user.\n")
            break

        frame_count += 1
        # Optional: Print every Nth frame prediction to console
        if frame_count % 30 == 0: 
            print(f"Frame {frame_count}: Detected {label} ({confidence:.2f})")

    cap.release()
    cv2.destroyAllWindows()
    return predictions