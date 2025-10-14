# app.py

import os
from interface.live_console import start_console
# Import the necessary functions from core/simulation
from core.sim import process_video_file, select_video_file 
from core.ml_image_classifier import WasteImageClassifier
from visual.plot_graphs import plot_classification_results

# Define the path to your model file
MODEL_PATH = "models/waste_cnn_model.h5"
# VIDEO_DATASET_PATH is no longer needed here as the user will select it

def main():
    print("=== Smart Waste Classifier ===")
    
    # 1. Initialize the Classifier
    print("Loading Machine Learning Model...")
    try:
        classifier = WasteImageClassifier(MODEL_PATH)
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load model from {MODEL_PATH}. Details: {e}")
        return

    # 2. Present Options to the User
    while True:
        print("\n--- Select Operation Mode ---")
        print("1: Live Camera Feed (Console + Plotting)")
        print("2: Process Video Dataset (Select File + Plotting)") # Updated text
        print("q: Quit")
        choice = input("Enter choice (1, 2, or q): ").strip().lower()

        predictions = []
        
        if choice == '1':
            print("\nMode: Live Camera Feed")
            predictions = start_console(classifier) 
            break
        
        elif choice == '2':
            print("\nMode: Video Dataset Simulation - Please select a video file...")
            
            # --- NEW LOGIC: Call the file selector ---
            selected_video_path = select_video_file()
            
            if selected_video_path:
                # If a file was selected, process it
                predictions = process_video_file(classifier, selected_video_path)
                break
            else:
                # If the user cancelled the dialog, loop back to the menu
                continue 
            
        elif choice == 'q':
            print("Exiting application. Goodbye!")
            return
            
        else:
            print("Invalid choice. Please enter 1, 2, or q.")

    # 3. Plot Results (Executed after live_console or process_video_file completes)
    if predictions:
        print("\nGenerating classification plots...")
        plot_classification_results(predictions)
        
    print("\nApplication finished.")

if __name__ == "__main__":
    main()