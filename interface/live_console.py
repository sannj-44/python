# interface/live_console.py

from core.ml_image_classifier import WasteImageClassifier
from core.live_feed_capture import capture_live_feed
from visual.plot_graphs import plot_results

def start_console():
    classifier = WasteImageClassifier("models/waste_cnn_model.h5")
    results = capture_live_feed(classifier, duration=10)
    plot_results(results)

if __name__ == "__main__":
    start_console()
