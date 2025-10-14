# interface/live_console.py
def start_console(classifier, duration=10):
    from core.live_feed_capture import capture_live_feed
    
    # Now call capture_live_feed with the classifier
    results = capture_live_feed(classifier, duration=10) 
    return results # Return the results