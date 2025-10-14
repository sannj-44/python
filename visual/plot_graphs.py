# visualization/plot_graphs.py

import matplotlib.pyplot as plt
from collections import Counter

def plot_classification_results(predictions):
    """Generates and displays a bar chart of the classification results."""
    if not predictions:
        print("No predictions to plot.")
        return

    # 1. Count the occurrences of each category
    counts = Counter(predictions)
    labels = list(counts.keys())
    values = list(counts.values())

    # 2. Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['green', 'blue', 'red', 'gray']) # Adjust colors as needed
    plt.title('Waste Classification Results')
    plt.xlabel('Waste Category')
    plt.ylabel('Number of Detections')

    # Display the percentage on the bars (Optional but helpful)
    total = sum(values)
    for i, v in enumerate(values):
        percentage = (v / total) * 100
        plt.text(i, v + 0.5, f"{percentage:.1f}%", ha='center', fontsize=10)

    plt.show()

# If you had other plotting functions, they would also go here.
# e.g., def plot_pie_chart(predictions): ...