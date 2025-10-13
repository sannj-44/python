# visualization/plot_graphs.py

import matplotlib.pyplot as plt
from collections import Counter

def plot_results(predictions):
    """
    Plots pie and bar charts for the predictions list.
    """
    if not predictions:
        print("No predictions to plot.")
        return

    counts = Counter(predictions)
    labels, values = zip(*counts.items())

    # Pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Waste Category Distribution (Live Feed)')
    plt.show()

    # Bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title('Waste Classification Count')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.show()
