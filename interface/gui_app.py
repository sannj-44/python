import tkinter as tk
from interface.live_console import start_console

root = tk.Tk()
root.title("Smart Waste Classifier")

tk.Label(root, text="Smart Waste Classifier").pack(pady=10)
tk.Button(root, text="Start Live Feed", command=start_console).pack(pady=10)
tk.Button(root, text="Exit", command=root.quit).pack(pady=10)

root.mainloop()
