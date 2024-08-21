import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw
import numpy as np

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint App")

        self.canvas = Canvas(root, bg="white", width=800, height=600)
        self.canvas.pack()

        # Initialize variables
        self.old_x = None
        self.old_y = None
        self.drawn_points = []
        self.current_stroke = []
        self.image = Image.new("RGB", (800, 600), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Bind events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.root.bind("s", self.save_strokes)

    def paint(self, event):
        x, y = event.x, event.y
        pressure = 1.0  # Simulated pressure; replace with actual pressure data if available

        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, x, y, width=pressure * 5, fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.old_x, self.old_y, x, y], fill="black", width=int(pressure * 5))
            self.current_stroke.append((x, y, pressure))
        
        self.old_x = x
        self.old_y = y

    def reset(self, event):
        self.drawn_points.append(np.array(self.current_stroke))
        self.current_stroke = []
        self.old_x = None
        self.old_y = None

    def save_strokes(self, event=None):
        filename = "strokes.npz"
        print(f"Saving {len(self.drawn_points)} strokes to {filename}")
        np.savez(filename, *self.drawn_points)

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()