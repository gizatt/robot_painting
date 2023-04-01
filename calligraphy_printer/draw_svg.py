import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from svgpathtools import svg2paths2, wsvg
paths, attributes, svg_attributes = svg2paths2('path241.svg')

# Let's print out the first path object and the color it was in the SVG
# We'll see it is composed of two CubicBezier objects and, in the SVG file it
# came from, it was red
cmap = plt.get_cmap("jet")
all_points = []
colors = []
for k, (path, attribs) in enumerate(zip(paths, attributes)):
    #print(path, " with style ", attribs["style"])
    path_length = path.length()
    N_points = math.ceil(path_length)
    ts = np.linspace(0, 1, N_points, endpoint=True)
    points = []
    for t in ts:
        point = path.point(t)
        points.append((point.real, point.imag))
    points = np.array(points)
    all_points.append(points)
    colors.append(cmap(float(k) / len(paths)))

fig, ax = plt.subplots()


def update(frame):
    plt.gcf()
    scats = []
    for points, color in zip(all_points, colors):
        scats.append(plt.scatter(
            points[:frame, 0], points[:frame, 1], color=color))
    return scats


anim = FuncAnimation(fig, update, max(
    [len(x) for x in all_points]), interval=20, blit=True)
plt.show()
