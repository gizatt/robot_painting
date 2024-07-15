import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from svgpathtools import svg2paths2, wsvg

def draw_svg(paths, attributes, svg_attributes, block: bool = True):
    plt.figure()
    cmap = plt.get_cmap("jet")
    all_points = []
    colors = []
    for k, (path, attribs) in enumerate(zip(paths, attributes)):
        #print(path, " with style ", attribs["style"])
        path_length = path.length()
        # Length is in mm; have roughly a point per mm.
        N_points = math.ceil(path_length)
        ts = np.linspace(0, 1, N_points, endpoint=True)
        points = []
        for t in ts:
            point = path.point(t)
            points.append((point.real, point.imag))
        points = np.array(points)
        all_points.append(points)

        color = cmap(float(k) / len(paths))
        plt.scatter(
            points[:, 0], points[:, 1], color=color
        )
    plt.show(block=block)

if __name__ == "__main__":
    draw_svg(*svg2paths2('eevee.svg'))