import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from svgpathtools import svg2paths2, wsvg
import time

from printer_controller import PrinterController

# Load, generate path info, and display in matplotlib.
paths, attributes, svg_attributes = svg2paths2('eevee.svg')
# We'll not draw points outside of this bbox.
draw_area_size_mm = np.array([200, 200])
lift_height_mm = 8.
starting_height = 5.
draw_height_mm = 4.
draw_speed_mms = 10


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

plt.show(False)

# Connect to printer
controller = PrinterController(port_name="COM12")
controller.disable_software_endstops()
# Printer won't allow itself to go below 0 z, so offset our coordinates up.
controller.set_current_location(0., 0., starting_height)
time.sleep(1.)
controller.move(0, 0, starting_height, speed=100)  # Gets steppers turned on.
controller.move(0, 0, lift_height_mm, speed=100)  # Gets steppers turned on.
time.sleep(1.)

print("Starting to draw!")

# Spool out these points, delaying appropriately based on move distances.
for points in all_points:
    # Move up and over to start.
    x0 = points[0, :]
    controller.move(x=x0[0], y=x0[1], z=lift_height_mm, wait=True)
    # Press into surface
    controller.move(x=x0[0], y=x0[1], z=draw_height_mm, wait=True)

    # Spool out all points
    for x in points:
        controller.move(x=x[0], y=x[1], z=draw_height_mm,
                        speed=draw_speed_mms, wait=True)
    xf = points[-1, :]
    # Raise
    controller.move(x=xf[0], y=xf[1], z=lift_height_mm, wait=True)

print("Done! Rehoming.")
controller.move(0, 0, starting_height, speed=1000, wait=True)
print("Sleeping after rehome...")
time.sleep(1.)
