import serial
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

from grbl_gcode_streamer import GRBLGCodeStreamer, get_servo

if __name__ == "__main__":
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=str)
    args = parser.parse_args()

    interface = GRBLGCodeStreamer(args.port, verbose=True)
    interface.send_command("G90")
    interface.send_command("G92 X0 Y0")
    interface.send_command("M3")

    dt = 0.05
    ts = np.arange(0.0, 8.0 * np.pi + 0.01, dt)
    x = np.sin(ts / 2.0) * 50.0
    y = np.sin(1.5 * ts / 2.0) * 50.0

    plt.plot(x, y)
    plt.pause(0.5)

    s = get_servo(np.sin(ts) * 0.25 + 0.6)
    
    last_xy = np.zeros(2)
    for t, x, y, s in zip(ts, x, y, s, strict=True):

        move_distance = np.linalg.norm(np.array([x, y]) - last_xy)
        # speed must be specified in mm/min, confusingly
        feed_rate = np.clip((move_distance / dt) * 60.0 * 0.25, 1, 6000)
        command = f"G1 X{x:0.2f} Y{y:0.2f} F{int(feed_rate)} S{s}"
        LOG.info("Sending iface %s", command)
        interface.send_command(command)

    start_time = time.time()
    while len(interface.command_queue) > 0:
        interface.update()

    print("Done")
    plt.show()
