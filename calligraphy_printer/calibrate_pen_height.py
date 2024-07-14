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
    for k in range(10):
        interface.update()


    while 1:
        data = input()
        try:
            height = float(data)
            cmd = f"G1 X0 Y0 F6000 S{get_servo(height)}"
            print(cmd)
            interface.send_command(cmd)
        except ValueError:
            print("Bad input.")
        interface.update()