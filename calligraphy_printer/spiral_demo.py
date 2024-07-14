import serial
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

from grbl_gcode_streamer import GRBLGCodeStreamer, get_servo

def run_spiral(interface):
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

def wait_and_update(interface, delay: float):
    start_time = time.time()
    while time.time() - start_time < delay:
        interface.update()
        time.sleep(0.1)

def wait_until_done(interface):
    while len(interface.sent_command_buffer) > 0 or len(interface.command_queue) > 0:
        interface.update()
        time.sleep(0.1)

def run_stroke_lines(interface):
    interface.send_command(f"G1 X0 Y0 F3000 S{get_servo(0)}")
    wait_until_done(interface)
    
    for line in range(5):
        x = line * 10
        interface.send_command(f"G1 X{x} Y0 S{get_servo(0)} F6000")
        wait_until_done(interface)
        
        stroke_len = 100.
        for t in np.linspace(0., 1., 20):
            y = stroke_len * t
            if t > 0.75:
                stroke_depth = (1 - t)*4
            elif t < 0.25:
                stroke_depth = t*4
            else:
                stroke_depth = 1.0
            stroke_depth = np.clip(stroke_depth, 0., 1.)
            stroke_depth = 0.5 + (line + 1) * 0.1 * stroke_depth
            interface.send_command(f"G1 X{x} Y{y} S{get_servo(stroke_depth)} F5000")
        
        interface.send_command(f"G1 X{x} Y{y} S{get_servo(0)} F6000")
        wait_until_done(interface)

    interface.send_command(f"G1 X0 Y0 S0 F6000")
    wait_until_done(interface)

if __name__ == "__main__":
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=str)
    args = parser.parse_args()

    interface = GRBLGCodeStreamer(args.port, verbose=True)
    interface.send_setting("$32=1")
    interface.send_command("G90")
    interface.send_command("G92 X0 Y0")
    interface.send_command("M3")

    #run_spiral(interface)
    run_stroke_lines(interface)