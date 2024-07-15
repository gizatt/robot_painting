import serial
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
from calibration import make_homog

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

from grbl_gcode_streamer import GRBLGCodeStreamer

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

def run_z_compensation_tester(interface):
    calib_data = np.load("calibration.npz", allow_pickle=True)
    im_size = calib_data["im_size"]
    robot_M_uv = calib_data["robot_M_uv"]

    interface.send_move_command(x=0, y=0, feed_rate=6000, stroke=0)
    interface.wait_and_update(1.0)

    # In UV coords
    # Go around outside
    uv = np.empty((0, 3))
    for radius in np.arange(0.05, 0.4, 0.05):

        new_uv = np.array([
            [-1.0, -1.0, 0],
            [-1.0, -1.0, 0.7],
            [1.0, -1.0, 0.7],
            [1.0, 1.0, 0.7],
            [-1.0, 1.0, 0.7],
            [-1.0, -1.0, 0.7],
            [-1.0, -1.0, 0.0],
        ]) * np.array([radius, radius, 1.]) + np.array([0.5, 0.5, 0.])
        uv = np.concatenate([uv, new_uv], axis=0)
    uv = uv.T
    uv[0, :] *= im_size[0]
    uv[1, :] *= im_size[1]
    uv[:2, :] = robot_M_uv @ make_homog(uv[:2, :])
    for x, y, s in uv.T:
        interface.send_move_command(x=x, y=y, stroke=s)
    
    interface.send_move_command(x=0, y=0, stroke=0)
    interface.update()
    interface.update_until_done(timeout=30.)
    LOG.info("All commands enqueued")

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
    run_z_compensation_tester(interface)