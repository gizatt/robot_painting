import cmd
import serial
import time
import argparse
import sys


class PrinterController():
    def __init__(self, port_name):
        try:
            self.ser = serial.Serial(port_name, 115200)
            # Need to wait for the printer to turn on.
            time.sleep(3.)
        except serial.serialutil.SerialException:
            print(f"Couldn't open serial port {port_name}.")
            from serial.tools import list_ports
            print(f"Other options: {[x.device for x in list_ports.comports()]}")
            sys.exit(-1)

    def command(self, command, timeout=30.):
        print(f"Sending: {command}")
        self.ser.write(str.encode(command))
        return self.wait_for_ack(timeout=timeout)

    def wait_for_ack(self, timeout=30.):
        done = False
        start_time = time.time()
        while not done and (time.time() - start_time) <= timeout:
            line = self.ser.readline()
            print(f"Waiting for ack: recv: {line}")
            if line == b'ok\n':
                done = True
            time.sleep(0.01)
            print(time.time() - start_time)
        return done

    def print(self):
        if (self.ser.inWaiting() > 0):
            # read the bytes and convert from binary array to ASCII
            data_str = self.ser.read(self.ser.inWaiting()).decode('ascii')
            print(f"Recv: {data_str}")

    def autohome(self):
        self.command("G28\r\n", timeout=30.)

    def set_current_location(self, x=0., y=0., z=0.):
        self.command(f"G92 X{x} Y{y} Z{z} \r\n", timeout=1.)

    def disable_steppers(self):
        self.command("M18\r\n", timeout=1.)

    def enable_steppers(self):
        self.command("M17\r\n", timeout=1.)

    def set_absolute_positioning(self):
        self.command("G90 \r\n", timeout=1.)

    def set_mm_units(self):
        self.command("G21 \r\n", timeout=1.)

    def move(self, x=None, y=None, z=None, speed=None):
        # x, y, z are in mm; speed is in mm/sec. We convert
        # the speed to mm/min and use as a feed rate command.
        cmd_str = "G1"
        if x is not None:
            cmd_str += f" X{x}"
        if y is not None:
            cmd_str += f" Y{y}"
        if z is not None:
            cmd_str += f" Z{z}"
        if speed is not None:
            cmd_str += f" F{speed * 60.}"
        self.command(f"{cmd_str} \r\n", timeout=1.)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Drive a connected 3D printer. Call an available command, specifying its arguments as pairs <arg name> <float value>.')

    #parser.add_argument(
    #    'command', type=str, help=f"Python command; you'll have a `controller` to use.")
    parser.add_argument('--port', type=str, default="COM12")

    args = parser.parse_args()

    controller = PrinterController(port_name=args.port)
    #eval(args.command)

    controller.set_current_location(0, 0, 0)
    controller.move(x=10, y=10, speed=100)
    time.sleep(5.)