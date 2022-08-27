import cmd
import serial
import time
import argparse


class PrinterController():
    def __init__(self, port_name):
        self.ser = serial.Serial(port_name, 115200)

    def command(self, command):
        print(f"Sending: {command}")
        self.ser.write(str.encode(command))
        if (self.ser.inWaiting() > 0):
            # read the bytes and convert from binary array to ASCII
            data_str = self.ser.read(self.ser.inWaiting()).decode('ascii')
            print(f"Recv: {data_str}")

    def autohome(self):
        self.command(self.ser, "G28\r\n")

    def set_current_location(self, x=0., y=0., z=0.):
        self.command(self.ser, f"G92 X{x} Y{y} Z{z} \r\n")

    def disable_steppers(self):
        self.command(self.ser, "M18\r\n")

    def enable_steppers(self):
        self.command(self.ser, "M17\r\n")

    def set_absolute_positioning(self):
        self.command(self.ser, "G90 \r\n")

    def set_mm_units(self):
        self.command(self.ser, "G21 \r\n")

    def move(self, x=None, y=None, z=None, speed=None):
        # x, y, z are in mm; speed is in mm/sec. We convert
        # the speed to mm/min and use as a feed rate command.
        cmd_str = "G1"
        if x is not None:
            cmd_str += f"X{x}"
        if y is not None:
            cmd_str += f"Y{y}"
        if z is not None:
            cmd_str += f"Z{z}"
        if speed is not None:
            cmd_str += f"F{speed / 60.}"
        self.command(self.ser, f"{cmd_str} \r\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Drive a connected 3D printer. Call an available command, specifying its arguments as pairs <arg name> <float value>.')

    parser.add_argument(
        'command', type=str, help=f"Python command; you'll have a `controller` to use.")
    parser.add_argument('--port', type=str, default="COM11")

    args = parser.parse_args()

    controller = PrinterController(port_name=args.port)
    eval(parser.command)
