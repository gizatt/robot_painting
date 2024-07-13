import cmd
import serial
import time
import argparse
import sys
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=str)
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, 115200)
    except:
        print(f"Couldn't open serial port {args.port}.")
        from serial.tools import list_ports
        print(f"Other options: {[x.device for x in list_ports.comports()]}")
        sys.exit(-1)

    print(f"Connected to {args.port}")

    while 1:
        data = input(">>|") +  "\n"
        ser.write(str.encode(data))
        time.sleep(0.5)

        if (ser.inWaiting() > 0):
            # read the bytes and convert from binary array to ASCII
            data_str = ser.read(ser.inWaiting()).decode('ascii')
            print(f"<<|{data_str.strip()}")
