import serial
import time
import argparse
import numpy as np
import logging

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class GRBLGCodeStreamer:
    """
    Implements the character-counting streaming protocol described
    [here](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Interface), based on
    example code from
    [here](https://github.com/gnea/grbl/blob/master/doc/script/stream.py).

    Enabling check mode is a "dry run" mode where commands are received
    and parsed on the Grbl side but not executed.
    """

    RX_BUFFER_SIZE = 128
    BAUD_RATE = 115200

    def __init__(self, port: str, check_mode: bool = False, verbose: bool = False):
        self.verbose = verbose
        self.check_mode = check_mode
        try:
            self.ser = serial.Serial(port, self.BAUD_RATE)
        except:
            error_str = f"Couldn't open serial port {port}.\n"
            from serial.tools import list_ports

            error_str += f"Other options: {[x.device for x in list_ports.comports()]}"
            raise RuntimeError(error_str)

        self.read_buffer = ""
        self.command_queue = []
        self.send_line_count = 0
        self.acknowledged_line_count = 0
        self.sent_command_buffer = []

        LOG.info("Initialization Grbl...")
        self.ser.write(str.encode(str(chr(24)) + "\r\n"))
        self._wait_for_acknowledgement(timeout=2.0)
        self.ser.flushInput()
        # Acknowledge reset.
        self.send_setting("$X")

        # Res
        if check_mode:
            LOG.info("Enabling Grbl Check-Mode.")
            self.send_setting("$C")

    def _read_line_nonblocking(self) -> str | None:
        while self.ser.inWaiting() > 0:
            data = self.ser.read(self.ser.inWaiting()).decode("ascii")
            self.read_buffer += data
        first_newline = self.read_buffer.find("\n")
        if first_newline > 0:
            line, self.read_buffer = (
                self.read_buffer[:first_newline],
                self.read_buffer[first_newline + 1 :],
            )
            if self.verbose:
                LOG.info("<<|%s", line)
            return line
        return None

    def _write(self, data):
        if self.verbose:
            LOG.info(">>|%s", data.strip())
        self.ser.write(str.encode(data))

    def _wait_for_acknowledgement(self, timeout: float = 1.0) -> bool:
        start_time = time.time()
        while 1:
            grbl_out = self._read_line_nonblocking()
            if grbl_out is not None:
                if "error" in grbl_out:
                    LOG.error(
                        "Received error from Grbl: response [%s]", grbl_out.strip()
                    )
                    return False
                elif "ok" in grbl_out or "Grbl 1.1h" in grbl_out:
                    return True
            if time.time() - start_time >= timeout:
                LOG.error("Timed out waiting for response.")
                return False
            time.sleep(0.01)

    def _safe_to_send(self, num_chars: int) -> int:
        return (
            sum([len(x) for x in self.sent_command_buffer]) + num_chars
            < self.RX_BUFFER_SIZE
        )

    def send_setting(self, data: str, timeout: float = 1.0):
        """
        Change a setting. This is a blocking call that does not return until the setting
        is confirmed.
        """
        assert len(data) < self.RX_BUFFER_SIZE + 1
        assert (
            len(self.sent_command_buffer) == 0
        ), "Can't interleave commands and settings."
        self._write(data + "\n")
        assert self._wait_for_acknowledgement(timeout=timeout)

    def send_command(self, data: str) -> int:
        """
        Submit a gcode command. An appropriate newline will automatically be
        appended. Returns a unique command ID integer which you can use to
        query the command state.
        """
        # Make sure the command fits in the buffer.
        data = data.strip() + "\n"
        assert len(data) < self.RX_BUFFER_SIZE + 1
        self.command_queue.append(data)

    def update(self):
        """
        Performs an update, sending and receiving serial data.
        """
        while 1:
            # Process any responses.
            while self.ser.inWaiting() > 0:
                rec = self.ser.readline().decode("ascii")  # Wait for grbl response
                if self.verbose:
                    LOG.info(f"<<{rec.strip()}")
                if "ok" in rec:
                    self.acknowledged_line_count += 1
                    assert len(self.sent_command_buffer) > 0
                    self.sent_command_buffer.pop(0)
                elif "error" in rec:
                    raise RuntimeError(
                        "Received error, corresponding command [%s]",
                        self.sent_command_buffer[0],
                    )
                if "MPos" in rec:
                    mpos = rec.split('|')[1][5:]
                    self.xyz = np.array([float(x) for x in mpos.split(",")])
                if "FS" in rec:
                    fs = rec.split('|')[2][3:]
                    self.spindle = float(fs.replace('>', ',').split(",")[1])
            if len(self.command_queue) == 0:
                break

            data = self.command_queue[0]
            if not self._safe_to_send(len(data)):
                break

            self.command_queue.pop(0)
            self._write(data)
            self.sent_command_buffer.append(data)

        self._write("?")

    def wait_and_update(self, delay: float):
        start_time = time.time()
        while time.time() - start_time < delay:
            self.update()
            time.sleep(0.1)

    def all_buffers_empty(self) -> bool:
        return len(self.sent_command_buffer) == 0 and len(self.command_queue) == 0

    def update_until_done(self, timeout: float) -> bool:
        start_time = time.time()
        self.update()
        while not self.all_buffers_empty():
            self.update()
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                return False
        return True

    def send_move_command(self, x: float, y: float, s: float, feed_rate: float = 6000):
        self.send_command(f"G1 X{x} Y{y} S{s} F{feed_rate}")


def get_servo(val):
    # val on [0, 1] to a servo speed value as an int.
    pwm_period = 1 / (62500 / 1024)
    servo_min = 0.001
    servo_max = 0.0018
    out = (servo_min + (servo_max - servo_min) * val) / pwm_period * 1000
    if isinstance(out, np.ndarray):
        return out.astype(int)
    else:
        return int(out)
