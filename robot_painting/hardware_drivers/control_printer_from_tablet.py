import numpy as np
import time
import serial

import pyglet
import pyglet.gl

from printer_controller import PrinterController

class ArrayImage:
    """Dynamic pyglet image of a numpy uint8 NxN array of float (0-to-1).

    Modified from an answer to https://stackoverflow.com/questions/9035712/numpy-array-is-shown-incorrect-with-pyglet."""

    def __init__(self, array):
        assert len(np.shape(array)) == 2
        assert array.dtype == np.uint8

        self._array = array

        self._tex_data = (pyglet.gl.GLubyte *
                          self._array.size).from_buffer(self._array)

        format_size = 1
        bytes_per_channel = 1
        self.pitch = array.shape[1] * format_size * bytes_per_channel
        self.image = pyglet.image.ImageData(
            array.shape[1], array.shape[0], "I", self._tex_data)
        self._update_image()

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, data):
        self._array[:, :] = data[:, :]
        self.update()

    def _update_image(self):
        self.image.set_data("I", self.pitch, self._tex_data)

    def update(self):
        self._update_image()


class CanvasManager():

    def __init__(self, window):
        self.window = window

        # Printing details
        self.draw_area_size_mm = np.array([150, 100])
        self.lift_height_mm = 8.
        self.starting_height = 5.
        self.draw_height_mm = 3.
        try:
            self.controller = PrinterController(port_name="COM12")
            self.controller.disable_software_endstops()
            # Printer won't allow itself to go below 0 z, so offset our coordinates up.
            self.controller.set_current_location(0., 0., self.starting_height)
            time.sleep(1.)
            self.controller.move(0, 0, self.starting_height, speed=100) # Gets steppers turned on.
        except serial.serialutil.SerialException:
            print("Couldn't get printer!")
            self.controller = None

        self.draw_area_size = np.array([1500, 1000])
        self.draw_area_pos = np.array([0, 0])
        window.set_size(*self.draw_area_size)
        window.set_minimum_size(*self.draw_area_size)

        # Need transpose to deal with numpy being row-column and
        # other coords being x (horizontal) then y (vertical).
        canvas_arr = np.ones((self.draw_area_size[1], self.draw_area_size[0]), dtype=np.uint8)*255
        self.canvas_img = ArrayImage(canvas_arr)
        self.mouse_xy = np.zeros(2)
        self.last_mouse_xy = np.zeros(2)
        self.mouse_state = False
        self.last_mouse_state = False

        @window.event
        def on_mouse_motion(x, y, dx, dy):
            self.mouse_xy[:] = (x, y)

        @window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            self.mouse_xy[:] = (x, y)

        @window.event
        def on_mouse_press(x, y, button, modifiers):
            self.mouse_state = True

        @window.event
        def on_mouse_release(x, y, button, modifiers):
            self.mouse_state = False

        pyglet.clock.schedule(self.on_frame)
        pyglet.clock.schedule_interval(self.on_control_tick, 0.02)
        
    def __del__(self):
        self.rehome()

    def rehome(self):
        if self.controller:
            # Put print head back at origin.
            self.controller.move(0, 0, self.starting_height, speed=1000)
            print("Waiting for final move to finish...")
            time.sleep(5.0)

    def on_control_tick(self, dt):
        if self.controller:
            if self.mouse_state:
                z = self.draw_height_mm
            else:
                z = self.lift_height_mm
            xy = np.clip( (self.mouse_xy - self.draw_area_pos) / self.draw_area_size, 0., 1.) * self.draw_area_size_mm
            print(f"Command xyz: {xy}, {z}")
            self.controller.move(x=xy[0], y=xy[1], z=z, speed=100.)

    def on_frame(self, dt):
        if self.mouse_state and self.last_mouse_state:
            # Fill a line from the last mouse location to the current location
            # Ultra-inefficient and lazy... just need some vis.
            dist = np.linalg.norm(self.mouse_xy - self.last_mouse_xy)
            r = 2  # draw radius
            for t in [0, ] + np.arange(0., dist, 2.).tolist():
                t_norm = t / (dist + 1E-6)
                x, y = self.mouse_xy * t_norm + \
                    self.last_mouse_xy * (1. - t_norm)
                u, v = self.get_data_coords(x, y)
                print(x, y)
                for i in range(-r, r):
                    if u + i < 0 or u + i >= self.draw_area_size[0]:
                        continue
                    for j in range(-r, r):
                        if v + j < 0 or v + j >= self.draw_area_size[1]:
                            continue
                        if i**2 + j**2 <= r**2:
                            self.canvas_img.array[v + j, u + i] = 0

        self.canvas_img.update()
        self.canvas_img.image.blit(0, 0)

        self.last_mouse_state = self.mouse_state
        self.last_mouse_xy[:] = self.mouse_xy[:]

    def get_data_coords(self, x, y):
        xy = np.array([x, y])
        return (xy - self.draw_area_pos).astype(np.int)


if __name__ == "__main__":
    window = pyglet.window.Window(resizable=True)
    
    canvas_manager = CanvasManager(window)

    pyglet.app.run()

    canvas_manager.rehome()
