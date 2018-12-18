import numpy as np
import torch

class PainterRobotInput:
    def __init__(self):
        raise NotImplementedError()

    def apply(self, t, state):
        raise NotImplementedError()


class PainterRobotLocalMoveInput(PainterRobotInput):
    '''
    Stores the input for an atomic
    action of the abstract painting robot
    representing a local move as part of a
    broader stroke.
    This action is translated into
    instantaneous commands + executed on
    hardware (or in sim).

    If the robot is in the air, it'll drop to
    the surface and apply tip_force before starting
    movement. Otherwise, it'll apply tip_force movement
    and immediately move.

    Coordinates are in the drawing image space
    (in pixels).
    move_direction is a 2-dimensional unit vector.
    move_amount is a distance to move. Distances
        will probably  be capped at some relatively small distance.
    Force is in Newtons, exerted downwards.
    '''
    def __init__(self,
                 move_direction,
                 move_amount,
                 tip_force):
        self.move_direction = move_direction
        self.move_amount = move_amount
        self.tip_force = tip_force


class PainterRobotLiftAndMoveInput(PainterRobotInput):
    '''
    Stores the input for an atomic
    action of the abstract painting robot
    representing lifting the brush into the
    air and hovering over a new location.

    Coordinates are in the drawing image space
    (in pixels).
    destination is an image coordinate to end up in.

    '''
    def __init__(self, destination):
        self.destination

    def apply(self, t, state):
        state.tip_position = self.destination


class PainterRobotGetPaintInput(PainterRobotInput):
    '''
    Stores the input for an atomic
    action of the abstract painting robot
    representing getting more color on the brush.
    It will return to whatever location is was at before
    getting color. If this involved lifting from the
    surface, it won't return to the surface.
    '''

    def __init__(self, color):
        self.color = color

    def apply(self, t, state):
        state.last_color = self.color
        state.last_got_color_time = t


class PainterRobotState:
    ''' Internal state of the painter robot.'''

    def __init__(self,
                 tip_position=torch.Tensor([0, 0]),
                 last_color="clean",
                 last_got_color_time=0):
        self.tip_position = tip_position
        self.last_color = "clean"
        self.last_got_color_time = 0
