from copy import deepcopy
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
import skimage
import scipy.ndimage
import cv2

from painting_robot_interface import (
    PainterRobotLocalMoveInput,
    PainterRobotLiftAndMoveInput,
    PainterRobotGetPaintInput,
    PainterRobotInput,
    PainterRobotState
)
from pytorch_planar_scene_drawing import (
    draw_sprites_at_poses,
    torch_images_to_numpy,
    numpy_images_to_torch
)

device = torch.device('cpu')


sim_color_options = {
    "clean": torch.tensor([0.0, 0.0, 0.0]),
    "red": torch.tensor([1.0, 0.0, 0.0]),
    "blue": torch.tensor([0.0, 1.0, 0.0]),
    "green": torch.tensor([0.0, 0.0, 1.0])
}


class SimState():
    def __init__(self):
        self.paint_on_brush = 0
        self.last_t = 0


def draw_circle_into_array(image, row, col, radius, color, copy=False):
    rr, cc = skimage.draw.circle(int(np.round(row)), int(np.round(col)),
                                 int(np.round(radius)), image.shape)
    if copy:
        ret_image = image.copy()
    else:
        ret_image = image
    for k in range(3):
        ret_image[rr, cc, k] = color[k]
    return ret_image


def secret_painting_process_model(
        t, painter_input, current_image, painter_state, sim_state):
    '''
    t: double
    painter_input: an instance of a subclass of PainterRobotInput
    current_image: a 3 x X x Y pixel image as a torch tensor.
    current_state: an instance of PainterRobotState
    '''
    assert isinstance(painter_input, PainterRobotInput)
    assert isinstance(current_image, torch.Tensor)
    assert current_image.dim() == 3
    assert current_image.shape[0] == 3

    new_painter_state = deepcopy(painter_state)
    new_image = current_image.clone()
    new_sim_state = deepcopy(sim_state)
    new_sim_state.last_t = t
    dt = t - sim_state.last_t

    if isinstance(painter_input, PainterRobotLocalMoveInput):
        # Build a sprite representing the application of that input.
        local_effect_area_size = 100
        sprite = np.zeros((local_effect_area_size, local_effect_area_size, 3))
        step_size = 1.
        for k in np.arange(0., painter_input.move_amount, step_size):
            sprite += draw_circle_into_array(
                sprite*0, 
                local_effect_area_size/2 + k*painter_input.move_direction[1],
                local_effect_area_size/2 + k*painter_input.move_direction[0],
                painter_input.tip_force*10 + np.random.randn(1)*2,
                sim_color_options[painter_state.last_color].numpy()*(1.0 - np.abs(np.random.randn(3))*0.1),
                copy=False) * new_sim_state.paint_on_brush
            new_sim_state.paint_on_brush *= 0.99
        sprite = np.clip(sprite, 0., 1.)
        # add some streaks
        burn_mask = np.abs(np.random.randn(25, 25))*0.025
        burn_mask = cv2.resize(burn_mask, dsize=(local_effect_area_size, local_effect_area_size), interpolation=cv2.INTER_NEAREST)
        for k in np.arange(0., painter_input.move_amount, step_size):
            sprite -= np.tile(np.expand_dims(scipy.ndimage.affine_transform(
                burn_mask, np.array([[1., 0., k*painter_input.move_direction[1]],
                                     [0., 1., k*painter_input.move_direction[0]]]),
                mode="wrap"), -1), [1, 1, 3])

        sprite = np.clip(sprite, 0., 1.)
        sprite = numpy_images_to_torch([sprite]).type(current_image.dtype)

        # Remove some paint from the brush
        sprite_pose = torch.tensor(
                [new_painter_state.tip_position[0],
                 new_painter_state.tip_position[1],
                 0.]
            ).unsqueeze(0)

        new_image = new_image + \
            draw_sprites_at_poses(
                sprite_pose, local_effect_area_size, local_effect_area_size,
                new_image.shape[1], new_image.shape[2],
                sprite)[0, ...]
        new_painter_state.tip_position += (
            painter_input.move_amount*painter_input.move_direction)
    else:
        painter_input.apply(t, new_painter_state)

        if isinstance(painter_input, PainterRobotGetPaintInput):
            assert(new_painter_state.last_color in sim_color_options.keys())
            new_sim_state.paint_on_brush = 1.

    return new_image, new_painter_state, new_sim_state


if __name__ == "__main__":
    canvas_x = 640
    canvas_y = 480
    current_image = torch.zeros(3, canvas_x, canvas_y)
    current_painter_state = PainterRobotState(
        tip_position=torch.Tensor([0, 0]),
        last_color="none",
        last_got_color_time=0)
    t = 0

    current_sim_state = SimState()

    last_got_paint_time = -100
    get_paint_input = PainterRobotGetPaintInput("red")

    plt.figure()
    while (t < 5.):
        if (t - last_got_paint_time > 1.0):
            painter_input = get_paint_input
            last_got_paint_time = t
        else:
            painter_input = PainterRobotLocalMoveInput(
                move_direction=torch.Tensor([np.cos(t), np.sin(t)]),
                move_amount=25,
                tip_force=1.)

        current_image, current_painter_state, current_sim_state = \
            secret_painting_process_model(
                t, painter_input, current_image,
                current_painter_state, current_sim_state)

        t += 0.1
        print t, current_painter_state.tip_position

    plt.imshow(torch_images_to_numpy(current_image.unsqueeze(0))[0])
    plt.show()