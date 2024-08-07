'''
    Based strongly on https://github.com/hzwer/ICCV2019-LearningToPaint/blob/master/baseline/train_renderer.py.
'''
import cv2
import os
import datetime
import cv2
import torch
import numpy as np
import imageio

import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from stroke_model_huang_2019 import Huang2019FCN
from stroke_data_generation import draw_brushstroke
from stroke_sampling import make_random_spline_unit_parameters, make_spline_from_unit_parameter_vector, SplineSamplingParams
from background_image_loader import BackgroundImageLoader

import torch.optim as optim

IMG_SIZE = 128 # Hardcoded by Huang 2019 stroke model.
BRUSH_SIZE = np.array([8, 8], dtype=np.int32)  # This needs to be even

def draw_stroke_from_spline_unit_params(img: np.ndarray, q0_unit: np.ndarray, v_unit: np.ndarray, sampling_params: SplineSamplingParams, brush: np.ndarray) -> np.ndarray:
    spline = make_spline_from_unit_parameter_vector(q0_unit, v_unit, sampling_params)
    return draw_brushstroke(img, spline, color=np.array([0., 0., 0.]), N_samples=64, brush=brush, brush_opacity=1., interp_type="naive")

def prioritized_loss(output, target):
    weighting = 0.5 + (1. - target)
    loss = torch.mean((output - target)**2 * weighting)
    return loss

def train():
    sampling_parameters = SplineSamplingParams()
    N_KNOTS = 5
    N_PARAMS = N_KNOTS * 3 + 2

    run_dir = os.path.split(__file__)[0]
    brush = imageio.imread(os.path.join(run_dir, "tests/data/test_brush.png"))
    brush = brush[:, :, 3] / 255.
    brush = cv2.resize(brush, BRUSH_SIZE,
                    interpolation=cv2.INTER_LINEAR)

    criterion = prioritized_loss # nn.MSELoss()
    net = Huang2019FCN(n_inputs=N_PARAMS)
    optimizer = optim.Adam(net.parameters(), lr=3e-6)
    batch_size = 128

    use_cuda = torch.cuda.is_available()
    assert use_cuda
    step = 0

    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter(os.path.join(run_dir, f"training_logs/{timestamp_str}/"))

    def save_model():
        if use_cuda:
            net.cpu()
        torch.save(net.state_dict(), os.path.join(run_dir, "trained_models/stroke_model_huang_2019.pkl"))
        if use_cuda:
            net.cuda()


    def load_weights():
        pretrained_dict = torch.load(os.path.join(run_dir,"trained_models/stroke_model_huang_2019.pkl"))
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)


    load_weights()
    while step < 50001:
        net.train()
        train_batch = []
        ground_truth = []
        for i in range(batch_size):
            q0_unit, v_unit = make_random_spline_unit_parameters(N_KNOTS)
            train_batch.append( np.r_[q0_unit, v_unit.flatten()] )
            # NOTE(gizatt) Huang model is just 128x128 output, so no color channel...
            img = np.ones((IMG_SIZE, IMG_SIZE, 3))
            ground_truth.append(draw_stroke_from_spline_unit_params(img, q0_unit, v_unit, sampling_parameters, brush)[:, :, 0])

        train_batch = torch.tensor(np.array(train_batch)).float()
        ground_truth = torch.tensor(np.array(ground_truth)).float()
        if use_cuda:
            net = net.cuda()
            train_batch = train_batch.cuda()
            ground_truth = ground_truth.cuda()
        gen = net(train_batch)
        optimizer.zero_grad()
        loss = criterion(gen, ground_truth)
        loss.backward()
        optimizer.step()
        print(step, loss.item())
        if step < 10000:
            lr = 1e-4
        elif step < 30000:
            lr = 1e-5
        else:
            lr = 1e-6
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        writer.add_scalar("train/loss", loss.item(), step)
        if step % 200 == 0:
            net.eval()
            gen = net(train_batch)
            loss = criterion(gen, ground_truth)
            writer.add_scalar("val/loss", loss.item(), step)
            for i in range(32):
                G = gen[i].cpu().data.numpy()
                GT = ground_truth[i].cpu().data.numpy()
                # Pad out to RGB images
                G = np.stack([G]*3, axis=0)
                writer.add_image("train/gen{}.png".format(i), G, step)
                GT = np.stack([GT]*3, axis=0)
                writer.add_image("train/ground_truth{}.png".format(i), GT, step)
        if step % 1000 == 0 and step > 0:
            save_model()
        step += 1
        

if __name__ == "__main__":
    train()