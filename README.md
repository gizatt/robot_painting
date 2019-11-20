# Image Painting

Ultimately, I want to make my robot paint pictures. Having my computer paint them digitally would be a good start.

## Approach / TODOs:

Roughly ordered digital painting todos:
  - Write a very simple differentiable painting program, following the oil model that I really like used in [this work](https://github.com/ctmakro/opencv_playground).
  - Attempt to reproduce the results in that repo (e.g. try random rollouts of brushstrokes and accept the best ones). Maybe inject some gradient descent.
  - Play with different losses -- e.g. a GAN loss, a style transfer loss.
  - Try to write an N-step MPC to try to paint target images. May need significant neural components. (Calculating cost, so MPC knows what to go downhill on: to start with, discretize the colors of the target image to their nearest neighbors in the color set, and then measure pixelwise error. An alternative could be to learn an encoding of all paintings with a VAE, and map the partial image and target image into that feature space, and take distance there, and take actions to decrease that distance. Might be cool as an extension.)

Roughly ordered robot painting todos:
  - Architect a more robot-like stroke controller and stroke model, and find a way to fit it to data. (Possible target "action" description: put brush down at a given coordinate (x, y, in image coordinates) with a given color (indexed into a preprogrammed color set) and a given force. Move it, with linear interp in x, y, and force, to a  another location (at a fixed a-priori tip speed)? Seems insufficiently expressive. Mixed continuous-discrete with velocity control but also brush lifting / setting feels better...)

## Overview / Notes

`pytorch_planar_scene_drawing.py` contains basic utilities for building up images
from subparts (sprites), and is my current sandbox.


## Some references

- [SPIRAL](https://deepmind.com/documents/183/SPIRAL.pdf) and [SPIRAL++](https://learning-to-paint.github.io/) tackles this with a discriminatively-trained RL agent that learns a recurrent / stateful policy over parameterized painting actions via REINFORCE (specifically advantage actor-critic). The latter writeup in particular includes numerous tricks used to improve the results. Interestingly, they note that the discriminator loss was more informative / led to better results than a pixel-space loss. (I suppose this makes sense, in the context of my own thinking about how the discontinuous nature of pixel-space loss -- but I should test some of that thinking experimentally.)
- [StrokeNet](https://openreview.net/pdf?id=HJxwDiActX) learns an encoder(agent)/decoder(generator) architecture, where the embedded representation is a list of strokes. The generator network is first fit to approximate a real renderer, and then the encoder is trained with an auto-encoding loss. Not terribly relevant here -- they focused more on just recovering small sets of strokes from images than generating really complex big ones like I want to.
- [Neural Painters](https://arxiv.org/pdf/1904.08410.pdf) is a neat follow-up to SPIRAL with a more light-weight (at least in terms of training time) architecture and Google Colab code. It relies on approximating the non-differentiable painting process with a neural net (trained as a GAN), and then learning a policy (with some simple fully connected net, looks like) on top of that. `reiinakano`'s got an [awesome blog](https://reiinakano.com/) too.
- [Huang et al](https://arxiv.org/pdf/1903.04411.pdf) (with code [here](https://github.com/hzwer/ICCV2019-LearningToPaint)) takes a similar model-based RL approach: build a model of the painting process, and use it to train an agent with what they term model-based DDPG (but it's a weird variant -- gradients from the differentiable renderer are available, and they train the critic to taking only the state and ignore the action).  They use a frame skipping trick -- making the agent roll out a couple actions at a time to force it to predict accurately at longer horizons -- which is useful for me to keep in mind. I would need to go over this carefully to understand -- there's also a GAN loss used (rather than a pixel-space loss, for the same reason as mentioned in SPIRAL), which I guess is trained simultaneously. There's a lot going on...
