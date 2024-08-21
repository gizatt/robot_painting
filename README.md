# Image Painting

Ultimately, I want to make my robot paint pictures.

I'm focusing on making implementation choices that support using a 3D printer with a top-down camera and brush [brush *pen*, to start] attached to it as my "robot". That means:

1) Observations are top-down images of the painting (but only taken between brushstrokes).
2) Actions are 3D brush position and velocity, without orientation control.

## Note on code generation

I'm using this project to try out using LLMs (specifically GPT-4) to speed up my dev process by helping me write code and tests. But I want to acknowledge that it's very hard to tell if it's re-using licensed code! So while I've applied an MIT license generally (as I usually do), note there's a risk that other licenses may leak have leaked in here.

## Thoughts on approach

Grab bag of attributes I think are important:

- The input is going to be just a reference image; I'm happy to play with applying styles to it down the road, but just reproducing it will be a good first challenge.
- I'd like this to produce detailed, crisp final images, and I think an important part of doing that is having a model that
  i) Works at multiple scales, so it can foveat parts of the image that need fine detail without needing to reason about the whole image at that resolution.
  ii) Works iteratively with each stroke making the image closer to the target.
  iii) Doesn't make serious mistakes, or has enough transparency that I can tune it to be conservative in media-specific ways. e.g. brush pens or calligraphy makes certain kinds of mistakes irreversible.
- The stroke model is mostly trained using randomly-acquired data from the environment, but can get fine-tuned at runtime, as each new stroke we execute is a new datapoint. I can also use hand-coded simple models to start. The input should be a foveated image region
- The "stroke generator" is the obvious hard part. I have many ideas about a fancy version (looking multiple strokes into the future to think about multi-stroke strategies), but a greedy single-stroke optimizer seems like a fine start. The simplest form of that may be to have a differentiable stroke model that optimizes a batch of strokes and picks the best. I've tried batched gradient descent against a sprite rendering model, but the gradients are really weak -- this could be a good opportunity to try out a diffusion model or other neural optimizer.
  - **I ought to try directly learning a differentiable approximation to the reward function as my first thing, then just do gradient descent on it.**
  - It looks like I might be able to try [this paper](https://arxiv.org/pdf/2205.09991.pdf) directly -- code [here](https://github.com/jannerm/diffuser). Issue: this classifier-guided sampling strategy uses diffusion to model the trajectories -- which I don't have much  trouble generating -- but learns a *separate* model for how "good" each trajectory is w.r.t. some reward function, and cludges that into the diffusion update as an extra term by taking its gradient. I feel like that gradient is the hard part for me -- learning the update step directly (a la the core idea of diffusion) rather than taking a gradient of a learned value function feels "better" here. Diffusion techniques seem to focus on capturing distributions of trajectories that are "interesting" -- e.g. capturing human demonstrations for imitation learning. So maybe this isn't what I'm looking for.
- To keep things simpler, trajectories could be forced to be fixed-length. If you want a shorter stroke, just cram the knots together.

I'd like to use an explicit MPC-like approach for this, where the method breaks down into a few components:

1) A paintstroke execution model that faithfully captures the result of apply a paintstroke, parameterized by some discrete (color) and continuous (brush pose, velocity, motion history) attributes. 
2) A controller that 


## Block diagrams

### Overall
```

(~~~~~~~~~~~~~~~~)   (******************)
(  Target image  )   (  Current image   )
(****************)   (******************)
        |                      |
       \|/                    \|/
|----------------------------------------------------------|
|  STROKE GENERATOR                                        |    |--------------|
|    Given current state, target image, and stroke model,  |<---| STROKE MODEL |
|    selects next stroke trajectory.                       |    |--------------|
|----------------------------------------------------------|               /|\
          |                                                                 |
          |                                                                 |
         \|/                                                                |
(~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~)    |
(  TIMED TRAJECTORY OF STROKE TO EXECUTE                               )    |
(    Starts and stops at zero velocity.                                )----|
(    Printer moves a safe distance above start point, descends to it,  )    |
(    executes, and then lifts from end point.                          )    |
(~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~)    |
         |                                                                  |
         |                                                                  |
    |----------|                                                            |
    |CONTROLLER|                                                            |
    | (hw only)|                                                            |
    |----------|                                                            |
         |                                                                  |
        \|/                                                                 |
|------------------|                 (~~~~~~~~~~~~~~~~~)                    |
|   ENVIRONMENT    | --------------> ( Resulting image ) --------------------
|------------------|                 (~~~~~~~~~~~~~~~~~)

```
### Stroke model
```
(~~~~~~~~~~~~~~~~~~~~~~~~~~)
( TIMED STROKE TRAJECTORY  )
(      fixed length?       )
(~~~~~~~~~~~~~~~~~~~~~~~~~~)         (Current image)
           |                               |
          \|/                             \|/
|---------------------------------------------------------------------|
|                   STROKE MODEL                                      |
|    - Capture (possibly multi-scale) a foveated views centered at    |
|      each knot of trajectory.                                       |
|    - Encode each patch to big vector with fully connected layer.    |
|    - Pass embeddings + paired trajectory params through big network |
|      to get outputs.  (Transformer encoder?)                        |
|                                                                     |
|---------------------------------------------------------------------|
                | 
               \|/
         (Resulting image)
```
Trained on collected input/output pairs generated with random actions.

And a similar model for expected reward:
```
(~~~~~~~~~~~~~~~~~~~~~~~~~~)
( TIMED STROKE TRAJECTORY  )
(      fixed length?       )
(~~~~~~~~~~~~~~~~~~~~~~~~~~)         (Current image)      (Target image)
           |                               |                     |
          \|/                             \|/                   \|/
|---------------------------------------------------------------------|
|                   STROKE MODEL                                      |
|    - Capture (possibly multi-scale) a foveated views centered at    |
|      each knot of trajectory of both current + target im.           |
|    - Encode each patch to big vector with fully connected layer.    |
|    - Pass embeddings + paired trajectory params through big network |
|      to get outputs.  (Transformer encoder?)                        |
|                                                                     |
|---------------------------------------------------------------------|
                |
               \|/
         (Expected loss)
```
Trained on input image + random stroke (or sometimes GT stroke) + target output
image.

These feel like they should share most of their representation... maybe
the patch encoding blocks, at least. The actual transformer input seqs
are different so no sharing there.

### Stroke generator using differentiable stroke model


## Overview / Notes

`pytorch_planar_scene_drawing.py` contains basic utilities for building up images
from subparts (sprites), and is my current sandbox.


## Some references

- [SPIRAL](https://deepmind.com/documents/183/SPIRAL.pdf) and [SPIRAL++](https://learning-to-paint.github.io/) tackles this with a discriminatively-trained RL agent that learns a recurrent / stateful policy over parameterized painting actions via REINFORCE (specifically advantage actor-critic). The latter writeup in particular includes numerous tricks used to improve the results. Interestingly, they note that the discriminator loss was more informative / led to better results than a pixel-space loss. (I suppose this makes sense, in the context of my own thinking about how the discontinuous nature of pixel-space loss -- but I should test some of that thinking experimentally.)
- [StrokeNet](https://openreview.net/pdf?id=HJxwDiActX) learns an encoder(agent)/decoder(generator) architecture, where the embedded representation is a list of strokes. The generator network is first fit to approximate a real renderer, and then the encoder is trained with an auto-encoding loss. Not terribly relevant here -- they focused more on just recovering small sets of strokes from images than generating really complex big ones like I want to.
- [Neural Painters](https://arxiv.org/pdf/1904.08410.pdf) is a neat follow-up to SPIRAL with a more light-weight (at least in terms of training time) architecture and Google Colab code. It relies on approximating the non-differentiable painting process with a neural net (trained as a GAN), and then learning a policy (with some simple fully connected net, looks like) on top of that. `reiinakano`'s got an [awesome blog](https://reiinakano.com/) too.
- [Huang et al](https://arxiv.org/pdf/1903.04411.pdf) (with code [here](https://github.com/hzwer/ICCV2019-LearningToPaint)) takes a similar model-based RL approach: build a model of the painting process, and use it to train an agent with what they term model-based DDPG (but it's a weird variant -- gradients from the differentiable renderer are available, and they train the critic to taking only the state and ignore the action).  They use a frame skipping trick -- making the agent roll out a couple actions at a time to force it to predict accurately at longer horizons -- which is useful for me to keep in mind. I would need to go over this carefully to understand -- there's also a GAN loss used (rather than a pixel-space loss, for the same reason as mentioned in SPIRAL), which I guess is trained simultaneously. There's a lot going on...

## Current calibration matrix

Final calibration data: {'robot_M_uv': array([[-2.47911806e-01, -2.37978766e-03,  3.07979134e+02],
       [ 9.05840222e-04, -2.33921244e-01,  1.88412235e+02]]), 'im_size': [1000, 622], 'lb': array([307.97913358, 188.41223525]), 'ub': array([58.58709963, 43.81906158])}