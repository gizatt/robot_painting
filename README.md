# Image Painting

I want to make my robot paint pictures.

## Approach / TODOs:

My first stab ought to look like:

- Write a black-box process model that, given a partial painting and
a new action description, returns the updated painting.
- Write a system to estimate the process model given a bunch of before/after
samples with their corresponding actions. (First pass can have no history,
but really, it should depend on the history... so maybe an LSTM model...)
- Write a trajopt formulation and run it as N-step MPC to try to paint
target images.

"Action" description: put brush down at a given coordinate (x, y, in image
coordinates) with a given color (indexed into a preprogrammed color set)
and a given force. Move it, with linear interp in x, y, and force, to a
another location (at a fixed a-priori tip speed).

Calculating cost, so MPC knows what to go downhill on: to start with,
discretize the colors of the target image to their nearest neighbors in the
color set, and then measure pixelwise error. An alternative could be to
learn an encoding of all paintings with a VAE, and map the partial image and
target image into that feature space, and take distance there, and take actions
to decrease that distance. Might be cool as an extension.

The black box model, as a feed-forward Python process (or as reality with
some trappings to do image processing to get down to a picture of the
current picture frame), takes exactly those inputs.


## Some references

[Deepmind SPIRAL](https://deepmind.com/documents/183/SPIRAL.pdf) tackles
this with a discriminatively-trained RL agent, that's trained in a
program-synthesis sort of way. (The agent learns an interative update
rule that takes the previous output and learns what next input to
provide to make the image closer to a target.)