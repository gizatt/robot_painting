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

## Some references

[Deepmind SPIRAL](https://deepmind.com/documents/183/SPIRAL.pdf) tackles
this with a discriminatively-trained RL agent, that's trained in a
program-synthesis sort of way. (The agent learns an interative update
rule that takes the previous output and learns what next input to
provide to make the image closer to a target.)