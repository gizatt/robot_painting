# Prototype 1: UNET, taking stroke parameterizing input.

# Huang 2019:  B&W stroke-predicting FCN.

Huang, Z., Heng, W., Zhou, S.: Learning to paint with model-based deep
reinforcement learning. In: Proceedings of the IEEE International Conference on
Computer Vision. pp. 8709–8718 (2019)

Input: Flat vector of `n_inputs` parameterizing a stroke.
Output: 128x128 single-channel stroke image.

1) Takes a flat vector of `n_inputs`.
2) Passes through a 4-layer RELU FCN which gradually the input to 4096 elements. 
3) Reshapes 4096 elements to (16x16) im with 16 channels.
4) Alternates 3-pixel-wide 2D convolutions (followed by RELU) that halve the
   number of channels each time, folllowed by resolution-doubling pixel shuffle
   layers. Repeats 3x to get to 128x128 single-channel output.

```
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 128, 128)
```