import imageio
import numpy as np
import matplotlib.pyplot as plt
import skfmm
from scipy import ndimage

# Load image
im = imageio.imread("test.png")
ima = np.array(im)

# Differentiate the inside / outside region
phi = np.int64(np.any(ima[:, :, :3], axis=2))
phi = np.where(phi, 0, -1) + 0.5

# Compute signed distance
# dx = cell(pixel) size
sd = skfmm.distance(phi, dx=1)

gx, gy = np.gradient(sd)

# Find local mins of SDF = where we should put strokes.
laplace_filter = ndimage.laplace(sd)
max = np.max(laplace_filter)
print(f"2nd deriv peaks at {max}")
lines = laplace_filter >= max * 0.5

plt.subplot(3, 1, 1)
plt.imshow(laplace_filter)
plt.colorbar()
plt.subplot(3, 1, 2)
plt.imshow(lines)
plt.show()
