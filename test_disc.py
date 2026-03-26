import numpy as np
from kwave.data import Vector
from kwave.utils.mapgen import make_disc
import pykwavers as kwa

# Params from validation
nx, ny = 64, 64
dx = 0.5e-3
center = (16e-3, 16e-3)
radius = 5e-3

# Kwavers
kwa_grid = kwa.Grid(nx, ny, 1, dx, dx, dx)
kwa_mask = kwa.make_disc(kwa_grid, (center[0], center[1], 0.0), radius)[:, :, 0]

# K-Wave
kw_grid_size = Vector([nx, ny])
kw_center = Vector([int(round(center[0]/dx)), int(round(center[1]/dx))])
kw_radius = int(round(radius/dx))
kw_mask = make_disc(kw_grid_size, kw_center, kw_radius)

diff = np.abs(kw_mask.astype(int) - kwa_mask.astype(int))
print(f"Total diff pixels: {np.sum(diff)}")

idx = np.where(diff > 0)
for i, j in zip(idx[0], idx[1]):
    dist_sq = (i - 32)**2 + (j - 32)**2
    print(f"Mismatch at ({i}, {j}): dist_sq={dist_sq} vs r^2=100 | kw={kw_mask[i,j]} kwa={kwa_mask[i,j]}")

