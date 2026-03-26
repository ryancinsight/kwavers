import numpy as np
from kwave.utils.mapgen import make_pixel_map
r = make_pixel_map(np.array([64, 64]), shift=[0, 0])
print(f"r at (32, 32): {r[32, 32]}")
print(f"r at (31, 31): {r[31, 31]}")
print(f"r at (21, 31): {r[21, 31]}")
print(f"Center is at: {np.unravel_index(np.argmin(r), r.shape)}")
