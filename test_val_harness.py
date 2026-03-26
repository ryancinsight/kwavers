import numpy as np
from kwave.data import Vector
from kwave.utils.mapgen import make_disc
import pykwavers as kwa

def compare_arrays(kwave_result, kwavers_result):
    if not isinstance(kwavers_result, np.ndarray):
        kwavers_result = np.array(kwavers_result)
        
    kwave_result = np.squeeze(kwave_result)
    kwavers_result = np.squeeze(kwavers_result)
    
    if kwave_result.dtype != kwavers_result.dtype:
        if kwave_result.dtype in [np.int64, np.int32] and kwavers_result.dtype == bool:
            kwavers_result = kwavers_result.astype(kwave_result.dtype)
        elif kwavers_result.dtype in [np.int64, np.int32] and kwave_result.dtype == bool:
            kwave_result = kwave_result.astype(kwavers_result.dtype)
            
    if kwave_result.shape != kwavers_result.shape:
        if len(kwave_result.shape) == 1 and len(kwavers_result.shape) == 1:
            diff_len = abs(len(kwave_result) - len(kwavers_result))
            if diff_len == 1:
                min_len = min(len(kwave_result), len(kwavers_result))
                kwave_result = kwave_result[:min_len]
                kwavers_result = kwavers_result[:min_len]
        
    if kwave_result.shape != kwavers_result.shape:
        return False, f"Shape mismatch: k-wave {kwave_result.shape} vs kwavers {kwavers_result.shape}"
    
    diff = np.abs(kwave_result.astype(float) - kwavers_result.astype(float))
    max_diff = np.max(diff)
    if max_diff > 1e-10:
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        return False, f"max diff: {max_diff:.2e} at {max_idx}, tolerance: 1.00e-10"
        
    return True, f"max diff: {max_diff:.2e}"

params = {
    "nx": 64, "ny": 64, "dx": 0.5e-3, "center": (16e-3, 16e-3), "radius": 5.0e-3
}

kwa_grid = kwa.Grid(int(params["nx"]), int(params["ny"]), 1, params["dx"], params["dx"], params["dx"])

kw_grid_size = Vector([int(params["nx"]), int(params["ny"])])
kw_cx = int(round(params["center"][0]/params["dx"])) + 1
kw_cy = int(round(params["center"][1]/params["dx"])) + 1
kw_center = Vector([kw_cx, kw_cy])
kw_radius = int(round(params["radius"]/params["dx"]))

kw_mask = make_disc(kw_grid_size, kw_center, kw_radius)
kwa_mask = kwa.make_disc(kwa_grid, (params["center"][0], params["center"][1], 0.0), params["radius"])

passed, msg = compare_arrays(kw_mask, kwa_mask)
print(f"Passed: {passed}")
print(f"Message: {msg}")
