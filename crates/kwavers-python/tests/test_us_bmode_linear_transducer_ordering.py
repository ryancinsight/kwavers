from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def load_example_module():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    module_path = examples_dir / "us_bmode_linear_transducer_compare.py"
    sys.path.insert(0, str(examples_dir))
    spec = importlib.util.spec_from_file_location("us_bmode_linear_transducer_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_mask_uses_transducer_active_mask_ssot():
    mod = load_example_module()
    kgrid, not_transducer, input_signal = mod.build_kwave_objects()

    source_mask, ux_signals = mod.build_source_mask_and_signals(kgrid, not_transducer, input_signal)

    px, py, pz = int(mod.PML_SIZE.x), int(mod.PML_SIZE.y), int(mod.PML_SIZE.z)
    active_region = source_mask[
        px : px + int(mod.GRID_SIZE_PTS.x),
        py : py + int(mod.GRID_SIZE_PTS.y),
        pz : pz + int(mod.GRID_SIZE_PTS.z),
    ]

    assert np.array_equal(active_region, np.asarray(not_transducer.active_elements_mask, dtype=np.float64))
    assert ux_signals.shape[0] == int(np.asarray(not_transducer.active_elements_mask).sum())


if __name__ == "__main__":
    raise SystemExit("Use pytest to run this module.")
