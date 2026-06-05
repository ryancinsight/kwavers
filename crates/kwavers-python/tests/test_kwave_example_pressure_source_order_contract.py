"""Helper-level pressure-source ordering contracts for vendored k-wave examples.

These tests pin the source-matrix row ordering independently of the solver
runtime. They verify that the pykwavers example builders emit the same
Fortran-order active-cell matrices as k-wave-python's `get_distributed_source_signal`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_kwave


skip_kwave = os.getenv("KWAVERS_SKIP_KWAVE", "0") == "1"
run_slow = os.getenv("KWAVERS_RUN_SLOW", "0") == "1"
slow_reason = "Set KWAVERS_RUN_SLOW=1 to run slow k-wave-python tests"


def _load_module(module_name: str, file_name: str):
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    module_path = examples_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@requires_kwave
@pytest.mark.skipif(skip_kwave, reason="KWAVERS_SKIP_KWAVE=1")
@pytest.mark.skipif(not run_slow, reason=slow_reason)
class TestPressureSourceOrderingContracts:
    """Exact source-matrix order checks for the arc and linear-array examples."""

    def test_at_array_as_source_distributed_signal_matches_kwave(self):
        module = _load_module("at_array_as_source_compare", "at_array_as_source_compare.py")
        arc_utils = _load_module("array_source_utils", "array_source_utils.py")

        kgrid = module.kWaveGrid(module.GRID_POINTS, module.GRID_SPACING)
        medium = module.kWaveMedium(sound_speed=module.SOUND_SPEED, density=module.DENSITY)
        kgrid.makeTime(medium.sound_speed)

        source_signal = module._build_source_signal_matrix(kgrid)
        karray = module._build_kwave_array(module.ARC_POSITIONS)
        reference = np.asarray(karray.get_distributed_source_signal(kgrid, source_signal), dtype=np.float64)
        py_mask, py_signal, _ = arc_utils.build_pykwavers_distributed_arc_signal(
            module._build_arc_geometries(),
            module.pkw.Grid(nx=module.NX, ny=module.NY, nz=1, dx=module.DX, dy=module.DY, dz=module.DX),
            source_signal,
        )

        assert np.array_equal(
            np.squeeze(np.asarray(karray.get_array_binary_mask(kgrid), dtype=bool)),
            py_mask,
        )
        assert np.max(np.abs(reference - py_signal)) < 1.0e-7

    def test_at_linear_array_source_matrix_matches_kwave(self):
        module = _load_module("linear_array_transducer_geometry", "linear_array_transducer_geometry.py")

        from kwave.data import Vector
        from kwave.kgrid import kWaveGrid

        kgrid = kWaveGrid(Vector([module.NX, module.NY, module.NZ]), Vector([module.DX, module.DX, module.DX]))
        kgrid.makeTime(module.C0, 0.5, 35e-6)

        source_signal = module.build_source_signal_matrix(kgrid)
        karray = module.build_kwave_array(kgrid)
        reference = np.asarray(karray.get_distributed_source_signal(kgrid, source_signal), dtype=np.float64)
        py_mask, py_weighted_mask, py_signal, _, _ = module.build_pykwavers_source_matrix_and_masks(
            module.pkw.Grid(module.NX, module.NY, module.NZ, module.DX, module.DX, module.DX),
            source_signal,
        )

        assert np.array_equal(
            np.asarray(karray.get_array_binary_mask(kgrid), dtype=bool),
            py_mask,
        )
        assert np.allclose(
            np.asarray(karray.get_array_grid_weights(kgrid), dtype=np.float64),
            py_weighted_mask,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        assert np.max(np.abs(reference - py_signal)) < 1.0e-7


if __name__ == "__main__":
    pytest.main(["-v", __file__])
