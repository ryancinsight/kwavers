"""
na_optimising_performance parity tests.

Validates the ``na_optimising_performance_compare.py`` script setup:
  * Source image (EXAMPLE_source_two.bmp) exists and loads to non-zero array
  * Grid parameters are correct (256×256, dx=10mm/256)
  * Cartesian sensor mask conversion produces non-empty grid mask inside PML
  * Full simulation runs only under KWAVERS_RUN_SLOW=1

All sub-second tests derive expected values analytically; no simulation is run
in the fast path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from parity_test_utils import (
    assert_decodable_nonblank_png,
    load_example_module,
    report_metric_value,
)

ROOT = Path(__file__).resolve().parents[1]
_paths = [ROOT / "python", ROOT / "examples",
          ROOT.parent / "external" / "k-wave-python"]
for _p in _paths:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _msys2 = Path("D:/msys64/ucrt64/bin")
    if _msys2.exists():
        os.add_dll_directory(str(_msys2))

try:
    import pykwavers as pkw  # noqa: F401
    _PYKWAVERS = True
except ImportError:
    _PYKWAVERS = False

try:
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.utils.mapgen import make_cart_circle
    from kwave.utils.conversion import cart2grid as kwave_cart2grid
    from kwave.utils.io import load_image
    from kwave.utils.matrix import resize
    _KWAVE = True
except ImportError:
    _KWAVE = False

requires_kwave = pytest.mark.skipif(not _KWAVE, reason="k-wave-python required")
run_slow = os.getenv("KWAVERS_RUN_SLOW", "0") == "1"
slow_reason = "Set KWAVERS_RUN_SLOW=1 to run full simulation parity test"

# ---------------------------------------------------------------------------
# Constants (must match na_optimising_performance_compare.py exactly)
# ---------------------------------------------------------------------------
NX = NY = 256
X = Y = 10e-3
DX = X / NX
DY = Y / NY
C0  = 1500.0
RHO = 1000.0
P0_MAGNITUDE      = 2.0
SENSOR_RADIUS     = 4.5e-3
NUM_SENSOR_POINTS = 100
PML_SIZE          = 20

SOURCE_IMAGE_PATH = (
    ROOT.parents[1] / "external" / "k-wave-python" / "tests" / "EXAMPLE_source_two.bmp"
)

_IMAGE_PRESENT = SOURCE_IMAGE_PATH.exists()
requires_image = pytest.mark.skipif(not _IMAGE_PRESENT, reason="EXAMPLE_source_two.bmp not found")
requires_parity_deps = pytest.mark.skipif(
    not (_PYKWAVERS and _KWAVE and _IMAGE_PRESENT),
    reason="pykwavers+k-wave-python+EXAMPLE_source_two.bmp required",
)


def _assert_metric_contract(text: str, section: str, thresholds: dict[str, float]) -> None:
    pearson = report_metric_value(text, "pearson_r", section)
    rms_ratio = report_metric_value(text, "rms_ratio", section)
    psnr_db = report_metric_value(text, "psnr_db", section)

    assert pearson >= thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= rms_ratio
    assert rms_ratio <= thresholds["rms_ratio_max"]
    assert psnr_db >= thresholds["psnr_db"]


@requires_parity_deps
def test_current_na_optimising_performance_artifacts_match_thresholds():
    module = load_example_module("na_optimising_performance_compare.py")
    thresholds = module.PARITY_THRESHOLDS

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    text = module.METRICS_PATH.read_text(encoding="utf-8")
    assert "parity_status: PASS" in text
    _assert_metric_contract(text, "[sensor time-series]", thresholds)
    _assert_metric_contract(text, "[p_final 2-D field]", thresholds)
    assert_decodable_nonblank_png(module.FIGURE_PATH)


# ---------------------------------------------------------------------------
# Fast physics tests
# ---------------------------------------------------------------------------

class TestGridParameters:
    """Grid constants match upstream na_optimising_performance.py exactly."""

    def test_grid_size(self):
        assert NX == 256
        assert NY == 256

    def test_grid_spacing(self):
        # dx = 10mm / 256 ≈ 3.906e-5 m
        assert DX == pytest.approx(X / NX, rel=1e-12)
        assert DY == pytest.approx(Y / NY, rel=1e-12)
        assert DX == pytest.approx(3.906_25e-5, rel=1e-4)

    def test_cfl_time_step_finite_and_positive(self):
        """k-wave makeTime produces a positive time step for c=1500 m/s."""
        if not _KWAVE:
            pytest.skip("k-wave-python not available")
        kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
        kgrid.makeTime(C0)
        dt = float(kgrid.dt)
        nt = int(kgrid.Nt)
        assert dt > 0.0
        assert nt > 0
        # CFL ~ 0.3: dt ≈ 0.3 * DX / C0 ≈ 7.8e-9 s
        expected_dt = 0.3 * DX / C0
        assert dt == pytest.approx(expected_dt, rel=0.5)


class TestSourceImage:
    """EXAMPLE_source_two.bmp loads and scales correctly."""

    @requires_kwave
    @requires_image
    def test_image_loads_non_zero(self):
        raw = load_image(str(SOURCE_IMAGE_PATH), is_gray=True)
        assert raw is not None
        arr = np.asarray(raw, dtype=np.float64)
        assert arr.ndim == 2
        assert float(arr.max()) > 0.0

    @requires_kwave
    @requires_image
    def test_image_resized_to_grid(self):
        raw = load_image(str(SOURCE_IMAGE_PATH), is_gray=True)
        p0 = P0_MAGNITUDE * resize(raw, [NX, NY])
        p0 = np.asarray(p0, dtype=np.float64)
        assert p0.shape == (NX, NY)
        # Peak amplitude matches P0_MAGNITUDE (restore_max not applied yet)
        assert float(np.abs(p0).max()) == pytest.approx(P0_MAGNITUDE, rel=0.1)

    @requires_kwave
    @requires_image
    def test_smoothed_p0_peak_preserved(self):
        """kwave_smooth(restore_max=True) must keep the peak amplitude."""
        from kwave.utils.filters import smooth as kwave_smooth
        raw = load_image(str(SOURCE_IMAGE_PATH), is_gray=True)
        p0 = P0_MAGNITUDE * np.asarray(resize(raw, [NX, NY]), dtype=np.float64)
        p0_smooth = np.asarray(kwave_smooth(p0, restore_max=True), dtype=np.float64)
        # Peak is preserved to within 1%
        assert float(np.abs(p0_smooth).max()) == pytest.approx(
            float(np.abs(p0).max()), rel=0.01
        )


class TestSensorMask:
    """Cartesian circle sensor is placed inside the physical domain."""

    @requires_kwave
    def test_sensor_mask_non_empty(self):
        kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
        kgrid.makeTime(C0)
        circle = make_cart_circle(SENSOR_RADIUS, NUM_SENSOR_POINTS)
        mask_2d, _, _ = kwave_cart2grid(kgrid, circle, order="C")
        mask_2d = np.asarray(mask_2d, dtype=bool)
        n_sensors = int(mask_2d.sum())

        assert n_sensors > 0
        assert n_sensors <= NUM_SENSOR_POINTS

    @requires_kwave
    def test_sensor_inside_grid(self):
        """Sensor circle (r=4.5 mm) lands within the grid index bounds.

        Note: r=4.5mm on a 10mm domain reaches ~4.5/10 * 256 ≈ 115 cells from
        center, leaving only ~13 cells to the boundary — inside the PML halo.
        k-wave handles near-PML sensors correctly; this test verifies only that
        all active cells have valid [0, NX) indices.
        """
        kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
        kgrid.makeTime(C0)
        circle = make_cart_circle(SENSOR_RADIUS, NUM_SENSOR_POINTS)
        mask_2d, _, _ = kwave_cart2grid(kgrid, circle, order="C")
        mask_2d = np.asarray(mask_2d, dtype=bool)
        active = np.argwhere(mask_2d)

        assert int(active[:, 0].min()) >= 0, "Sensor x-index out of range"
        assert int(active[:, 0].max()) <  NX
        assert int(active[:, 1].min()) >= 0, "Sensor y-index out of range"
        assert int(active[:, 1].max()) <  NY

    @requires_kwave
    def test_sensor_radius_within_physical_grid(self):
        """Sensor radius 4.5 mm < half-domain 5 mm → fits inside grid."""
        half_domain = X / 2
        assert SENSOR_RADIUS < half_domain, (
            f"Sensor radius {SENSOR_RADIUS*1e3:.2f} mm >= half-domain "
            f"{half_domain*1e3:.2f} mm; would clip"
        )


# ---------------------------------------------------------------------------
# Slow full-simulation test (KWAVERS_RUN_SLOW=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not run_slow, reason=slow_reason)
@pytest.mark.skipif(not (_PYKWAVERS and _KWAVE and _IMAGE_PRESENT),
                    reason="pykwavers+k-wave-python+EXAMPLE_source_two.bmp required")
class TestNaOptimisingPerformanceSimulation:
    """Full parity simulation for na_optimising_performance."""

    def test_sensor_data_pearson_r(self, monkeypatch):
        """Pearson r >= 0.85 between aligned k-wave and pykwavers sensor matrices."""
        monkeypatch.setenv("KWAVERS_REFRESH_CACHE", "1")

        import importlib
        spec = importlib.util.spec_from_file_location(
            "na_optimising_performance_compare",
            str(ROOT / "examples" / "na_optimising_performance_compare.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        inputs  = mod.build_shared_inputs()
        kw_res  = mod.run_kwave(inputs, no_cache=True)
        pkw_res = mod.run_pykwavers(inputs, no_cache=True)

        perm         = inputs["sensor_row_perm"]
        kw_p_aligned = kw_res["pressure"][perm]
        pkw_p        = pkw_res["pressure"]

        corr_matrix = np.corrcoef(kw_p_aligned.ravel(), pkw_p.ravel())
        pearson_r = float(corr_matrix[0, 1])
        assert pearson_r >= mod.PARITY_THRESHOLDS["pearson_r"], (
            f"Sensor Pearson r = {pearson_r:.4f} "
            f"< {mod.PARITY_THRESHOLDS['pearson_r']:.4f} "
            f"(kw peak {float(np.abs(kw_p_aligned).max()):.3e} Pa  "
            f"pkw peak {float(np.abs(pkw_p).max()):.3e} Pa)"
        )

    def test_p_final_field_pearson_r(self, monkeypatch):
        """Pearson r >= 0.85 between k-wave and pykwavers p_final fields."""
        monkeypatch.setenv("KWAVERS_REFRESH_CACHE", "1")

        import importlib
        spec = importlib.util.spec_from_file_location(
            "na_optimising_performance_compare",
            str(ROOT / "examples" / "na_optimising_performance_compare.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        inputs  = mod.build_shared_inputs()
        kw_res  = mod.run_kwave(inputs, no_cache=True)
        pkw_res = mod.run_pykwavers(inputs, no_cache=True)

        kw_pf  = kw_res["p_final"].ravel()
        pkw_pf = pkw_res["p_final"].ravel()
        corr   = np.corrcoef(kw_pf, pkw_pf)
        pearson_r = float(corr[0, 1])
        assert pearson_r >= mod.PARITY_THRESHOLDS["pearson_r"], (
            f"p_final Pearson r = {pearson_r:.4f} "
            f"< {mod.PARITY_THRESHOLDS['pearson_r']:.4f}"
        )
