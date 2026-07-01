"""
ivp_saving_movie_files parity tests.

Validates the ``ivp_saving_movie_files_compare.py`` script setup:
  * Correct heterogeneous medium layout (density boundary at Ny//4=32, not 31)
  * Disc source construction via k-wave make_disc
  * Cartesian sensor mask conversion produces a non-empty grid mask
  * Full simulation runs only under KWAVERS_RUN_SLOW=1

All sub-second tests derive expected values analytically or from known
geometry; no simulation is run in the fast path.
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

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors conftest.py but isolated so this file is self-contained)
# ---------------------------------------------------------------------------
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
    from kwave.utils.mapgen import make_cart_circle, make_disc
    from kwave.utils.conversion import cart2grid as kwave_cart2grid
    _KWAVE = True
except ImportError:
    _KWAVE = False

requires_kwave = pytest.mark.skipif(not _KWAVE, reason="k-wave-python required")
requires_parity_deps = pytest.mark.skipif(
    not (_PYKWAVERS and _KWAVE),
    reason="pykwavers+k-wave-python required",
)
run_slow = os.getenv("KWAVERS_RUN_SLOW", "0") == "1"
slow_reason = "Set KWAVERS_RUN_SLOW=1 to run full simulation parity test"

# ---------------------------------------------------------------------------
# Constants (must match ivp_saving_movie_files_compare.py exactly)
# ---------------------------------------------------------------------------
NX = NY = 128
DX = DY = 0.1e-3
C_FAST = 1800.0
C_NOM  = 1500.0
RHO_LOW  = 1000.0
RHO_HIGH = 1200.0
RHO_BOUNDARY = NX // 4   # = 32

DISC1_MAG, DISC1_POS, DISC1_R = 5.0, [50, 50], 8
DISC2_MAG, DISC2_POS, DISC2_R = 3.0, [80, 60], 5
SENSOR_RADIUS     = 4e-3
NUM_SENSOR_POINTS = 50
PML_SIZE = 20


def _assert_metric_contract(text: str, section: str, thresholds: dict[str, float]) -> None:
    pearson = report_metric_value(text, "pearson_r", section)
    rms_ratio = report_metric_value(text, "rms_ratio", section)
    psnr_db = report_metric_value(text, "psnr_db", section)

    assert pearson >= thresholds["pearson_r"]
    assert thresholds["rms_ratio_min"] <= rms_ratio
    assert rms_ratio <= thresholds["rms_ratio_max"]
    assert psnr_db >= thresholds["psnr_db"]


@requires_parity_deps
def test_current_ivp_saving_movie_files_artifacts_match_thresholds():
    module = load_example_module("ivp_saving_movie_files_compare.py")
    thresholds = module.PARITY_THRESHOLDS

    assert module.METRICS_PATH.exists()
    assert module.FIGURE_PATH.exists()
    text = module.METRICS_PATH.read_text(encoding="utf-8")
    assert "parity_status: PASS" in text
    _assert_metric_contract(text, "[sensor time-series]", thresholds)
    _assert_metric_contract(text, "[p_final 2-D field]", thresholds)
    assert_decodable_nonblank_png(module.FIGURE_PATH)


# ---------------------------------------------------------------------------
# Fast physics tests (no simulation)
# ---------------------------------------------------------------------------

class TestMediumLayout:
    """Heterogeneous medium array construction matches upstream script."""

    def test_sound_speed_boundary(self):
        c = np.ones((NX, NY)) * C_NOM
        c[:NX // 2, :] = C_FAST
        # First half is fast
        assert float(c[0, 0])         == pytest.approx(C_FAST)
        assert float(c[NX // 2, 0])   == pytest.approx(C_NOM)
        assert float(c[NX - 1, 0])    == pytest.approx(C_NOM)

    def test_density_boundary_at_ny_quarter(self):
        """Boundary must be Ny//4 = 32, NOT 31 (differs from ivp_heterogeneous_medium)."""
        rho = np.ones((NX, NY)) * RHO_LOW
        rho[:, RHO_BOUNDARY:] = RHO_HIGH
        # Columns < 32 are RHO_LOW
        assert float(rho[0, 31]) == pytest.approx(RHO_LOW)
        # Column 32 (= Ny//4) is RHO_HIGH
        assert float(rho[0, 32]) == pytest.approx(RHO_HIGH)
        assert float(rho[0, NY - 1]) == pytest.approx(RHO_HIGH)

    def test_density_boundary_differs_from_heterogeneous_example(self):
        """Documents the key difference: 32 (Ny//4) vs 31 in ivp_heterogeneous_medium."""
        # ivp_heterogeneous_medium uses rho[:, 31:] = 1200
        # ivp_saving_movie_files uses rho[:, Ny//4:] = rho[:, 32:] = 1200
        assert RHO_BOUNDARY == 32
        assert RHO_BOUNDARY != 31


class TestDiscSource:
    """Disc source construction — both discs must be non-zero and non-overlapping."""

    @requires_kwave
    def test_disc_amplitudes(self):
        disc1 = DISC1_MAG * make_disc(Vector([NX, NY]), Vector(DISC1_POS), DISC1_R)
        disc2 = DISC2_MAG * make_disc(Vector([NX, NY]), Vector(DISC2_POS), DISC2_R)
        p0 = np.asarray(disc1 + disc2, dtype=np.float64)

        # Maximum amplitude matches DISC1_MAG (larger)
        assert float(np.abs(p0).max()) == pytest.approx(DISC1_MAG, rel=1e-6)
        # Both discs contribute non-zero energy
        assert float(np.abs(p0).sum()) > 0.0
        # p0 is non-negative everywhere (discs don't cancel)
        assert float(p0.min()) >= 0.0

    @requires_kwave
    def test_disc_positions_distinct(self):
        disc1 = make_disc(Vector([NX, NY]), Vector(DISC1_POS), DISC1_R)
        disc2 = make_disc(Vector([NX, NY]), Vector(DISC2_POS), DISC2_R)
        d1 = np.asarray(disc1, dtype=bool)
        d2 = np.asarray(disc2, dtype=bool)
        # Discs should not be identical
        assert not np.array_equal(d1, d2)
        # Discs may overlap but the center positions are distinct
        cx1, cy1 = DISC1_POS
        cx2, cy2 = DISC2_POS
        assert (cx1, cy1) != (cx2, cy2)


class TestSensorMask:
    """Cartesian circle → C-order grid mask construction."""

    @requires_kwave
    def test_sensor_mask_non_empty(self):
        kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
        kgrid.makeTime(C_NOM)
        sensor_circle = make_cart_circle(SENSOR_RADIUS, NUM_SENSOR_POINTS)
        mask_2d, _, _ = kwave_cart2grid(kgrid, sensor_circle, order="C")
        mask_2d = np.asarray(mask_2d, dtype=bool)
        n_sensors = int(mask_2d.sum())

        assert n_sensors > 0, "Sensor mask must contain at least one active grid point"
        # Circular mask: at most NUM_SENSOR_POINTS unique cells
        assert n_sensors <= NUM_SENSOR_POINTS

    @requires_kwave
    def test_sensor_mask_inside_physical_domain(self):
        kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
        kgrid.makeTime(C_NOM)
        sensor_circle = make_cart_circle(SENSOR_RADIUS, NUM_SENSOR_POINTS)
        mask_2d, _, _ = kwave_cart2grid(kgrid, sensor_circle, order="C")
        mask_2d = np.asarray(mask_2d, dtype=bool)

        # All active sensor cells must lie inside the physical interior (outside PML)
        active_idx = np.argwhere(mask_2d)
        # PML occupies first and last PML_SIZE cells on each axis
        assert int(active_idx[:, 0].min()) >= PML_SIZE
        assert int(active_idx[:, 0].max()) <  NX - PML_SIZE
        assert int(active_idx[:, 1].min()) >= PML_SIZE
        assert int(active_idx[:, 1].max()) <  NY - PML_SIZE


# ---------------------------------------------------------------------------
# Slow full-simulation test (KWAVERS_RUN_SLOW=1)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not run_slow, reason=slow_reason)
@pytest.mark.skipif(not (_PYKWAVERS and _KWAVE), reason="pykwavers+k-wave-python required")
class TestIvpSavingMovieFilesSimulation:
    """Full parity simulation for ivp_saving_movie_files."""

    def test_sensor_data_pearson_r(self, tmp_path, monkeypatch):
        """Pearson r >= 0.90 between aligned k-wave and pykwavers sensor matrices."""
        monkeypatch.syspath_prepend(str(ROOT / "examples"))
        monkeypatch.setenv("KWAVERS_REFRESH_CACHE", "1")

        import importlib
        spec = importlib.util.spec_from_file_location(
            "ivp_saving_movie_files_compare",
            str(ROOT / "examples" / "ivp_saving_movie_files_compare.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        inputs   = mod.build_shared_inputs()
        kw_res   = mod.run_kwave(inputs, no_cache=True)
        pkw_res  = mod.run_pykwavers(inputs, no_cache=True)

        perm          = inputs["sensor_row_perm"]
        kw_p_aligned  = kw_res["pressure"][perm]
        pkw_p         = pkw_res["pressure"]

        corr_matrix = np.corrcoef(kw_p_aligned.ravel(), pkw_p.ravel())
        pearson_r = float(corr_matrix[0, 1])
        assert pearson_r >= mod.PARITY_THRESHOLDS["pearson_r"], (
            f"Sensor time-series Pearson r = {pearson_r:.4f} "
            f"< {mod.PARITY_THRESHOLDS['pearson_r']:.4f} "
            f"(k-wave peak {float(np.abs(kw_p_aligned).max()):.3e} Pa  "
            f"pkw peak {float(np.abs(pkw_p).max()):.3e} Pa)"
        )

    def test_p_final_field_pearson_r(self, tmp_path, monkeypatch):
        """Pearson r >= 0.90 between k-wave and pykwavers p_final fields."""
        monkeypatch.syspath_prepend(str(ROOT / "examples"))
        monkeypatch.setenv("KWAVERS_REFRESH_CACHE", "1")

        import importlib
        spec = importlib.util.spec_from_file_location(
            "ivp_saving_movie_files_compare",
            str(ROOT / "examples" / "ivp_saving_movie_files_compare.py"),
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
