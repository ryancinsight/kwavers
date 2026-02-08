"""
Shared pytest fixtures for pykwavers validation test suite.

Provides common grid, medium, source, and sensor configurations
used across all parity and validation tests.
"""

import numpy as np
import pytest

import pykwavers as kw

# ---------------------------------------------------------------------------
# k-wave-python availability
# ---------------------------------------------------------------------------

try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.data import Vector
    from kwave.utils.mapgen import make_disc, make_ball
    from kwave.utils.signals import tone_burst

    HAS_KWAVE = True
except ImportError:
    HAS_KWAVE = False

# Marker for tests requiring k-wave-python
requires_kwave = pytest.mark.skipif(not HAS_KWAVE, reason="k-wave-python not installed")


# ---------------------------------------------------------------------------
# Small grid fixtures (fast, for unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def grid():
    """32^3 grid with 0.1 mm spacing."""
    return kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)


@pytest.fixture
def medium():
    """Homogeneous water medium: c=1500 m/s, rho=1000 kg/m^3."""
    return kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)


@pytest.fixture
def source(grid):
    """1 MHz, 100 kPa plane wave source."""
    return kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)


@pytest.fixture
def sensor():
    """Point sensor at center of 32^3 grid with 0.1mm spacing."""
    return kw.Sensor.point(position=(0.0016, 0.0016, 0.0016))


# ---------------------------------------------------------------------------
# Medium-sized grid fixtures (for parity tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_64():
    """64^3 grid with 0.1 mm spacing."""
    return kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)


@pytest.fixture
def grid_2d():
    """64x64x1 quasi-2D grid with 0.1 mm spacing."""
    return kw.Grid(nx=64, ny=64, nz=1, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)


# ---------------------------------------------------------------------------
# Simulation parameter fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def water_params():
    """Standard water acoustic parameters."""
    return {
        "sound_speed": 1500.0,
        "density": 1000.0,
        "frequency": 1e6,
        "amplitude": 1e5,
    }


@pytest.fixture
def bone_params():
    """Cortical bone acoustic parameters."""
    return {
        "sound_speed": 3000.0,
        "density": 1850.0,
    }


@pytest.fixture
def soft_tissue_params():
    """Soft tissue acoustic parameters."""
    return {
        "sound_speed": 1540.0,
        "density": 1050.0,
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_cfl_dt(dx, sound_speed, cfl=0.3):
    """Compute CFL-stable time step."""
    return cfl * dx / sound_speed


def compute_error_metrics(reference, test):
    """Compute L2, Linf, and correlation between two signals."""
    min_len = min(len(reference.flatten()), len(test.flatten()))
    ref = reference.flatten()[:min_len].astype(np.float64)
    tst = test.flatten()[:min_len].astype(np.float64)

    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-30:
        return {"l2_error": 0.0, "linf_error": 0.0, "correlation": 1.0}

    l2_error = float(np.linalg.norm(tst - ref) / ref_norm)
    linf_error = float(np.max(np.abs(tst - ref)) / np.max(np.abs(ref))) if np.max(np.abs(ref)) > 0 else 0.0

    if np.std(ref) > 1e-30 and np.std(tst) > 1e-30:
        correlation = float(np.corrcoef(ref, tst)[0, 1])
    else:
        correlation = 1.0 if np.allclose(ref, tst) else 0.0

    return {"l2_error": l2_error, "linf_error": linf_error, "correlation": correlation}
