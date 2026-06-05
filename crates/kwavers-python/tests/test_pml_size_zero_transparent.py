"""Regression test: pml_size=0 yields a transparent (non-absorbing) boundary.

Background
----------
The `Simulation` constructor stored `pml_size` separately from `pml_config`, and
the run path reads only `pml_config` — so `pml_size` was silently ignored and a
default ~20-cell absorbing PML was always applied. This test pins the corrected
behavior:

* `pml_size=0`  → transparent boundary → a 1-D standing-wave initial-value
  problem is *sustained* (modal amplitude follows cos(ωt), including the sign
  reversal at ωt = π).
* `pml_size>0`  → absorbing PML → the standing wave's counter-propagating
  constituents are absorbed and the modal amplitude decays.

The spectral PSTD operator is intrinsically periodic, so a transparent boundary
makes the column a lossless resonator that reproduces p₀ sin(kx) cos(ωt).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pykwavers as kw

C0 = 1481.0
RHO0 = 998.0


def _standing_wave_modal_amplitude(pml_size, n_periods=1.0):
    """Run a 1-D standing-wave IVP and return (omega*t grid, A(t)/p0)."""
    nx = 256
    dx = 0.2e-3
    D = nx * dx
    k = 2.0 * math.pi / D                 # periodic-compatible fundamental
    om = C0 * k
    T = 2.0 * math.pi / om
    p0 = 1.0e5
    x = (np.arange(nx) + 0.5) * dx
    shape = np.sin(k * x)

    p_init = np.zeros((nx, 1, 1))
    p_init[:, 0, 0] = p0 * shape
    grid = kw.Grid(nx=nx, ny=1, nz=1, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=C0, density=RHO0)
    src = kw.Source.from_initial_pressure(p_init.copy())
    mask = np.zeros((nx, 1, 1), dtype=bool)
    mask[:, 0, 0] = True
    sim = kw.Simulation(grid, medium, src, kw.Sensor.from_mask(mask),
                        solver=kw.SolverType.PSTD, pml_size=pml_size)

    dt = 0.15 * dx / C0
    n_steps = int(math.ceil(n_periods * T / dt))
    res = sim.run(time_steps=n_steps, dt=dt)
    sd = np.asarray(res.sensor_data, dtype=float)
    dt_a = float(res.dt)
    amp = (sd.T @ shape) / (shape @ shape) / p0     # modal amplitude per step
    n = np.arange(sd.shape[1])
    return om * (n * dt_a), amp


def test_pml_size_zero_sustains_standing_wave():
    """pml_size=0 → transparent boundary → standing wave sustained as cos(ωt)."""
    wt, amp = _standing_wave_modal_amplitude(pml_size=0, n_periods=1.0)

    # Sample the amplitude near ωt = 0, π/2, π and compare to cos(ωt).
    for target in (0.0, math.pi / 2.0, math.pi):
        i = int(np.argmin(np.abs(wt - target)))
        expected = math.cos(wt[i])
        assert amp[i] == pytest.approx(expected, abs=0.05), (
            f"pml_size=0 should sustain the standing wave: at ωt={wt[i]:.3f} "
            f"got A={amp[i]:.3f}, expected cos={expected:.3f}"
        )

    # The sign reversal at ωt ≈ π is the decisive signature of a sustained
    # (undamped) standing wave; an absorbing boundary never reaches it.
    i_pi = int(np.argmin(np.abs(wt - math.pi)))
    assert amp[i_pi] < -0.85, (
        f"pml_size=0 must reach the negative half-cycle (A≈−1 at ωt=π); "
        f"got {amp[i_pi]:.3f}"
    )


def test_pml_size_nonzero_decays_standing_wave():
    """pml_size>0 → absorbing PML → standing wave amplitude decays (no sign flip)."""
    wt, amp = _standing_wave_modal_amplitude(pml_size=10, n_periods=1.0)
    i_pi = int(np.argmin(np.abs(wt - math.pi)))
    # With absorption, the constituents are damped crossing the domain; the
    # mode never completes the swing to A ≈ −1.
    assert amp[i_pi] > -0.7, (
        f"pml_size=10 should damp the standing wave (no full sign reversal); "
        f"got A={amp[i_pi]:.3f} at ωt=π"
    )


def test_pml_size_zero_and_nonzero_differ():
    """The two boundary regimes must produce materially different dynamics."""
    _, amp0 = _standing_wave_modal_amplitude(pml_size=0, n_periods=1.0)
    _, amp10 = _standing_wave_modal_amplitude(pml_size=10, n_periods=1.0)
    # If pml_size were ignored (the old bug) these would be identical.
    assert np.max(np.abs(amp0 - amp10)) > 0.3, (
        "pml_size=0 and pml_size=10 produced near-identical results — "
        "pml_size is being ignored by the run path"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
