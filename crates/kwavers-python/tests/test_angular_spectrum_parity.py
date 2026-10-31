#!/usr/bin/env python3
"""Parity: Rust-backed angular-spectrum propagators vs k-wave-python.

``pykwavers.angular_spectrum_cw`` / ``backward_angular_spectrum_cw`` are thin
facades over ``kwavers_solver::analytical::angular_spectrum`` (KW-PY-ANGULAR-
SPECTRUM). These tests prove the binding agrees with the k-wave-python
reference within the derived tolerance (Pearson r >= 0.9999, PSNR >= 60 dB on
|pressure|, matching the compare-example criterion: double-precision FFT
round-off only), and that forward-then-backward recovers a band-limited
source.
"""

import numpy as np
import pytest

_ = pytest.importorskip("pykwavers")

import pykwavers as pkw
from pykwavers.kwave_python_bridge import KWAVE_PYTHON_AVAILABLE

if not KWAVE_PYTHON_AVAILABLE:
    pytest.skip("k-wave-python not available", allow_module_level=True)

from kwave.utils.angular_spectrum_cw import angular_spectrum_cw as kw_angular_spectrum_cw

F0 = 1_000_000.0
C0 = 1500.0
DX = 0.15e-3
N = 64
APERTURE_RADIUS = 2e-3
ALPHA_COEFF = 0.5
ALPHA_POWER = 1.5
Z_POSITIONS = [5e-3, 10e-3, 20e-3]


def _circular_piston_source():
    x = (np.arange(N) - N / 2.0) * DX
    y = (np.arange(N) - N / 2.0) * DX
    xx, yy = np.meshgrid(x, y, indexing="ij")
    src = np.zeros((N, N), dtype=complex)
    src[xx**2 + yy**2 <= APERTURE_RADIUS**2] = 1.0
    return src


def _metrics(a, b):
    a = np.abs(a).ravel()
    b = np.abs(b).ravel()
    corr = np.corrcoef(a, b)[0, 1]
    rmse = np.sqrt(np.mean((a - b) ** 2))
    psnr = 20.0 * np.log10(a.max() / (rmse + 1e-30))
    return float(corr), float(psnr)


@pytest.mark.parametrize("angular_restriction", [True, False])
@pytest.mark.parametrize("absorbing", [False, True])
def test_angular_spectrum_cw_matches_kwave_reference(angular_restriction, absorbing):
    """Rust propagator agrees with k-wave-python per plane and scenario."""
    src = _circular_piston_source()
    if absorbing:
        medium_kw = {"sound_speed": C0, "alpha_coeff": ALPHA_COEFF, "alpha_power": ALPHA_POWER}
        medium_pkw = dict(medium_kw)
    else:
        medium_kw = int(C0)
        medium_pkw = C0
    # k-wave-python takes one float z plane per call (beartype enforced).
    kw_planes = [
        kw_angular_spectrum_cw(
            input_plane=src.real,
            dx=DX,
            z_pos=float(z),
            f0=int(F0),
            medium=medium_kw,
            angular_restriction=angular_restriction,
        )[:, :, 0]
        for z in Z_POSITIONS
    ]
    pkw_pressure = pkw.angular_spectrum_cw(
        input_plane=src,
        dx=DX,
        z_pos=list(Z_POSITIONS),
        f0=F0,
        medium=medium_pkw,
        angular_restriction=angular_restriction,
    )
    assert pkw_pressure.shape == (N, N, len(Z_POSITIONS))
    for zi, (kw_plane, z) in enumerate(zip(kw_planes, Z_POSITIONS)):
        corr, psnr = _metrics(kw_plane, pkw_pressure[:, :, zi])
        assert corr >= 0.9999, f"z={z}: Pearson r={corr:.6f}"
        assert psnr >= 60.0, f"z={z}: PSNR={psnr:.1f} dB"


def test_backward_round_trip_recovers_band_limited_source():
    """Forward-then-backward through the Rust binding recovers a Gaussian."""
    sigma = 0.6e-3
    x = (np.arange(N) - N / 2.0) * DX
    xx, yy = np.meshgrid(x, x, indexing="ij")
    src = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)).astype(complex)
    z = 3.0e-3
    fwd = pkw.angular_spectrum_cw(
        input_plane=src, dx=DX, z_pos=[z], f0=3.0e6, medium=C0,
    )
    recon = pkw.backward_angular_spectrum_cw(
        measurement_plane=np.ascontiguousarray(fwd[:, :, 0]),
        dx=DX, z_m=z, f0=3.0e6, medium=C0,
    )
    rel = np.linalg.norm((recon - src).ravel()) / np.linalg.norm(src.ravel())
    assert rel < 1.0e-6, f"round-trip rel err {rel:.3e}"


def test_degenerate_inputs_raise_value_error():
    """Rust validation surfaces as ValueError through the binding."""
    src = _circular_piston_source()
    with pytest.raises(ValueError, match="Nyquist"):
        pkw.angular_spectrum_cw(
            input_plane=src, dx=1.0e-3, z_pos=[5e-3], f0=F0, medium=C0,
        )
    with pytest.raises(ValueError, match="alpha_np_per_m"):
        pkw.angular_spectrum_cw(
            input_plane=src, dx=DX, z_pos=[5e-3], f0=F0,
            medium={"sound_speed": C0, "alpha_coeff": -0.5, "alpha_power": 1.5},
        )
