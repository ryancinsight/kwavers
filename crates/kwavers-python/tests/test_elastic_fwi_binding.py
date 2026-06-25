"""Binding-surface test for the elastic shear-wave FWI (ADR 033).

Verifies that `kw.elastic_shear_fwi_reconstruct` is exposed and that it actually
runs the inversion: a stiff disk phantom is recovered as clearly stiffer than the
background, with the background preserved. A broken gradient/solve would leave the
output at the homogeneous background (lesion peak ~= bg), failing these checks.
"""

import numpy as np
import pytest

import pykwavers as kw

# Compressible test medium (c_P = sqrt(3)*c_S) so the elastic CFL stays tractable;
# the FWI machinery is identical to the near-incompressible tissue case.
RHO = 1000.0
C_S = 2.0
C_P = 3.4641016
MU_BG = RHO * C_S * C_S  # 4000 Pa
DX = 1.0e-3
N = 36


def _phantom():
    mu = np.full((N, N), MU_BG)
    c = N / 2.0
    for i in range(N):
        for j in range(N):
            if np.hypot(i - c, j - c) <= 5.0:
                mu[i, j] = 3.0 * MU_BG
    return mu, c


def test_symbol_exposed():
    assert hasattr(kw, "elastic_shear_fwi_reconstruct")


def test_reconstructs_stiff_lesion():
    mu_true, c = _phantom()
    mu_rec = kw.elastic_shear_fwi_reconstruct(
        mu_true,
        DX,
        RHO,
        C_S,
        C_P,
        n_steps=180,
        iterations=12,
    )
    assert mu_rec.shape == (N, N)
    assert np.all(np.isfinite(mu_rec))

    lesion = np.zeros((N, N), dtype=bool)
    bg = np.zeros((N, N), dtype=bool)
    for i in range(8, 28):
        for j in range(8, 28):
            if np.hypot(i - c, j - c) <= 5.0:
                lesion[i, j] = True
            else:
                bg[i, j] = True

    # The lesion is recovered as clearly stiffer than background (a broken solve
    # would leave it ~= MU_BG).
    assert mu_rec[lesion].max() >= 1.8 * MU_BG, mu_rec[lesion].max()
    # The background is preserved (no global drift).
    assert abs(float(mu_rec[bg].mean()) - MU_BG) <= 0.2 * MU_BG, float(mu_rec[bg].mean())


def test_rejects_invalid_medium():
    mu_true, _ = _phantom()
    # c_p^2 < 2 c_s^2 violates thermodynamic stability -> error.
    with pytest.raises(Exception):
        kw.elastic_shear_fwi_reconstruct(mu_true, DX, RHO, 2.0, 2.0)
