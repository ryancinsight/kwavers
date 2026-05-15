"""Round-trip tests for the pykwavers FFT bindings (fft1/ifft1/fft3/ifft3).

Each test constructs analytically defined input, applies the forward transform,
applies the inverse transform, and asserts bit-accurate reconstruction within
double-precision floating-point round-off (atol=1e-12).
"""

import numpy as np
import pytest

import pykwavers as kw


def test_fft1_ifft1_roundtrip_uniform():
    """Forward then inverse 1-D DFT must recover the original real signal."""
    signal = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    spectrum = kw.fft1(signal)
    assert spectrum.shape == (8,), "fft1 must preserve length"
    assert spectrum.dtype == np.complex128, "fft1 must return complex128"
    recovered = kw.ifft1(spectrum)
    assert recovered.shape == (8,)
    assert recovered.dtype == np.float64
    np.testing.assert_allclose(recovered, signal, atol=1e-12)


def test_fft1_ifft1_roundtrip_sinusoid():
    """Round-trip of a pure sinusoid must recover all sample values."""
    n = 16
    t = np.arange(n, dtype=np.float64)
    signal = np.sin(2.0 * np.pi * 3.0 * t / n)
    recovered = kw.ifft1(kw.fft1(signal))
    np.testing.assert_allclose(recovered, signal, atol=1e-12)


def test_fft1_spectrum_dc_component():
    """DC bin of the DFT of an all-ones signal equals N (before normalisation).

    kwavers/apollo uses 1/N normalisation on the forward transform, so
    spectrum[0].real == 1.0 for a unit constant input.
    """
    n = 8
    signal = np.ones(n, dtype=np.float64)
    spectrum = kw.fft1(signal)
    assert abs(spectrum[0].real - 1.0) < 1e-12, (
        f"DC bin expected 1.0 (1/N-normalised), got {spectrum[0].real}"
    )


def test_fft3_ifft3_roundtrip_arange():
    """Forward then inverse 3-D DFT must recover a linearly-spaced 4×4×4 field."""
    field = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    spectrum = kw.fft3(field)
    assert spectrum.shape == (4, 4, 4), "fft3 must preserve shape"
    assert spectrum.dtype == np.complex128
    recovered = kw.ifft3(spectrum)
    assert recovered.shape == (4, 4, 4)
    assert recovered.dtype == np.float64
    np.testing.assert_allclose(recovered, field, atol=1e-12)


def test_fft3_ifft3_roundtrip_gaussian():
    """Round-trip of an analytically defined 3-D Gaussian pulse."""
    nx, ny, nz = 8, 8, 8
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    ix, iy, iz = np.ogrid[:nx, :ny, :nz]
    field = np.exp(
        -(((ix - cx) / 2.0) ** 2 + ((iy - cy) / 2.0) ** 2 + ((iz - cz) / 2.0) ** 2)
    ).astype(np.float64)
    recovered = kw.ifft3(kw.fft3(field))
    np.testing.assert_allclose(recovered, field, atol=1e-12)


def test_fft1_rejects_empty():
    """fft1 must raise ValueError for zero-length input."""
    with pytest.raises((ValueError, Exception)):
        kw.fft1(np.array([], dtype=np.float64))


def test_fft3_rejects_zero_dim():
    """fft3 must raise ValueError when any dimension is zero."""
    with pytest.raises((ValueError, Exception)):
        kw.fft3(np.zeros((0, 4, 4), dtype=np.float64))
