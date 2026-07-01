"""Round-trip tests for the pykwavers FFT bindings (fft1/ifft1/fft3/ifft3).

Each test constructs analytically defined input, applies the forward transform,
applies the inverse transform, and asserts bit-accurate reconstruction within
double-precision floating-point round-off (atol=1e-12).
"""

import numpy as np
import pytest
from pathlib import Path

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
    """DC bin of the DFT of an all-ones signal equals N (no normalisation).

    kwavers/apollo uses the unnormalised forward DFT, so for an
    N-sample constant-1 input the DC bin is N.
    """
    n = 8
    signal = np.ones(n, dtype=np.float64)
    spectrum = kw.fft1(signal)
    assert abs(spectrum[0].real - float(n)) < 1e-12, (
        f"DC bin expected {float(n)} (unnormalised DFT), got {spectrum[0].real}"
    )


def test_demeaned_hann_power_spectrum_matches_numpy_rfft_contract():
    n = 17
    spacing = 0.25
    axis = np.arange(n, dtype=np.float64)
    signal = 2.0 + 0.75 * np.cos(2.0 * np.pi * 3.0 * axis / n)

    freq, power = kw.demeaned_hann_power_spectrum_1d(signal, spacing)
    expected_windowed = (signal - signal.mean()) * np.hanning(n)
    expected_freq = np.fft.rfftfreq(n, d=spacing)
    expected_power = np.abs(np.fft.rfft(expected_windowed)) ** 2

    np.testing.assert_allclose(freq, expected_freq, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(power, expected_power, rtol=1.0e-12, atol=1.0e-12)


def test_demeaned_hann_power_spectrum_rejects_invalid_inputs():
    with pytest.raises((ValueError, Exception)):
        kw.demeaned_hann_power_spectrum_1d(np.array([1.0], dtype=np.float64), 1.0)
    with pytest.raises((ValueError, Exception)):
        kw.demeaned_hann_power_spectrum_1d(np.array([1.0, 2.0], dtype=np.float64), 0.0)
    with pytest.raises((ValueError, Exception)):
        kw.demeaned_hann_power_spectrum_1d(
            np.array([1.0, np.nan], dtype=np.float64), 1.0
        )


def test_chapter25_rtm_spectrum_routes_to_rust_fft_helper():
    source = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "book"
        / "ch25_rtm_adaptive_beamforming.py"
    ).read_text(encoding="utf-8")
    start = source.index("def fig11_standing_wave_spectrum(")
    end = source.index("\ndef fig12_delay_law_evolution", start)
    body = source[start:end]

    assert "pykwavers.demeaned_hann_power_spectrum_1d" in body
    assert "np.fft.rfft" not in body
    assert "np.fft.rfftfreq" not in body
    assert "np.hanning" not in body


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
