from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_chapter03_fubini_waveform_uses_rust_wave_helper():
    source = (BOOK_DIR / "ch03_nonlinear_acoustics.py").read_text(encoding="utf-8")

    assert "kw.fubini_waveform" in source
    assert "kw.hann_windowed_harmonic_amplitudes" in source
    assert "kw.fubini_harmonic_spectrum" not in source
    assert "np.fft.rfft" not in source
    assert "np.hanning" not in source
    assert "p += P0 * bn_val * np.sin" not in source
    assert "P0 * np.sin(OMEGA0 * t_src)" not in source
    assert "Optional: pykwavers" not in source


def test_fubini_waveform_binding_matches_harmonic_expansion():
    import pykwavers as kw

    freq_hz = 1.0e6
    p0_pa = 1.0e6
    sigma = 0.5
    n_max = 5
    t = np.linspace(0.0, 1.0 / freq_hz, 9, dtype=np.float64)

    waveform = np.asarray(kw.fubini_waveform(t, p0_pa, freq_hz, sigma, n_max))
    spectrum = np.asarray(kw.fubini_harmonic_spectrum(n_max, sigma))
    expected = np.zeros_like(t)
    for idx, coefficient in enumerate(spectrum):
        n = idx + 1
        expected += p0_pa * coefficient * np.sin(n * 2.0 * np.pi * freq_hz * t)

    # Same f64 harmonic order as the Rust kernel; 1e-6 Pa is below one
    # micro-part-per-million of the 1 MPa test amplitude.
    assert np.allclose(waveform, expected, rtol=1.0e-12, atol=1.0e-6)


def test_fubini_waveform_binding_reduces_to_sinusoid_at_zero_sigma():
    import pykwavers as kw

    freq_hz = 1.0e6
    p0_pa = 1.0e6
    t = np.linspace(0.0, 1.0 / freq_hz, 9, dtype=np.float64)
    waveform = np.asarray(kw.fubini_waveform(t, p0_pa, freq_hz, 0.0, 8))
    expected = p0_pa * np.sin(2.0 * np.pi * freq_hz * t)

    # Same f64 sinusoid expression as the sigma=0 Rust branch; 1e-6 Pa is below
    # one micro-part-per-million of the 1 MPa test amplitude.
    assert np.allclose(waveform, expected, rtol=1.0e-12, atol=1.0e-6)


def test_hann_windowed_harmonic_amplitudes_matches_numpy_reference():
    import pykwavers as kw

    n_samples = 64
    sample_rate_hz = 16_000.0
    dt_s = 1.0 / sample_rate_hz
    fundamental_hz = 1_000.0
    t = np.arange(n_samples, dtype=np.float64) * dt_s
    traces = np.vstack(
        [
            3.0 * np.sin(2.0 * np.pi * fundamental_hz * t)
            + 0.5 * np.sin(2.0 * np.pi * 2.0 * fundamental_hz * t),
            1.5 * np.sin(2.0 * np.pi * 3.0 * fundamental_hz * t),
        ]
    )

    got = np.asarray(
        kw.hann_windowed_harmonic_amplitudes(
            traces[:, :], dt_s, fundamental_hz, 3
        )
    )
    window = np.hanning(n_samples)
    df_hz = 1.0 / (n_samples * dt_s)
    expected = np.zeros((2, 3), dtype=np.float64)
    for row, trace in enumerate(traces):
        spectrum = np.abs(np.fft.rfft(trace * window)) * 2.0 / window.sum()
        for harmonic in range(1, 4):
            idx = int(round(harmonic * fundamental_hz / df_hz))
            expected[row, harmonic - 1] = spectrum[idx]

    # Differential PyO3 boundary check against NumPy's same Hann-windowed DFT
    # convention; tolerance is f64 roundoff for a length-64 exact-bin transform.
    assert np.allclose(got, expected, rtol=1.0e-12, atol=1.0e-12)


def test_hann_windowed_harmonic_amplitudes_rejects_invalid_sample_period():
    import pykwavers as kw

    traces = np.zeros((1, 8), dtype=np.float64)
    with np.testing.assert_raises(ValueError):
        kw.hann_windowed_harmonic_amplitudes(traces, 0.0, 1_000.0, 1)
