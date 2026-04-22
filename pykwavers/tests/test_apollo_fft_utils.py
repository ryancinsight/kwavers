import numpy as np

import pykwavers as kw


def test_fft1_roundtrip():
    signal = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    spectrum = kw.fft1(signal)
    recovered = kw.ifft1(spectrum)
    assert np.allclose(signal, recovered, atol=1e-12)


def test_fft3_roundtrip():
    field = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    spectrum = kw.fft3(field)
    recovered = kw.ifft3(spectrum)
    assert np.allclose(field, recovered, atol=1e-12)


def test_nufft_fast_tracks_exact():
    positions = np.array([0.01, 0.09, 0.23, 0.51], dtype=np.float64)
    values = np.array([1.0 + 0.0j, 0.5 + 0.2j, -0.2 + 0.3j, 0.1 - 0.4j], dtype=np.complex128)
    exact = kw.nufft_type1_1d(positions, values, 0.05, n_out=16)
    fast = kw.nufft_type1_1d_fast(positions, values, 0.05, n_out=16)
    assert np.allclose(exact, fast, atol=1e-5)


def test_fft_backend_capabilities():
    capabilities = kw.fft_backend_capabilities()
    assert capabilities["cpu"]["available"] is True
    assert capabilities["cpu"]["supports_fft_3d"] is True
    assert "wgpu" in capabilities


def test_available_fft_backends_contains_cpu():
    backends = kw.available_fft_backends()
    assert "cpu" in backends
