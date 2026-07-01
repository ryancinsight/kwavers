from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_chapter04_array_panel_uses_rust_beam_pattern_helper():
    source = (BOOK_DIR / "ch04_transducer_arrays_beamforming.py").read_text(encoding="utf-8")

    assert "kw.beam_pattern_magnitude" in source
    assert "kw.grating_lobe_angles" in source
    assert "kw.apodization_window_response" in source
    assert "kw.lateral_resolution_m(float(depth_m / aperture_m), LAM)" in source
    assert "kw.delay_law_focus_2d(el_positions, el_z, 0.0, Z_F, C0)" in source
    assert "kw.beam_pattern_2d(x_pts, z_pts, el_positions, el_z, F0, C0, w_hann, delays)" in source
    assert "kw.bli_stencil_weights(np.asarray([frac_offset]), N_STENCIL)" in source
    assert "kw.linear_array_factor(" not in source
    assert "np.fft.fft" not in source
    assert "np.fft.fftshift" not in source
    assert "np.arcsin(sin_gl)" not in source
    assert "kw.lateral_resolution_m(depths_mm * 1e-3" not in source
    assert "kw.bli_stencil_weights(grid_idx.astype(float), frac_offset, EPS_BLI)" not in source
    assert "np.meshgrid(x_pts, z_pts)" not in source


def test_chapter11_bli_accuracy_uses_rust_error_curves():
    source = (BOOK_DIR / "ch11_sources_and_transducers.py").read_text(encoding="utf-8")

    assert "kw.bli_interpolation_error_curves(" in source
    assert "weights @ samples" not in source
    assert "np.sqrt(np.mean((reconstructed - ideal) ** 2))" not in source


def test_beam_pattern_and_grating_lobes_match_chapter_geometry():
    import pykwavers as kw

    sound_speed = 1540.0
    frequency = 5.0e6
    wavelength = sound_speed / frequency
    pitch = wavelength
    wavenumber = 2.0 * np.pi / wavelength
    steer_rad = np.deg2rad(20.0)
    theta_rad = np.deg2rad(np.linspace(-90.0, 90.0, 361))

    beam_mag = np.asarray(
        kw.beam_pattern_magnitude(
            theta_rad,
            wavenumber,
            pitch,
            64,
            steer_rad,
            wavenumber * pitch / 2.0,
        )
    )
    grating_lobes = np.asarray(kw.grating_lobe_angles(wavenumber, pitch, steer_rad))

    assert np.isclose(beam_mag.max(), 1.0, rtol=0.0, atol=1.0e-15)
    assert np.all(beam_mag >= 0.0)
    np.testing.assert_allclose(
        grating_lobes,
        np.asarray([np.arcsin(np.sin(steer_rad) - 1.0)]),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_lateral_resolution_binding_uses_f_number_contract():
    import pykwavers as kw

    depth_m = 40.0e-3
    aperture_m = 20.0e-3
    wavelength_m = 1540.0 / 5.0e6

    assert np.isclose(
        kw.lateral_resolution_m(depth_m / aperture_m, wavelength_m),
        0.886 * 2.0 * wavelength_m,
        rtol=1.0e-15,
        atol=0.0,
    )


def test_apodization_response_binding_matches_numpy_fft_convention():
    import pykwavers as kw

    n_elements = 8
    nfft = 32
    weights, spatial_freq, response_db = kw.apodization_window_response(
        n_elements, "Hann", nfft
    )
    weights = np.asarray(weights)
    spatial_freq = np.asarray(spatial_freq)
    response_db = np.asarray(response_db)

    expected = np.fft.fftshift(np.fft.fft(weights, n=nfft))
    expected_db = 20.0 * np.log10(np.abs(expected) / np.abs(expected).max() + 1.0e-12)
    expected_freq = np.linspace(-0.5, 0.5, nfft) * n_elements

    assert weights.shape == (n_elements,)
    assert response_db.shape == (nfft,)
    np.testing.assert_allclose(spatial_freq, expected_freq, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(response_db, expected_db, rtol=1.0e-12, atol=1.0e-12)


def test_beam_pattern_binding_accepts_axes_without_python_mesh():
    import pykwavers as kw

    x_axis = np.asarray([-0.5e-3, 0.0, 0.5e-3], dtype=np.float64)
    z_axis = np.asarray([20.0e-3, 25.0e-3], dtype=np.float64)
    elem_x = np.asarray([-0.15e-3, 0.15e-3], dtype=np.float64)
    elem_z = np.zeros_like(elem_x)
    weights = np.asarray([1.0, 1.0], dtype=np.float64)
    delays = np.asarray(kw.delay_law_focus_2d(elem_x, elem_z, 0.0, 25.0e-3, 1540.0))

    field_re, field_im = kw.beam_pattern_2d(
        x_axis,
        z_axis,
        elem_x,
        elem_z,
        5.0e6,
        1540.0,
        weights,
        delays,
    )

    assert field_re.shape == (3, 2)
    assert field_im.shape == (3, 2)
    assert np.isfinite(np.hypot(field_re, field_im)).all()


def test_bli_stencil_binding_returns_normalized_even_stencils():
    import pykwavers as kw

    weights = np.asarray(kw.bli_stencil_weights(np.asarray([0.0, 0.25, 0.5]), 14))

    assert weights.shape == (3, 14)
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(3), rtol=1.0e-15, atol=1.0e-15)


def test_bli_error_curve_binding_returns_improving_bli_curve():
    import pykwavers as kw

    ppw = np.asarray([8.0, 12.0, 16.0, 20.0])
    delta = np.linspace(0.0, 1.0, 128, endpoint=False)
    nearest, bli = kw.bli_interpolation_error_curves(ppw, delta, 8)

    assert nearest.shape == ppw.shape
    assert bli.shape == ppw.shape
    assert np.all(np.isfinite(nearest))
    assert np.all(np.isfinite(bli))
    assert np.all(bli < nearest)
