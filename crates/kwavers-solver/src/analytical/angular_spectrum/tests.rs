//! Tests for the CW angular-spectrum propagator.

use super::{backproject_cw, fft_freq, fft_len, propagate_cw, CwMedium};
use apollo::{fft_2d_complex, ifft_2d_complex, Complex64};
use eunomia::FloatElement;
use kwavers_core::error::{ConfigError, KwaversError};
use leto::Array2;

fn gaussian(nx: usize, ny: usize, dx: f64, sigma: f64) -> Array2<Complex64> {
    let mut src = Array2::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            let x = (f64::from_count(i) - f64::from_count(nx) / 2.0) * dx;
            let y = (f64::from_count(j) - f64::from_count(ny) / 2.0) * dx;
            src[[i, j]] = Complex64::new(((-x * x - y * y) / (2.0 * sigma * sigma)).exp(), 0.0);
        }
    }
    src
}

/// apollo's `fft`/`ifft` pair must match the numpy convention (forward
/// unnormalised, inverse scaled by 1/N): the propagator relies on it.
#[test]
fn test_fft_ifft_round_trip_is_identity() {
    let mut field = Array2::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            let (fi, fj) = (f64::from_count(i), f64::from_count(j));
            field[[i, j]] = Complex64::new(fi + 0.25 * fj, 0.5 * fi - fj);
        }
    }
    let back = ifft_2d_complex(&fft_2d_complex(&field));
    let mut max_err = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let d = back[[i, j]] - field[[i, j]];
            max_err = max_err.max((d.re * d.re + d.im * d.im).sqrt());
        }
    }
    assert!(
        max_err < 1.0e-12,
        "fft/ifft must round-trip to 1e-12, got {max_err:.3e}"
    );
}

/// The FFT length doubles past the next power of two (k-wave-python:
/// `2 ** (ceil(log2(max)) + 1)`).
#[test]
fn test_fft_len_matches_reference() {
    for (dim, want) in [(1, 2), (2, 4), (3, 8), (100, 256), (128, 256), (129, 512)] {
        assert_eq!(fft_len(dim), want, "fft_len({dim})");
    }
}

/// FFT-order wavenumber grid: DC exactly at index 0, negative Nyquist at
/// index N/2 (the reference's centred `k_vec[N // 2] = 0.0` lands at index 0
/// after `ifftshift`; the Nyquist bin keeps `-(N/2)·dk` and stays evanescent).
#[test]
fn test_fft_freq_matches_ifftshifted_reference() {
    let n = 256;
    let dx = 0.15e-3;
    let step = 2.0 * std::f64::consts::PI / (f64::from_count(n) * dx);
    assert_eq!(fft_freq(0, n, step), 0.0, "DC must be exactly zero");
    assert_eq!(
        fft_freq(n / 2, n, step),
        -(f64::from_count(n) / 2.0) * step,
        "Nyquist bin must stay negative"
    );
    assert!(
        fft_freq(n / 2, n, step) < 0.0,
        "Nyquist bin must be evanescent-side"
    );
}

/// A zero-distance plane returns the input bit-for-bit (no FFT round-off).
#[test]
fn test_zero_distance_returns_input_exactly() {
    let src = gaussian(16, 16, 0.15e-3, 2.0e-3);
    let out = propagate_cw(
        &src,
        0.15e-3,
        &[0.0],
        1.0e6,
        CwMedium::lossless(1500.0),
        true,
        0,
    )
    .expect("valid propagation");
    for i in 0..16 {
        for j in 0..16 {
            assert_eq!(out[[i, j, 0]], src[[i, j]], "z=0 must copy input");
        }
    }
}

/// Forward then backward propagation recovers a band-limited Gaussian:
/// `H_back · H_fwd = 1` on propagating modes, evanescent modes are filtered
/// both ways.
#[test]
fn test_forward_backward_round_trip_recovers_source() {
    // Wide domain, short range: the diffracted beam stays inside the
    // window, so cropping loses no energy and the round trip is exact.
    for angular_restriction in [true, false] {
        let nx = 64;
        let dx = 0.15e-3;
        let src = gaussian(nx, nx, dx, 0.6e-3);
        let z = 3.0e-3;
        // f0 = 3 MHz concentrates the Gaussian spectrum (sigma_k/k = 0.2)
        // well inside every cutoff, isolating operator error from filter loss.
        let fwd = propagate_cw(
            &src,
            dx,
            &[z],
            3.0e6,
            CwMedium::lossless(1500.0),
            angular_restriction,
            0,
        )
        .expect("valid forward propagation");
        let mut meas = Array2::zeros((nx, nx));
        for i in 0..nx {
            for j in 0..nx {
                meas[[i, j]] = fwd[[i, j, 0]];
            }
        }
        let recon = backproject_cw(&meas, dx, z, 3.0e6, 1500.0, angular_restriction)
            .expect("valid backprojection");
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..nx {
            for j in 0..nx {
                let d = recon[[i, j]] - src[[i, j]];
                num += d.re * d.re + d.im * d.im;
                let s = src[[i, j]];
                den += s.re * s.re + s.im * s.im;
            }
        }
        let rel = (num / den).sqrt();
        assert!(
            rel < 1.0e-6,
            "round trip (restriction={angular_restriction}) rel err {rel:.3e}"
        );
    }
}

/// Absorption strictly attenuates the on-axis pressure versus lossless.
#[test]
fn test_absorption_attenuates_on_axis_pressure() {
    let nx = 32;
    let dx = 0.15e-3;
    let src = gaussian(nx, nx, dx, 0.6e-3);
    let z = [20.0e-3];
    let free = propagate_cw(&src, dx, &z, 1.0e6, CwMedium::lossless(1500.0), true, 0)
        .expect("valid lossless propagation");
    let lossy = propagate_cw(
        &src,
        dx,
        &z,
        1.0e6,
        CwMedium {
            sound_speed: 1500.0,
            alpha_np_per_m: 5.0,
        },
        true,
        0,
    )
    .expect("valid lossy propagation");
    let c = nx / 2;
    let mag = |v: Complex64| (v.re * v.re + v.im * v.im).sqrt();
    assert!(
        mag(free[[c, c, 0]]) > 0.0,
        "lossless centre must be nonzero"
    );
    assert!(
        mag(lossy[[c, c, 0]]) < mag(free[[c, c, 0]]),
        "absorption must attenuate the on-axis pressure"
    );
}

/// Every degenerate scalar input is a typed `Config` error, never a panic.
#[test]
fn test_propagate_rejects_degenerate_inputs() {
    let src = gaussian(8, 8, 0.15e-3, 2.0e-3);
    // (dx, f0, c0, alpha, z): each row breaks exactly one parameter.
    let cases = [
        (0.0, 1.0e6, 1500.0, 0.0, 5.0e-3),
        (0.15e-3, -1.0e6, 1500.0, 0.0, 5.0e-3),
        (0.15e-3, 1.0e6, f64::NAN, 0.0, 5.0e-3),
        (0.15e-3, 1.0e6, 1500.0, -1.0, 5.0e-3),
        (0.15e-3, 1.0e6, 1500.0, 0.0, f64::NAN),
        // Nyquist: dx > c0 / (2 f0) = 0.75 mm here.
        (1.0e-3, 1.0e6, 1500.0, 0.0, 5.0e-3),
    ];
    for (dx, f0, c0, alpha, z) in cases {
        let medium = CwMedium {
            sound_speed: c0,
            alpha_np_per_m: alpha,
        };
        let err = propagate_cw(&src, dx, &[z], f0, medium, true, 0)
            .expect_err("degenerate input must be rejected");
        assert!(
            matches!(err, KwaversError::Config(ConfigError::InvalidValue { .. })),
            "degenerate input must report Config::InvalidValue, got {err}"
        );
    }
}
