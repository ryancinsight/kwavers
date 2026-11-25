use super::*;
use kwavers_math::fft::KSpaceCalculator;
use std::f64::consts::PI;

// ── Spectral Laplacian correctness ─────────────────────────────────────

/// **Theorem (∇²[constant] = 0)**:
///
/// For any constant function f = C, ∇²f = 0 everywhere.
/// Spectrally: −|k|²·FFT(C) = 0 for all k ≠ 0 (DC bin has k²=0), so
/// IFFT(result) = 0.
#[test]
fn spectral_laplacian_of_constant_is_zero() {
    let n = 16usize;
    let dx = 1.0e-3_f64;
    let k_sq = KSpaceCalculator::generate_k_squared(n, n, n, dx, dx, dx);

    let field = leto::Array3::from_elem([n, n, n], 7.5_f64);
    let lap = compute_laplacian_spectral(&field, &k_sq);

    let max_abs = lap.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    // Budget: N³·log₂(N)·ε_machine ≈ 16³·4·2.2e-16 ≈ 3.6e-11
    assert!(
        max_abs < 1e-8,
        "∇²(constant) must be zero; max_abs={max_abs:.3e}"
    );
}

/// **Theorem (spectral Laplacian of sin(kx·x))**:
///
/// For f(x,y,z) = sin(k₁·x), the exact Laplacian is:
///
/// ```text
/// ∇²f = ∂²f/∂x² = −k₁² sin(k₁·x)
/// ```
///
/// The spectral method reproduces this to machine precision for any
/// DFT-representable mode.  We verify the interior L∞ relative error < 1e-8.
#[test]
fn spectral_laplacian_of_sine_matches_analytical() {
    let n = 32usize;
    let dx = 1.0e-3_f64;
    let k_sq = KSpaceCalculator::generate_k_squared(n, n, n, dx, dx, dx);

    let k1 = 2.0 * PI / (n as f64 * dx); // fundamental wavenumber
    let mut field = leto::Array3::<f64>::zeros((n, n, n));
    for i in 0..n {
        let x = i as f64 * dx;
        let val = (k1 * x).sin();
        for j in 0..n {
            for k in 0..n {
                field[[i, j, k]] = val;
            }
        }
    }

    let lap = compute_laplacian_spectral(&field, &k_sq);

    // Check interior points only (boundaries may pick up Gibbs from non-periodic extension)
    let center = n / 2;
    let mut max_rel_err = 0.0_f64;
    for i in 1..n - 1 {
        let x = i as f64 * dx;
        let expected = -k1 * k1 * (k1 * x).sin();
        let computed = lap[[i, center, center]];
        let denom = k1 * k1; // normalise by |k|² to get relative error
        let rel_err = (computed - expected).abs() / denom;
        max_rel_err = max_rel_err.max(rel_err);
    }

    assert!(
        max_rel_err < 1e-8,
        "∇²(sin(k₁x)) spectral relative error = {max_rel_err:.3e} (must be < 1e-8)"
    );
}

/// **Theorem (scratch-reusing path bit-identical to allocating path)**:
///
/// `compute_laplacian_spectral_into` must produce results bitwise identical to
/// `compute_laplacian_spectral` on the same input, confirming the scratch-reusing
/// scratch buffer path does not alter the computation.
#[test]
fn spectral_laplacian_into_is_bitwise_identical_to_allocating() {
    let n = 16usize;
    let dx = 1.0e-3_f64;
    let k_sq = KSpaceCalculator::generate_k_squared(n, n, n, dx, dx, dx);

    let k1 = 2.0 * PI / (n as f64 * dx);
    let mut field = leto::Array3::<f64>::zeros((n, n, n));
    for i in 0..n {
        let x = i as f64 * dx;
        for j in 0..n {
            for k in 0..n {
                field[[i, j, k]] = (k1 * x).cos() * (k1 * j as f64 * dx).sin();
            }
        }
    }

    // Allocating path
    let lap_alloc = compute_laplacian_spectral(&field, &k_sq);

    // Zero-allocation path
    let shape = (n, n, n);
    let mut fft_scratch = Array3::<Complex64>::from_elem([n, n, n], Complex64::default());
    let mut lap_into = leto::Array3::<f64>::zeros(shape);
    compute_laplacian_spectral_into(&field, &k_sq, &mut fft_scratch, &mut lap_into);

    // Results must be bitwise identical
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                assert_eq!(
                    lap_alloc[[i, j, k]].to_bits(),
                    lap_into[[i, j, k]].to_bits(),
                    "spectral_laplacian_into result differs at [{i},{j},{k}]: \
                     alloc={:.6e} into={:.6e}",
                    lap_alloc[[i, j, k]],
                    lap_into[[i, j, k]]
                );
            }
        }
    }
}

/// A transposed owned array can have the same shape as a C-order array.
/// A different mode on each axis detects accidental physical-order traversal.
#[test]
fn full_spectrum_laplacian_respects_strided_fields_symbols_and_destinations() {
    use leto::{Layout, VecStorage};

    let shape = [4, 4, 6];
    let layout = Layout::try_new(shape, [6, 24, 1], 0).unwrap();
    let mut field = Array3::new(layout, VecStorage::fill(96, 0.0)).unwrap();
    let mut multiplier = field.clone();
    let mut out = field.clone();
    let mut scratch = Array3::new(layout, VecStorage::fill(96, Complex64::default())).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..6 {
                let x = PI * f64::from(u8::try_from(i).unwrap()) / 2.0;
                let y = PI * f64::from(u8::try_from(j).unwrap());
                field[[i, j, k]] = x.cos() + 0.5 * y.cos();
                // Distinct weights for the two Hermitian mode pairs.
                multiplier[[i, j, k]] = if i == 1 || i == 3 { 3.0 } else { 7.0 };
            }
        }
    }
    assert!(out.as_slice().is_none());
    compute_laplacian_spectral_into(&field, &multiplier, &mut scratch, &mut out);

    for i in 0..4 {
        for j in 0..4 {
            let x = PI * f64::from(u8::try_from(i).unwrap()) / 2.0;
            let y = PI * f64::from(u8::try_from(j).unwrap());
            let expected = -3.0 * x.cos() - 3.5 * y.cos();
            for k in 0..6 {
                assert!((out[[i, j, k]] - expected).abs() < 1e-12);
            }
        }
    }
}

#[test]
#[should_panic(expected = "Laplacian multiplier shape mismatch")]
fn full_spectrum_rejects_permuted_extents_with_the_same_element_count() {
    compute_laplacian_spectral_into(
        &Array3::zeros([2, 3, 4]),
        &Array3::zeros([3, 2, 4]),
        &mut Array3::zeros([2, 3, 4]),
        &mut Array3::zeros([2, 3, 4]),
    );
}
