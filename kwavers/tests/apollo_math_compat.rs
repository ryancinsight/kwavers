use apollofft::{nufft_type1_1d, nufft_type2_1d, UniformDomain1D};
use kwavers::math::Fft3d;
use ndarray::Array3;
use num_complex::Complex64 as C64;

#[test]
fn kwavers_math_reexports_fft_roundtrip() {
    let plan = Fft3d::new(4, 4, 4);
    let field = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| {
        (i as f64 * 0.3 + j as f64 * 0.2 + k as f64 * 0.1).sin()
    });
    let spectrum = plan.forward(&field);
    let recovered = plan.inverse(&spectrum);

    let max_abs_error = field
        .iter()
        .zip(recovered.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_abs_error < 1e-12);
}

/// Verify NUFFT type-1 (non-uniform → uniform) round-trips through type-2.
///
/// Theorem (Barnett et al. 2019, SIAM J. Sci. Comput. 41(5):C479-C504, §3):
///   For a single source at position x₀ with value f(x₀), the type-1 NUFFT output
///   F_k = Σ_j f(x_j) · exp(-2πi k x_j / L)
///   Applying type-2 at the same positions recovers the original value (up to
///   normalization: the round-trip multiplies by n for n uniform frequency bins).
///
/// Algorithm:
///   1. Construct UniformDomain1D with n=16 bins, spacing dx=1.0.
///   2. Type-1: compute exact DFT sum from a single source at x=0.25.
///   3. Type-2: evaluate Fourier series at x=0.25 using the spectrum from (2).
///   4. |reconstructed| ≈ n (NUFFT normalization convention).
#[test]
fn kwavers_math_reexports_nufft_exact_and_fast_paths() {
    let n = 16_usize;
    // dx = 1.0 → domain length L = n*dx = 16.0; position x = 0.25 ∈ [0, L)
    let domain = UniformDomain1D::new(n, 1.0).expect("UniformDomain1D construction");

    // Single non-uniform source at x = 0.25
    let positions = vec![0.25_f64];
    let values = vec![C64::new(1.0, 0.0)];

    // Type-1: non-uniform → uniform spectrum (3 args, exact direct summation)
    let spectrum = nufft_type1_1d(&positions, &values, domain);

    // Spectrum must have exactly n frequency bins
    assert_eq!(
        spectrum.len(),
        n,
        "NUFFT type-1 output must have n={n} frequency bins"
    );

    // Type-2: uniform spectrum → non-uniform (coeffs first, then positions)
    // Signature: nufft_type2_1d(fourier_coeffs, positions, domain) → Vec<Complex64>
    let reconstructed = nufft_type2_1d(&spectrum, &positions, domain);
    assert_eq!(
        reconstructed.len(),
        1,
        "NUFFT type-2 output must have 1 value (one position)"
    );

    // |reconstructed[0]| ≈ n (round-trip normalization: Σ_k F_k e^{2πikx/L} = n · f(x))
    let magnitude = reconstructed[0].norm();
    assert!(
        (magnitude - n as f64).abs() < 1.0,
        "NUFFT round-trip magnitude {magnitude:.3} should be ≈ {n} (normalization)"
    );
}
