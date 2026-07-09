use super::operator::SpectralDerivativeOperator;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array3;

/// **Theorem (spectral derivative exactness for DFT-representable modes):**
///
/// For a grid function `f[n] = A·sin(2π·m·n/N)` where m ∈ {1, …, N/2−1}
/// (an exactly DFT-representable mode not at Nyquist), the spectral
/// derivative `F⁻¹[iω[k]·F[f]]` equals the exact continuous derivative
/// `∂f/∂x = A·k_m·cos(k_m·x_n)` to within floating-point rounding, where
/// `k_m = 2πm/(N·Δx)`.
///
/// **Proof.** The DFT of `f[n] = A sin(2πmn/N)` has exactly two non-zero bins:
/// bin m with coefficient `−iAN/2` and bin N−m with coefficient `iAN/2`.
/// Multiplying by `iω[k]` at bin m gives `iω[m]·(−iAN/2) = ω[m]·AN/2`,
/// and at bin N−m gives `iω[N−m]·(iAN/2) = −ω[N−m]·AN/2`. Since
/// `ω[m] = 2πm/(N·Δx) = k_m` and `ω[N−m] = −k_m` (negative frequency
/// conjugate), IDFT recovers `A·k_m·cos(2πmn/N)` exactly. No aliasing
/// occurs because m < N/2. (Trefethen 2000, Thm. 3.1.)
///
/// **Dealiasing note.** The 2/3-rule filter preserves mode m when
/// `k_m = 2πm/(N·Δx) ≤ 2π/(3·Δx)`, i.e., m ≤ N/3. For N=32 and m=1
/// (fundamental mode), this holds trivially.

/// Spectral derivative of sin(k₁·x) equals k₁·cos(k₁·x) to machine precision
/// for the fundamental mode (m=1, exactly DFT-representable, passes 2/3 dealiasing).
#[test]
fn spectral_derivative_x_exact_for_fundamental_mode() {
    let n = 32usize;
    let dx = 0.001_f64;
    let op = SpectralDerivativeOperator::new(n, n, n, dx, dx, dx);

    // k₁ = 2π/(N·Δx): fundamental mode, m=1 < N/3 ≈ 10.67 → passes 2/3-rule filter
    let k1 = TWO_PI / (n as f64 * dx);

    let mut field = Array3::<f64>::zeros([n, n, n]);
    for i in 0..n {
        let x = i as f64 * dx;
        for j in 0..n {
            for l in 0..n {
                field[[i, j, l]] = (k1 * x).sin();
            }
        }
    }

    let deriv = op.derivative_x(&field.view()).unwrap();

    // For a DFT-representable mode, the spectral derivative equals the exact
    // continuous derivative at every grid point to within floating-point rounding.
    // Expected: ∂/∂x[sin(k₁·x)] = k₁·cos(k₁·x)
    // The derivative amplitude is k₁; normalize absolute error by k₁ to obtain a
    // scale-invariant measure valid at zero-crossings of cos(k₁·x).
    let mut max_abs_err = 0.0_f64;
    for i in 0..n {
        let x = i as f64 * dx;
        let expected = k1 * (k1 * x).cos();
        let abs_err = (deriv[[i, n / 2, n / 2]] - expected).abs();
        max_abs_err = max_abs_err.max(abs_err);
    }
    // Normalized error: max_abs_err / k₁ gives relative accuracy uniform over the period.
    // FFT rounding budget: O(N·log₂(N)·ε_mach) ≈ 32·5·2.2e-16 ≈ 3.5e-14; allow 1e-10.
    let normalized_err = max_abs_err / k1;
    assert!(
        normalized_err < 1e-10,
        "spectral derivative of fundamental mode must match k₁·cos(k₁·x) to ≤1e-10 (normalized); \
         max_abs={max_abs_err:.3e}, k₁={k1:.3e}, normalized={normalized_err:.3e} \
         (DFT-representable mode → exact spectral derivative, Trefethen 2000 Thm. 3.1)"
    );
}

fn create_test_operator() -> SpectralDerivativeOperator {
    SpectralDerivativeOperator::new(32, 32, 32, 0.001, 0.001, 0.001)
}

#[test]
fn test_operator_creation() {
    let op = create_test_operator();
    assert_eq!(op.nx, 32);
    assert_eq!(op.ny, 32);
    assert_eq!(op.nz, 32);
}

#[test]
fn test_derivative_sinusoidal_x() {
    let op = create_test_operator();

    let mut field = Array3::zeros([32, 32, 32]);
    let k = TWO_PI / (32.0 * 0.001);

    for i in 0..32 {
        let x = i as f64 * 0.001;
        for j in 0..32 {
            for l in 0..32 {
                field[[i, j, l]] = (k * x).sin();
            }
        }
    }

    let field_view = field.view();
    let deriv = op.derivative_x(&field_view).unwrap();

    let expected_center = (k * 0.016).cos() * k;
    let computed = deriv[[16, 16, 16]];
    assert!(
        (computed - expected_center).abs() < 0.01,
        "Center point error: {} vs {}",
        computed,
        expected_center
    );
}

#[test]
fn test_derivative_output() {
    let op = create_test_operator();

    let mut field = Array3::zeros([32, 32, 32]);
    for i in 0..32 {
        let x = i as f64 * 0.001;
        for j in 0..32 {
            let y = j as f64 * 0.001;
            for l in 0..32 {
                field[[i, j, l]] =
                    (-(x - 0.016).powi(2) / 0.0001).exp() * (-(y - 0.016).powi(2) / 0.0001).exp();
            }
        }
    }

    let field_view = field.view();
    let deriv_x = op.derivative_x(&field_view).unwrap();
    let deriv_y = op.derivative_y(&field_view).unwrap();
    let deriv_z = op.derivative_z(&field_view).unwrap();

    assert_eq!(deriv_x.shape(), &[32, 32, 32]);
    assert_eq!(deriv_y.shape(), &[32, 32, 32]);
    assert_eq!(deriv_z.shape(), &[32, 32, 32]);

    assert!(deriv_x.iter().all(|&x| x.is_finite()));
    assert!(deriv_y.iter().all(|&y| y.is_finite()));
    assert!(deriv_z.iter().all(|&z| z.is_finite()));

    let center_val = deriv_x[[16, 16, 16]];
    assert!(center_val.abs() < 1.0, "Derivative values seem reasonable");
}

#[test]
fn test_derivatives_all_axes() {
    let op = create_test_operator();

    let field = Array3::from_elem([32, 32, 32], 5.0);
    let field_view = field.view();

    let dx = op.derivative_x(&field_view).unwrap();
    let dy = op.derivative_y(&field_view).unwrap();
    let dz = op.derivative_z(&field_view).unwrap();

    assert!(dx.iter().all(|&x| x.abs() < 1e-10));
    assert!(dy.iter().all(|&y| y.abs() < 1e-10));
    assert!(dz.iter().all(|&z| z.abs() < 1e-10));
}

#[test]
fn test_invalid_field_size() {
    let op = create_test_operator();
    let field = Array3::zeros([16, 32, 32]);
    let field_view = field.view();

    let result = op.derivative_x(&field_view);
    assert!(result.is_err());
}
