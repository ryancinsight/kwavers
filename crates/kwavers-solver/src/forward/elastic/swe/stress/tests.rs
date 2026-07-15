use super::*;
use kwavers_grid::Grid;
use leto::Array3;

/// fd1_x on a linear field f = A*x gives the constant A at all interior points.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_fd1_x_linear_field() {
    let nx = 10;
    let dx = 0.001;
    let a = 3.0_f64;
    let mut f = Array3::<f64>::zeros((nx, 1, 1));
    for i in 0..nx {
        f[[i, 0, 0]] = a * (i as f64) * dx;
    }
    // Interior points: derivative should equal A.
    for i in 2..nx - 2 {
        let d = fd1_x(f.view(), i, 0, 0, nx, dx);
        assert!(
            (d - a).abs() < 1e-10,
            "fd1_x at i={i}: got {d}, expected {a}"
        );
    }
}

/// fd1_y on a linear field f = B*y gives the constant B at all interior points.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_fd1_y_linear_field() {
    let ny = 10;
    let dy = 0.001;
    let b = -2.5_f64;
    let mut f = Array3::<f64>::zeros((1, ny, 1));
    for j in 0..ny {
        f[[0, j, 0]] = b * (j as f64) * dy;
    }
    for j in 2..ny - 2 {
        let d = fd1_y(f.view(), 0, j, 0, ny, dy);
        assert!(
            (d - b).abs() < 1e-10,
            "fd1_y at j={j}: got {d}, expected {b}"
        );
    }
}

/// fd1_z on a linear field f = C*z gives the constant C at all interior points.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_fd1_z_linear_field() {
    let nz = 10;
    let dz = 0.001;
    let c = 1.7_f64;
    let mut f = Array3::<f64>::zeros((1, 1, nz));
    for k in 0..nz {
        f[[0, 0, k]] = c * (k as f64) * dz;
    }
    for k in 2..nz - 2 {
        let d = fd1_z(f.view(), 0, 0, k, nz, dz);
        assert!(
            (d - c).abs() < 1e-10,
            "fd1_z at k={k}: got {d}, expected {c}"
        );
    }
}

/// Degenerate axis (size=1) must return 0.0 without panic.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_fd1_degenerate_axes() {
    let f = Array3::<f64>::ones((1, 1, 1));
    assert_eq!(fd1_x(f.view(), 0, 0, 0, 1, 0.001), 0.0);
    assert_eq!(fd1_y(f.view(), 0, 0, 0, 1, 0.001), 0.0);
    assert_eq!(fd1_z(f.view(), 0, 0, 0, 1, 0.001), 0.0);
}

/// Uniform displacement ux = A (constant) → zero stress divergence.
///
/// ## Numerical note
///
/// Displacement values must be exactly representable in f64 (binary fractions such as
/// 0.5 = 2⁻¹, 0.25 = 2⁻², 0.125 = 2⁻³).  Non-binary values (e.g. 0.3, 0.1) are not
/// representable exactly; the 4th-order interior stencil and the 1st/2nd-order boundary
/// stencils then produce *different* ULP-level rounding errors, so the stress arrays
/// are not numerically constant across j/k even though they should be physically zero.
/// The resulting non-constant stress produces a spurious FD-of-stress value ≈ 0.024
/// that far exceeds any physically meaningful tolerance.  Exact binary fractions cancel
/// identically in all stencil variants, giving div = 0.0 to floating-point precision.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_stress_divergence_uniform_displacement() {
    let n = 10;
    let dx = 0.001;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let lambda = Array3::from_elem([n, n, n], 1e9_f64);
    let mu = Array3::from_elem([n, n, n], 5e8_f64);
    use super::super::types::ElasticWaveField;
    let mut field = ElasticWaveField::new(n, n, n);
    // Use exact binary fractions: 0.5=2⁻¹, 0.25=2⁻², 0.125=2⁻³.
    // Non-binary values produce stencil-dependent ULP rounding that makes
    // stress spatially non-uniform at the 1e-5 level when multiplied by λ,μ∼1e9.
    field.ux.fill(0.5);
    field.uy.fill(0.25);
    field.uz.fill(0.125);
    let (dx_arr, dy_arr, dz_arr) = stress_divergence(&grid, &lambda, &mu, &field);
    for k in 2..n - 2 {
        for j in 2..n - 2 {
            for i in 2..n - 2 {
                // With exact binary fractions, all FD stencils cancel exactly → 0.0.
                // The tolerance 1e-10 guards against any unexpected ULP drift.
                assert!(
                    dx_arr[[i, j, k]].abs() < 1e-10,
                    "div_x at ({i},{j},{k}) = {}",
                    dx_arr[[i, j, k]]
                );
                assert!(
                    dy_arr[[i, j, k]].abs() < 1e-10,
                    "div_y at ({i},{j},{k}) = {}",
                    dy_arr[[i, j, k]]
                );
                assert!(
                    dz_arr[[i, j, k]].abs() < 1e-10,
                    "div_z at ({i},{j},{k}) = {}",
                    dz_arr[[i, j, k]]
                );
            }
        }
    }
}

/// Linear ux = A·x in a homogeneous medium with μ=0 (fluid).
///
/// εxx = A (constant) → σxx = (λ+2μ)·A (constant) → ∂σxx/∂x = 0.
/// Acceleration must be zero everywhere in the interior.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_stress_divergence_linear_ux_fluid() {
    let n = 12;
    let dx = 1e-3;
    let grid = Grid::new(n, 1, 1, dx, dx, dx).unwrap();
    let la_val = 2.25e9_f64; // water-like λ
    let lambda = Array3::from_elem([n, 1, 1], la_val);
    let mu = Array3::zeros((n, 1, 1)); // fluid: μ=0
    use super::super::types::ElasticWaveField;
    let mut field = ElasticWaveField::new(n, 1, 1);
    // Linear displacement ux = A·x → constant strain → constant σxx → zero divergence
    let a = 0.01_f64;
    for i in 0..n {
        field.ux[[i, 0, 0]] = a * (i as f64) * dx;
    }
    let (div_x, div_y, div_z) = stress_divergence(&grid, &lambda, &mu, &field);
    for i in 2..n - 2 {
        assert!(
            div_x[[i, 0, 0]].abs() < 1e-3,
            "div_x at i={i} = {}",
            div_x[[i, 0, 0]]
        );
        assert_eq!(div_y[[i, 0, 0]], 0.0);
        assert_eq!(div_z[[i, 0, 0]], 0.0);
    }
}

/// Quadratic ux = A·x² in a homogeneous fluid.
///
/// εxx = 2A·x → σxx = (λ)·2A·x → ∂σxx/∂x = 2λA (constant).
/// The interior acceleration a_x = 2λA/ρ must match within FD error.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_stress_divergence_quadratic_ux_fluid() {
    let n = 12;
    let dx = 1e-3;
    let grid = Grid::new(n, 1, 1, dx, dx, dx).unwrap();
    let la_val = 2.25e9_f64;
    let lambda = Array3::from_elem([n, 1, 1], la_val);
    let mu = Array3::zeros((n, 1, 1));
    use super::super::types::ElasticWaveField;
    let mut field = ElasticWaveField::new(n, 1, 1);
    let a = 10.0_f64;
    for i in 0..n {
        let x = (i as f64) * dx;
        field.ux[[i, 0, 0]] = a * x * x;
    }
    let expected = 2.0 * la_val * a; // ∂σxx/∂x = 2λA
    let (div_x, _, _) = stress_divergence(&grid, &lambda, &mu, &field);
    for i in 3..n - 3 {
        let got = div_x[[i, 0, 0]];
        let rel_err = (got - expected).abs() / expected.abs();
        assert!(
            rel_err < 1e-4,
            "div_x at i={i}: got {got:.6e}, expected {expected:.6e}, rel_err={rel_err:.2e}"
        );
    }
}
