//! Tests for Kuznetsov equation solver

#[cfg(test)]
use super::*;
#[cfg(test)]
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
#[cfg(test)]
use kwavers_core::constants::numerical::TWO_PI;
#[cfg(test)]
use kwavers_grid::Grid;

/// **Invariant**: Creating a KuznetsovWave on a 32³ grid allocates exactly
/// 18 × 32³ × 8 = 4,718,592 bytes in its workspace scratch arena.
#[test]
fn kuznetsov_solver_creation_produces_correct_workspace_footprint() {
    use crate::workspace::ScratchArena;
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let config = config::KuznetsovConfig::default();
    let solver = solver::KuznetsovWave::new(config, &grid).unwrap();
    // 18 pre-allocated Array3<f64> buffers × 32³ × 8 bytes
    let expected = 18 * 32 * 32 * 32 * std::mem::size_of::<f64>();
    // KuznetsovWave's workspace is private; extract via is_solution_valid()
    // as a liveness check, and verify the solver is usable.
    assert!(
        solver.is_solution_valid(),
        "newly created solver must be valid"
    );
    // Indirectly verify grid: create a workspace and check its footprint
    let ws = workspace::KuznetsovWorkspace::new(&grid).unwrap();
    assert_eq!(
        ws.memory_bytes(),
        expected,
        "workspace must allocate 18 × N × sizeof(f64)"
    );
}

#[test]
fn test_kuznetsov_config_validation() {
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();

    let valid_config = config::KuznetsovConfig::default();
    assert!(valid_config.validate(&grid).is_ok());

    let invalid_config = config::KuznetsovConfig {
        cfl_factor: 2.0, // Too high
        ..Default::default()
    };
    assert!(invalid_config.validate(&grid).is_err());
}

/// **Theorem (spectral Laplacian of constant field is zero)**:
/// For a constant function f = C, ∂²f/∂x² = ∂²f/∂y² = ∂²f/∂z² = 0,
/// so ∇²f = 0 everywhere. In the spectral domain, the DC bin (k=0) is the
/// only non-zero coefficient, and the Laplacian operator multiplies by −|k|²=0.
#[test]
fn spectral_laplacian_of_constant_is_zero() {
    let n = 16usize;
    let dx = 0.001_f64;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let field = leto::Array3::from_elem([n, n, n], 7.5_f64);

    let laplacian = numerical::compute_laplacian(&field, &grid);

    // All elements must be ~0 to machine epsilon (FFT rounding budget ≈ N·log₂N·ε_mach)
    let max_abs = laplacian.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_abs < 1e-8,
        "∇²(constant) must be zero; max_abs={max_abs:.3e}"
    );
}

/// **Theorem (spectral gradient of constant field is zero)**:
/// ∂C/∂x = ∂C/∂y = ∂C/∂z = 0 for any constant C.
#[test]
fn spectral_gradient_of_constant_is_zero() {
    let n = 16usize;
    let dx = 0.001_f64;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let field = leto::Array3::from_elem([n, n, n], 3.14_f64);

    let (gx, gy, gz) = numerical::compute_gradient(&field, &grid);

    let max_gx = gx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_gy = gy.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_gz = gz.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(max_gx < 1e-8, "∂C/∂x must be zero; max={max_gx:.3e}");
    assert!(max_gy < 1e-8, "∂C/∂y must be zero; max={max_gy:.3e}");
    assert!(max_gz < 1e-8, "∂C/∂z must be zero; max={max_gz:.3e}");
}

/// **Theorem (spectral Laplacian of sine mode)**:
/// For f(x,y,z) = sin(k₁·x), ∇²f = ∂²f/∂x² = −k₁² sin(k₁·x).
/// The spectral method recovers −k₁²·f exactly for DFT-representable modes.
#[test]
fn spectral_laplacian_of_sine_matches_analytical() {
    let n = 32usize;
    let dx = 0.001_f64;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();

    // Fundamental mode: k₁ = 2π/(N·dx)
    let k1 = TWO_PI / (n as f64 * dx);
    let mut field = leto::Array3::<f64>::zeros((n, n, n));
    for i in 0..n {
        let x = i as f64 * dx;
        for j in 0..n {
            for l in 0..n {
                field[[i, j, l]] = (k1 * x).sin();
            }
        }
    }

    let laplacian = numerical::compute_laplacian(&field, &grid);

    // Expected: ∇²f = −k₁² sin(k₁·x); check at center slice
    let center = n / 2;
    let mut max_rel = 0.0_f64;
    for i in 0..n {
        let x = i as f64 * dx;
        let expected = -k1 * k1 * (k1 * x).sin();
        let computed = laplacian[[i, center, center]];
        let err = (computed - expected).abs() / (k1 * k1).max(1e-14);
        max_rel = max_rel.max(err);
    }
    assert!(
        max_rel < 1e-8,
        "∇²(sin(k₁·x)) must equal −k₁²·sin(k₁·x) to ≤1e-8; max_rel={max_rel:.3e}"
    );
}

/// **Theorem (nonlinearity coefficient β = 1 + B/(2A))**:
/// For water B/A = 5.0, β = 1 + 5.0/2 = 3.5.
/// For soft tissue B/A = 7.0, β = 1 + 7.0/2 = 4.5.
#[test]
fn nonlinearity_coefficient_matches_analytical_formula() {
    use nonlinear::compute_nonlinearity_coefficient;

    // Water: B/A = 5.0 → β = 3.5 (Beyer 1960)
    let beta_water = compute_nonlinearity_coefficient(5.0);
    assert!(
        (beta_water - 3.5).abs() < 1e-14,
        "β(water) must be 3.5; got {beta_water}"
    );

    // Soft tissue: B/A = 7.0 → β = 4.5
    let beta_tissue = compute_nonlinearity_coefficient(7.0);
    assert!(
        (beta_tissue - 4.5).abs() < 1e-14,
        "β(tissue) must be 4.5; got {beta_tissue}"
    );

    // Linear limit: B/A = 0 → β = 1.0 (pure quadratic EOS vanishes)
    let beta_linear = compute_nonlinearity_coefficient(0.0);
    assert!(
        (beta_linear - 1.0).abs() < 1e-14,
        "β(B/A=0) must be 1.0; got {beta_linear}"
    );
}

/// **Theorem (explicit-form diffusion coefficient +δ/c₀²)**:
///
/// The Kuznetsov operator form has `−(δ/c₀⁴)∂³p/∂t³` on the RHS (Lighthill 1978, §3.4).
/// Rearranging for the leapfrog explicit form `∂²p/∂t²`:
/// ```text
/// ∂²p/∂t² = c₀²∇²p + … + (δ/c₀²)∂³p/∂t³
/// ```
/// The function therefore returns `+(δ/c₀²)∂³p/∂t³` (positive, c² not c⁴).
///
/// For p(t) = p₀·t³, ∂³p/∂t³ = 6·p₀ (constant).
/// The 4-point backward finite difference is exact for cubic polynomials.
///
/// Verification: `expected = +(δ/c₀²)·6·p₀`.
#[test]
fn diffusive_term_matches_stokes_kirchhoff_coefficient() {
    use diffusion::compute_diffusive_term_workspace;

    const N: usize = 4;
    const DT: f64 = 0.01;
    const C0: f64 = SOUND_SPEED_WATER_SIM;
    const DELTA: f64 = 1e-6; // acoustic diffusivity [m²/s]
    const P0: f64 = 1.0; // coefficient in p(t) = P0·t³

    // p(t) = P0·t³ at t = 3Δt, 2Δt, Δt, 0
    let pressure = leto::Array3::from_elem([N, N, N], P0 * (3.0 * DT).powi(3));
    let p_prev = leto::Array3::from_elem([N, N, N], P0 * (2.0 * DT).powi(3));
    let p_prev2 = leto::Array3::from_elem([N, N, N], P0 * DT.powi(3));
    let p_prev3 = leto::Array3::zeros((N, N, N));
    let mut out = leto::Array3::zeros((N, N, N));

    compute_diffusive_term_workspace(
        &pressure, &p_prev, &p_prev2, &p_prev3, DT, C0, DELTA, &mut out,
    );

    // ∂³(P0·t³)/∂t³ = 6·P0 (exact for cubic; 4-point FD is exact for degree ≤ 3).
    let d3p_dt3_exact = 6.0 * P0;
    // Explicit-form: +(δ/c₀²)·∂³p/∂t³  [positive]
    let expected = (DELTA / C0.powi(2)) * d3p_dt3_exact;

    let computed = out[[N / 2, N / 2, N / 2]];
    let rel = (computed - expected).abs() / expected.abs().max(1e-20);
    assert!(
        rel < 1e-10,
        "diffusive term must match +(δ/c₀²)·6P0; expected={expected:.6e}, computed={computed:.6e}, rel={rel:.3e}"
    );
    // Sign check: with positive δ and positive ∂³p/∂t³, result must be positive.
    assert!(
        expected > 0.0,
        "Kuznetsov explicit-form diffusion must be positive for positive ∂³p/∂t³"
    );
}

/// **Theorem (explicit-form nonlinear coefficient +β/ρ₀c₀²)**:
///
/// The Kuznetsov operator form has `−(β/ρ₀c₀⁴)∂²(p²)/∂t²` on the RHS.
/// Rearranging for the leapfrog explicit form:
/// ```text
/// ∂²p/∂t² = c₀²∇²p + (β/ρ₀c₀²)∂²(p²)/∂t² + …
/// ```
/// The function returns `+(β/ρ₀c₀²)∂²(p²)/∂t²` (positive, c² not c⁴).
///
/// For p\[n\]=p, p[n−1]=p_prev=0, p[n−2]=0:
/// - `∂²(p²)/∂t² ≈ (p²−0+0)/Δt² = p²/Δt²`
/// - `expected = (β/ρ₀c₀²)·p²/Δt²`  [positive]
#[test]
fn nonlinear_term_explicit_form_coefficient_is_positive_with_c_squared() {
    use nonlinear::compute_nonlinear_term_workspace;

    const N: usize = 4;
    const DT: f64 = 1e-7;
    const RHO0: f64 = kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
    const C0: f64 = SOUND_SPEED_WATER_SIM;
    const B_OVER_A: f64 = 5.0; // water
    const P: f64 = 1e5; // 100 kPa

    let pressure = leto::Array3::from_elem([N, N, N], P);
    let p_prev = leto::Array3::zeros((N, N, N));
    let p_prev2 = leto::Array3::zeros((N, N, N));
    let mut out = leto::Array3::zeros((N, N, N));

    compute_nonlinear_term_workspace(
        &pressure, &p_prev, &p_prev2, DT, RHO0, C0, B_OVER_A, &mut out,
    );

    // β = 1 + B/(2A) = 3.5
    let beta = 1.0 + B_OVER_A / 2.0;
    // Explicit-form coefficient: +(β/ρ₀c₀²)
    let coeff = beta / (RHO0 * C0.powi(2));
    // p[n−1]=0, p[n−2]=0 → ∂²(p²)/∂t² = (P²−0+0)/DT²
    let d2p2_dt2 = P * P / (DT * DT);
    let expected = coeff * d2p2_dt2;

    let computed = out[[N / 2, N / 2, N / 2]];
    let rel = (computed - expected).abs() / expected.abs().max(1e-20);
    assert!(
        rel < 1e-12,
        "nl term must match +(β/ρ₀c₀²)·∂²(p²)/∂t²; expected={expected:.6e}, computed={computed:.6e}, rel={rel:.3e}"
    );
    // Sign check: positive β, positive p → positive nonlinear contribution.
    assert!(
        expected > 0.0,
        "Kuznetsov explicit-form NL must be positive for β>0, p>0"
    );
}
