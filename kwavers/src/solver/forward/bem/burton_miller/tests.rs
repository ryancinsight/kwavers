use super::assembler::BurtonMillerAssembler;
use super::config::BurtonMillerConfig;
use num_complex::Complex64;

#[test]
fn test_burton_miller_config_creation() {
    let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
    assert!(cfg.wavenumber > 0.0);
    assert!(cfg.coupling_alpha.norm() > 0.0);
}

#[test]
fn test_burton_miller_config_custom_alpha() {
    let cfg =
        BurtonMillerConfig::new(100_000.0, 1500.0).with_coupling_alpha(Complex64::new(0.0, 1.0));
    assert_eq!(cfg.coupling_alpha, Complex64::new(0.0, 1.0));
}

#[test]
fn test_assembler_creation() {
    let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
    let _assembler = BurtonMillerAssembler::new(cfg);
}

#[test]
fn test_greens_function_helmholtz() {
    let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
    let assembler = BurtonMillerAssembler::new(cfg);
    let g = assembler.greens_function_helmholtz(cfg.wavenumber, 0.1);
    assert!(!g.re.is_nan() && !g.im.is_nan());
    assert!(g.norm() > 0.0);
}

#[test]
fn test_triangle_normal() {
    let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
    let assembler = BurtonMillerAssembler::new(cfg);
    let n1 = [0.0, 0.0, 0.0];
    let n2 = [1.0, 0.0, 0.0];
    let n3 = [0.0, 1.0, 0.0];
    let normal = assembler.triangle_normal(n1, n2, n3);
    assert!((normal[2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_triangle_area() {
    let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
    let assembler = BurtonMillerAssembler::new(cfg);
    let n1 = [0.0, 0.0, 0.0];
    let n2 = [1.0, 0.0, 0.0];
    let n3 = [0.0, 1.0, 0.0];
    let area = assembler.triangle_area(n1, n2, n3);
    assert!((area - 0.5).abs() < 1e-10);
}

#[test]
fn test_distance() {
    let cfg = BurtonMillerConfig::new(100_000.0, 1500.0);
    let assembler = BurtonMillerAssembler::new(cfg);
    let p1 = [0.0, 0.0, 0.0];
    let p2 = [3.0, 4.0, 0.0];
    let dist = assembler.distance(&p1, &p2);
    assert!((dist - 5.0).abs() < 1e-10);
}

/// Parallel normals n_x = n_y = ẑ, field point y above collocation x.
/// ∂²G/(∂n_x ∂n_y) = G * [k² + 2ik/r − 2/r²]
#[test]
fn test_hypersingular_parallel_normals_matches_formula() {
    let k = 2.0_f64;
    let r = 1.0_f64;
    let cfg = BurtonMillerConfig::new(1000.0, 1500.0);
    let assembler = BurtonMillerAssembler::new(cfg);
    let collocation = [0.0_f64, 0.0, 0.0];
    let point = [0.0_f64, 0.0, r];
    let nx = [0.0_f64, 0.0, 1.0];
    let ny = [0.0_f64, 0.0, 1.0];
    let result =
        assembler.greens_function_double_normal_derivative(k, r, &collocation, &point, &ny, &nx);
    let g = assembler.greens_function_helmholtz(k, r);
    let expected = g * Complex64::new(k * k - 2.0 / (r * r), 2.0 * k / r);
    let diff = (result - expected).norm();
    assert!(
        diff < 1e-12,
        "parallel normals: |result − expected| = {:.3e}",
        diff
    );
}

/// Static limit k=0: ∂²G/(∂n_x ∂n_y) = (nxny − 3coscos)/(4πr³).
#[test]
fn test_hypersingular_static_limit_matches_dipole_kernel() {
    let k = 0.0_f64;
    let r = 2.0_f64;
    let cfg = BurtonMillerConfig::new(1.0, 1.0);
    let assembler = BurtonMillerAssembler::new(cfg);
    let collocation = [0.0_f64, 0.0, 0.0];
    let point = [r, 0.0, 0.0];
    let nx = [0.0_f64, 0.0, 1.0];
    let ny = [0.0_f64, 0.0, 1.0];
    let result =
        assembler.greens_function_double_normal_derivative(k, r, &collocation, &point, &ny, &nx);
    let g_static = 1.0 / (4.0 * std::f64::consts::PI * r);
    let expected = Complex64::new(g_static / (r * r), 0.0);
    let diff = (result - expected).norm();
    assert!(
        diff < 1e-14,
        "static limit: |result − expected| = {:.3e}",
        diff
    );
}

/// Perpendicular normals → result ≈ 0.
#[test]
fn test_hypersingular_all_perpendicular_is_zero() {
    let k = 5.0_f64;
    let r = 1.5_f64;
    let cfg = BurtonMillerConfig::new(1000.0, 1500.0);
    let assembler = BurtonMillerAssembler::new(cfg);
    let collocation = [0.0_f64, 0.0, 0.0];
    let point = [r, 0.0, 0.0];
    let nx = [0.0_f64, 1.0, 0.0];
    let ny = [0.0_f64, 0.0, 1.0];
    let result =
        assembler.greens_function_double_normal_derivative(k, r, &collocation, &point, &ny, &nx);
    assert!(
        result.norm() < 1e-14,
        "all-perpendicular: expected 0, got {:.3e}",
        result.norm()
    );
}

/// Near-singularity guard: r < 1e-10 returns exactly zero.
#[test]
fn test_hypersingular_near_singular_returns_zero() {
    let cfg = BurtonMillerConfig::new(1000.0, 1500.0);
    let assembler = BurtonMillerAssembler::new(cfg);
    let collocation = [0.0_f64, 0.0, 0.0];
    let point = [1e-11_f64, 0.0, 0.0];
    let r = 1e-11_f64;
    let n = [0.0_f64, 0.0, 1.0];
    let result =
        assembler.greens_function_double_normal_derivative(2.0, r, &collocation, &point, &n, &n);
    assert_eq!(result, Complex64::new(0.0, 0.0));
}
