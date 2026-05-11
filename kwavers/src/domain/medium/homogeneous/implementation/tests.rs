use crate::domain::grid::Grid;
use crate::domain::medium::{core::CoreMedium, elastic::ElasticProperties, viscous::ViscousProperties};

use super::HomogeneousMedium;

#[test]
fn test_water_properties() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let water = HomogeneousMedium::water(&grid);

    assert_eq!(water.density(0, 0, 0), 998.0);
    assert_eq!(water.sound_speed(0, 0, 0), 1482.0);
    assert_eq!(water.viscosity(0.0, 0.0, 0.0, &grid), 1.0e-3);
}

#[test]
fn test_blood_properties() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let blood = HomogeneousMedium::blood(&grid);

    assert_eq!(blood.density(0, 0, 0), 1060.0);
    assert_eq!(blood.sound_speed(0, 0, 0), 1570.0);
    assert_eq!(blood.viscosity(0.0, 0.0, 0.0, &grid), 3.5e-3);
}

#[test]
fn test_air_properties() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let air = HomogeneousMedium::air(&grid);

    assert_eq!(air.density(0, 0, 0), 1.204);
    assert_eq!(air.sound_speed(0, 0, 0), 343.0);
}

/// Verifies the closed-form Lamé-from-wave-speeds inversion in
/// `HomogeneousMedium::elastic_homogeneous` against the analytical
/// dispersion relations for an isotropic linear elastic solid:
///
///   c_p = sqrt((λ + 2μ) / ρ)   ⇒   λ + 2μ = ρ·c_p²
///   c_s = sqrt(μ / ρ)          ⇒   μ      = ρ·c_s²
///
/// Test medium uses k-Wave example values (cp=1500, cs=800, ρ=1200).
/// # Panics
/// - Panics if `elastic_homogeneous must succeed for valid speeds`.
///
#[test]
fn test_elastic_homogeneous_lame_inversion_satisfies_dispersion() {
    let grid = Grid::new(8, 8, 8, 1e-4, 1e-4, 1e-4).unwrap();
    let cp = 1500.0_f64;
    let cs = 800.0_f64;
    let rho = 1200.0_f64;

    let med = HomogeneousMedium::elastic_homogeneous(rho, cp, cs, &grid)
        .expect("elastic_homogeneous must succeed for valid speeds");

    // Lamé parameters per closed form
    let mu = med.lame_mu_value();
    let lambda = med.lame_lambda_value();
    assert!((mu - rho * cs * cs).abs() < 1e-6, "μ mismatch: {} vs {}", mu, rho * cs * cs);
    assert!(
        (lambda - rho * (cp * cp - 2.0 * cs * cs)).abs() < 1e-6,
        "λ mismatch: {} vs {}",
        lambda,
        rho * (cp * cp - 2.0 * cs * cs),
    );

    // Dispersion-relation round-trip: ElasticProperties::compressional_wave_speed
    // and shear_wave_speed must recover the input speeds within float epsilon.
    let cp_back = med.compressional_wave_speed(0.0, 0.0, 0.0, &grid);
    let cs_back = med.shear_wave_speed(0.0, 0.0, 0.0, &grid);
    assert!((cp_back - cp).abs() < 1e-9, "c_p round-trip: got {}", cp_back);
    assert!((cs_back - cs).abs() < 1e-9, "c_s round-trip: got {}", cs_back);
}

/// Fluid limit: c_s = 0 must yield μ = 0 and λ = ρ·c_p² (acoustic bulk
/// modulus). Verifies that an elastic medium with zero shear support
/// reduces to a fluid identically.
/// # Panics
/// - Panics if `c_s = 0 must be permitted (fluid limit)`.
///
#[test]
fn test_elastic_homogeneous_fluid_limit_zero_shear_speed() {
    let grid = Grid::new(8, 8, 8, 1e-4, 1e-4, 1e-4).unwrap();
    let cp = 1500.0_f64;
    let rho = 1000.0_f64;

    let med = HomogeneousMedium::elastic_homogeneous(rho, cp, 0.0, &grid)
        .expect("c_s = 0 must be permitted (fluid limit)");
    assert_eq!(med.lame_mu_value(), 0.0);
    assert!(
        (med.lame_lambda_value() - rho * cp * cp).abs() < 1e-6,
        "λ should equal bulk modulus K = ρ·c_p² in fluid limit; got {}",
        med.lame_lambda_value(),
    );
    assert!(med.shear_wave_speed(0.0, 0.0, 0.0, &grid).abs() < 1e-12);
    assert!((med.compressional_wave_speed(0.0, 0.0, 0.0, &grid) - cp).abs() < 1e-9);
}

/// Stability bound: c_s > c_p / √2 violates λ ≥ 0 (Poisson ratio ν < 0
/// auxetic regime is rejected by `elastic_homogeneous`). The constructor
/// must return `None` rather than producing an unstable medium.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_elastic_homogeneous_rejects_unstable_speeds() {
    let grid = Grid::new(4, 4, 4, 1e-4, 1e-4, 1e-4).unwrap();
    // c_s² · 2 > c_p² ⇒ λ < 0 — must reject
    let res = HomogeneousMedium::elastic_homogeneous(1000.0, 1500.0, 1200.0, &grid);
    assert!(res.is_none(), "Unstable elastic configuration must be rejected");

    // Density / speed positivity
    assert!(HomogeneousMedium::elastic_homogeneous(0.0, 1500.0, 800.0, &grid).is_none());
    assert!(HomogeneousMedium::elastic_homogeneous(1000.0, 0.0, 800.0, &grid).is_none());
    assert!(HomogeneousMedium::elastic_homogeneous(1000.0, 1500.0, -1.0, &grid).is_none());

    // NaN / Inf rejection
    assert!(HomogeneousMedium::elastic_homogeneous(f64::NAN, 1500.0, 800.0, &grid).is_none());
    assert!(HomogeneousMedium::elastic_homogeneous(1000.0, f64::INFINITY, 800.0, &grid).is_none());
}

/// `set_lame_parameters` must reject non-finite, negative-μ, or negative-λ
/// values; valid pairs must be applied verbatim.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_set_lame_parameters_validation_and_assignment() {
    let grid = Grid::new(4, 4, 4, 1e-4, 1e-4, 1e-4).unwrap();
    let mut med = HomogeneousMedium::water(&grid);

    // Valid assignment is applied verbatim
    med.set_lame_parameters(2.2e9, 0.0).unwrap();
    assert_eq!(med.lame_lambda_value(), 2.2e9);
    assert_eq!(med.lame_mu_value(), 0.0);

    // Negative μ rejected
    assert!(med.set_lame_parameters(2.2e9, -1.0).is_err());
    // Negative λ rejected
    assert!(med.set_lame_parameters(-1.0, 1.0e9).is_err());
    // Non-finite rejected
    assert!(med.set_lame_parameters(f64::NAN, 1.0e9).is_err());
    assert!(med.set_lame_parameters(1.0e9, f64::INFINITY).is_err());

    // Original valid values must remain untouched after rejected setter calls
    assert_eq!(med.lame_lambda_value(), 2.2e9);
    assert_eq!(med.lame_mu_value(), 0.0);
}
