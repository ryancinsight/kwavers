use super::*;
use crate::core::error::{KwaversError, KwaversResult};

fn assert_invalid_dimension<T>(result: KwaversResult<T>, dim: usize) {
    match result {
        Err(KwaversError::InvalidInput(message)) => {
            assert_eq!(message, format!("CPML dimension {dim} out of range [0, 2]"));
        }
        Err(error) => panic!("expected invalid-dimension error, got {error:?}"),
        Ok(_) => panic!("expected invalid-dimension error for dim={dim}"),
    }
}

/// `PerDimensionPML::get` must return `Err` for dimension > 2, not panic.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_per_dimension_pml_invalid_dim_returns_err() {
    let pml = PerDimensionPML::uniform(20);
    assert_eq!(pml.get(0).unwrap(), 20);
    assert_eq!(pml.get(1).unwrap(), 20);
    assert_eq!(pml.get(2).unwrap(), 20);
    assert_invalid_dimension(pml.get(3), 3);
    assert_invalid_dimension(pml.get(99), 99);
}

/// `PerDimensionAlpha::get` must return `Err` for dimension > 2, not panic.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_per_dimension_alpha_invalid_dim_returns_err() {
    let alpha = PerDimensionAlpha::uniform(2.0);
    assert_eq!(alpha.get(0).unwrap().to_bits(), 2.0_f64.to_bits());
    assert_eq!(alpha.get(1).unwrap().to_bits(), 2.0_f64.to_bits());
    assert_eq!(alpha.get(2).unwrap().to_bits(), 2.0_f64.to_bits());
    assert_invalid_dimension(alpha.get(3), 3);
}

/// `CPMLConfig::sigma_factor_for_dimension` and `thickness_for_dimension`
/// must propagate the error from `PerDimensionAlpha::get` / `PerDimensionPML::get`.
#[test]
fn test_cpml_config_dimension_methods_invalid_dim_returns_err() {
    let cfg = CPMLConfig::default();
    assert_invalid_dimension(cfg.sigma_factor_for_dimension(3), 3);
    assert_invalid_dimension(cfg.thickness_for_dimension(3), 3);
    assert_invalid_dimension(
        cfg.theoretical_reflection_for_dimension(3, 1.0, 1e-3, 1500.0),
        3,
    );
}

#[test]
fn test_cpml_config_dimension_methods_preserve_axis_values() {
    let cfg = CPMLConfig::default()
        .with_pml_size(4, 7, 11)
        .with_alpha_xyz(1.25, 2.5, 5.0);

    assert_eq!(cfg.thickness_for_dimension(0).unwrap(), 4);
    assert_eq!(cfg.thickness_for_dimension(1).unwrap(), 7);
    assert_eq!(cfg.thickness_for_dimension(2).unwrap(), 11);
    assert_eq!(
        cfg.sigma_factor_for_dimension(0).unwrap().to_bits(),
        1.25_f64.to_bits()
    );
    assert_eq!(
        cfg.sigma_factor_for_dimension(1).unwrap().to_bits(),
        2.5_f64.to_bits()
    );
    assert_eq!(
        cfg.sigma_factor_for_dimension(2).unwrap().to_bits(),
        5.0_f64.to_bits()
    );
    assert_eq!(cfg.sigma_factor.to_bits(), 5.0_f64.to_bits());
}

#[test]
fn test_theoretical_reflection_for_dimension_uses_axis_thickness() {
    let cfg = CPMLConfig::default().with_pml_size(2, 3, 5).with_alpha(2.0);
    let cos_theta = 0.5;
    let dx = 1.0e-3;
    let sound_speed = 1500.0;
    let m = cfg.polynomial_order;
    let sigma_max =
        cfg.sigma_factor * (m + 1.0) * sound_speed / (150.0 * std::f64::consts::PI * dx);

    for (dim, thickness) in [(0, 2.0), (1, 3.0), (2, 5.0)] {
        let expected =
            cfg.target_reflection * (-(m + 1.0) * sigma_max * thickness * cos_theta).exp();
        let actual = cfg
            .theoretical_reflection_for_dimension(dim, cos_theta, dx, sound_speed)
            .unwrap();
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}
