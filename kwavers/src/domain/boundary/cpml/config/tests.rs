use super::*;

/// `PerDimensionPML::get` must return `Err` for dimension > 2, not panic.
#[test]
fn test_per_dimension_pml_invalid_dim_returns_err() {
    let pml = PerDimensionPML::uniform(20);
    assert!(pml.get(0).is_ok(), "dim=0 must succeed");
    assert!(pml.get(1).is_ok(), "dim=1 must succeed");
    assert!(pml.get(2).is_ok(), "dim=2 must succeed");
    assert!(pml.get(3).is_err(), "dim=3 must return Err");
    assert!(pml.get(99).is_err(), "dim=99 must return Err");
}

/// `PerDimensionAlpha::get` must return `Err` for dimension > 2, not panic.
#[test]
fn test_per_dimension_alpha_invalid_dim_returns_err() {
    let alpha = PerDimensionAlpha::uniform(2.0);
    assert!(alpha.get(0).is_ok(), "dim=0 must succeed");
    assert!(alpha.get(1).is_ok(), "dim=1 must succeed");
    assert!(alpha.get(2).is_ok(), "dim=2 must succeed");
    assert!(alpha.get(3).is_err(), "dim=3 must return Err");
}

/// `CPMLConfig::sigma_factor_for_dimension` and `thickness_for_dimension`
/// must propagate the error from `PerDimensionAlpha::get` / `PerDimensionPML::get`.
#[test]
fn test_cpml_config_dimension_methods_invalid_dim_returns_err() {
    let cfg = CPMLConfig::default();
    assert!(cfg.sigma_factor_for_dimension(3).is_err());
    assert!(cfg.thickness_for_dimension(3).is_err());
    assert!(cfg
        .theoretical_reflection_for_dimension(3, 1.0, 1e-3, 1500.0)
        .is_err());
}
