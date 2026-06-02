//! Tests for PML boundary conditions.

use super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_domain::grid::Grid;
use ndarray::Array3;

#[test]
fn test_pml_creation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let config = SwePmlConfig::default();
    let pml = ElasticSwePMLBoundary::new(&grid, config);

    assert_eq!(pml.attenuation(16, 16, 16), 0.0);
    assert!(!pml.is_in_pml(16, 16, 16));

    assert!(pml.attenuation(0, 16, 16) > 0.0);
    assert!(pml.is_in_pml(0, 16, 16));
}

#[test]
fn test_pml_profile() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let config = SwePmlConfig {
        thickness: 5,
        sigma_max: 100.0,
        profile_order: 2,
        reflection_target: 1e-5,
    };
    let pml = ElasticSwePMLBoundary::new(&grid, config);

    let sigma_1 = pml.attenuation(4, 16, 16);
    let sigma_2 = pml.attenuation(3, 16, 16);
    let sigma_3 = pml.attenuation(0, 16, 16);

    assert!(sigma_1 < sigma_2);
    assert!(sigma_2 < sigma_3);
    assert!((sigma_3 - 100.0).abs() < 1e-10);
}

#[test]
fn test_pml_damping() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let config = SwePmlConfig::default();
    let pml = ElasticSwePMLBoundary::new(&grid, config);

    let mut vx = Array3::<f64>::ones((32, 32, 32));
    let mut vy = Array3::<f64>::ones((32, 32, 32));
    let mut vz = Array3::<f64>::ones((32, 32, 32));

    let dt = 1e-7;
    pml.apply_damping(&mut vx, &mut vy, &mut vz, dt);

    assert!((vx[[16, 16, 16]] - 1.0).abs() < 1e-10);
    assert!(vx[[0, 16, 16]] < 1.0);
    assert!(vx[[0, 16, 16]] > 0.0);
}

#[test]
fn test_theoretical_reflection() {
    // R = exp(-2 * σ_max * L_pml / c_max)
    // optimize_sigma_max inverts this formula.
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let c_max = SOUND_SPEED_WATER_SIM;
    let thickness = 10;
    let target_reflection = 0.005;

    let sigma_optimized =
        ElasticSwePMLBoundary::optimize_sigma_max(target_reflection, c_max, &grid, thickness);

    let config = SwePmlConfig {
        thickness,
        sigma_max: sigma_optimized,
        profile_order: 2,
        reflection_target: target_reflection,
    };
    let pml = ElasticSwePMLBoundary::new(&grid, config);
    let reflection = pml.theoretical_reflection(c_max, &grid);

    assert!(
        reflection < 0.01,
        "Reflection {} exceeds 1% threshold",
        reflection
    );
    assert!(
        reflection > 0.0,
        "Reflection {} must be positive",
        reflection
    );
    assert!(
        (reflection - target_reflection).abs() / target_reflection < 0.01,
        "Reflection {} differs from target {} by more than 1%",
        reflection,
        target_reflection
    );
}

#[test]
fn test_sigma_max_optimization() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let target_reflection = 1e-6;
    let c_max = SOUND_SPEED_WATER_SIM;
    let thickness = 10;

    let sigma_opt =
        ElasticSwePMLBoundary::optimize_sigma_max(target_reflection, c_max, &grid, thickness);
    assert!(sigma_opt > 0.0);

    let config = SwePmlConfig {
        thickness,
        sigma_max: sigma_opt,
        profile_order: 2,
        reflection_target: target_reflection,
    };
    let pml = ElasticSwePMLBoundary::new(&grid, config);
    let achieved_reflection = pml.theoretical_reflection(c_max, &grid);

    assert!((achieved_reflection.log10() - target_reflection.log10()).abs() < 1.0);
}

#[test]
fn test_pml_volume_fraction() {
    // Grid: 50^3; PML thickness: 5 pts.
    // Interior: (50-10)^3 = 64000 points; total: 125000.
    // PML fraction: 1 - 64000/125000 = 0.488.
    let grid = Grid::new(50, 50, 50, 1e-3, 1e-3, 1e-3).unwrap();
    let config = SwePmlConfig {
        thickness: 5,
        ..Default::default()
    };
    let pml = ElasticSwePMLBoundary::new(&grid, config);
    let vol_frac = pml.volume_fraction();

    assert!(vol_frac > 0.0);
    assert!(vol_frac < 1.0);
    assert!(vol_frac > 0.3, "PML volume fraction {} too small", vol_frac);
    assert!(
        vol_frac < 0.6,
        "PML volume fraction {} exceeds 60% threshold",
        vol_frac
    );
}

#[test]
fn test_pml_mask() {
    let grid = Grid::new(20, 20, 20, 1e-3, 1e-3, 1e-3).unwrap();
    let config = SwePmlConfig {
        thickness: 3,
        ..Default::default()
    };
    let pml = ElasticSwePMLBoundary::new(&grid, config);
    let mask = pml.get_mask();

    assert_eq!(mask[[10, 10, 10]], 0.0);
    assert_eq!(mask[[0, 10, 10]], 1.0);
}
