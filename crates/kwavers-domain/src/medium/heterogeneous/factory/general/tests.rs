use super::*;
use kwavers_core::constants::fundamental::{
    DENSITY_WATER_NOMINAL,
    SOUND_SPEED_TISSUE,
    SOUND_SPEED_WATER_SIM,
};
use kwavers_core::constants::tissue_acoustics::DENSITY_BLOOD;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use crate::grid::Grid;
use ndarray::Array3;

#[test]
fn test_from_arrays_basic() {
    let c = Array3::from_elem((10, 10, 10), SOUND_SPEED_WATER_SIM);
    let rho = Array3::from_elem((10, 10, 10), DENSITY_WATER_NOMINAL);

    let medium = HeterogeneousFactory::from_arrays(c, rho, None, None, None, MHZ_TO_HZ).unwrap();

    assert_eq!(medium.sound_speed[[0, 0, 0]], SOUND_SPEED_WATER_SIM);
    assert_eq!(medium.density[[0, 0, 0]], DENSITY_WATER_NOMINAL);
    assert_eq!(medium.reference_frequency, MHZ_TO_HZ);
    assert_eq!(medium.alpha_power[[0, 0, 0]], 1.0);
}

#[test]
fn test_from_arrays_with_optional() {
    let c = Array3::from_elem((10, 10, 10), SOUND_SPEED_WATER_SIM);
    let rho = Array3::from_elem((10, 10, 10), DENSITY_WATER_NOMINAL);
    let alpha = Array3::from_elem((10, 10, 10), 0.5);
    let yexp = Array3::from_elem((10, 10, 10), 1.5_f64);
    let ba = Array3::from_elem((10, 10, 10), 6.0);

    let medium =
        HeterogeneousFactory::from_arrays(c, rho, Some(alpha), Some(yexp), Some(ba), MHZ_TO_HZ).unwrap();

    assert_eq!(medium.absorption[[0, 0, 0]], 0.5);
    assert_eq!(medium.alpha_power[[0, 0, 0]], 1.5);
    assert_eq!(medium.nonlinearity[[0, 0, 0]], 6.0);
}

#[test]
fn test_from_functions() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();

    let medium = HeterogeneousFactory::from_functions(
        &grid,
        |_x, _y, _z| SOUND_SPEED_WATER_SIM,
        |_x, _y, _z| DENSITY_WATER_NOMINAL,
        Some(Box::new(|_x, _y, z| if z > 0.005 { 0.5 } else { 0.0 })),
        Some(Box::new(|_x, _y, _z| 1.5)),
        None,
        MHZ_TO_HZ,
    );

    assert_eq!(medium.sound_speed[[0, 0, 0]], SOUND_SPEED_WATER_SIM);
    assert_eq!(medium.density[[0, 0, 0]], DENSITY_WATER_NOMINAL);
    assert_eq!(medium.alpha_power[[0, 0, 0]], 1.5);
}

#[test]
fn test_from_layers() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();

    let layers = vec![
        (0.0, 0.005, SOUND_SPEED_WATER_SIM, DENSITY_WATER_NOMINAL, 0.0, 0.0),
        (0.005, 0.010, SOUND_SPEED_TISSUE, DENSITY_BLOOD, 0.5, 6.0),
    ];

    let medium = HeterogeneousFactory::from_layers(&grid, &layers, MHZ_TO_HZ);

    assert_eq!(medium.sound_speed[[0, 0, 0]], SOUND_SPEED_WATER_SIM);
    assert_eq!(medium.density[[0, 0, 0]], DENSITY_WATER_NOMINAL);
    assert_eq!(medium.sound_speed[[0, 0, 9]], SOUND_SPEED_TISSUE);
    assert_eq!(medium.density[[0, 0, 9]], DENSITY_BLOOD);
}

#[test]
fn test_from_elastic_arrays_lame_inversion() {
    // Analytical verification: bone-like solid
    // c_p=3000 m/s, c_s=1500 m/s, ρ=1900 kg/m³
    // μ = 1900 * 1500² = 4.275e9 Pa
    // λ = 1900 * (3000² - 2*1500²) = 1900 * (9e6 - 4.5e6) = 8.55e9 Pa
    let n = 4usize;
    let cp = Array3::from_elem((n, n, n), 3000.0_f64);
    let cs = Array3::from_elem((n, n, n), SOUND_SPEED_WATER_SIM);
    let rho = Array3::from_elem((n, n, n), 1900.0_f64);

    let med =
        HeterogeneousFactory::from_elastic_arrays(cp.view(), cs.view(), rho.view(), MHZ_TO_HZ).unwrap();

    let mu_expected = 1900.0 * SOUND_SPEED_WATER_SIM.powi(2);
    let lambda_expected = 1900.0 * (3000.0_f64.powi(2) - 2.0 * SOUND_SPEED_WATER_SIM.powi(2));
    assert!(
        (med.lame_mu[[0, 0, 0]] - mu_expected).abs() < 1.0,
        "μ={} expected={}",
        med.lame_mu[[0, 0, 0]],
        mu_expected
    );
    assert!(
        (med.lame_lambda[[0, 0, 0]] - lambda_expected).abs() < 1.0,
        "λ={} expected={}",
        med.lame_lambda[[0, 0, 0]],
        lambda_expected
    );
    assert_eq!(med.sound_speed[[0, 0, 0]], 3000.0);
    assert_eq!(med.shear_sound_speed[[0, 0, 0]], SOUND_SPEED_WATER_SIM);
}

#[test]
fn test_from_elastic_arrays_fluid_voxel() {
    // c_s = 0 → fluid: μ = 0, λ = ρ·c_p²
    let n = 2usize;
    let cp = Array3::from_elem((n, n, n), SOUND_SPEED_WATER_SIM);
    let cs = Array3::zeros((n, n, n));
    let rho = Array3::from_elem((n, n, n), DENSITY_WATER_NOMINAL);

    let med =
        HeterogeneousFactory::from_elastic_arrays(cp.view(), cs.view(), rho.view(), MHZ_TO_HZ).unwrap();

    assert_eq!(med.lame_mu[[0, 0, 0]], 0.0);
    assert!((med.lame_lambda[[0, 0, 0]] - DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM.powi(2)).abs() < 1.0);
}

#[test]
fn test_from_elastic_arrays_stability_violation() {
    // c_s > c_p/sqrt(2) → λ < 0, invalid
    let n = 2usize;
    let cp = Array3::from_elem((n, n, n), 1000.0_f64);
    let cs = Array3::from_elem((n, n, n), 900.0_f64);
    let rho = Array3::from_elem((n, n, n), DENSITY_WATER_NOMINAL);

    let result = HeterogeneousFactory::from_elastic_arrays(cp.view(), cs.view(), rho.view(), MHZ_TO_HZ);
    assert!(result.is_err(), "Expected stability error");
}
