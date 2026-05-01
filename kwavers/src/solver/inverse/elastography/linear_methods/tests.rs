//! Value-semantic regression tests for linear elastography inversion methods.

use std::f64::consts::PI;

use super::direct::direct_inversion;
use super::directional::directional_phase_gradient_inversion;
use super::phase_gradient::{compute_phase_gradient_speed, phase_gradient_inversion};
use super::time_of_flight::time_of_flight_inversion;
use super::volumetric::volumetric_time_of_flight_inversion;
use super::ShearWaveInversion;
use crate::domain::grid::Grid;
use crate::domain::imaging::ultrasound::elastography::InversionMethod;
use crate::physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;
use crate::solver::inverse::elastography::config::ShearWaveInversionConfig;

#[test]
fn test_time_of_flight_inversion() {
    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let displacement = DisplacementField::zeros(20, 20, 20);

    let result = time_of_flight_inversion(&displacement, &grid, 1000.0, 100.0);
    assert!(result.is_ok(), "TOF inversion should succeed");
}

#[test]
fn test_phase_gradient_inversion() {
    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let displacement = DisplacementField::zeros(20, 20, 20);

    let result = phase_gradient_inversion(&displacement, &grid, 1000.0, 100.0);
    assert!(result.is_ok(), "Phase gradient inversion should succeed");
}

#[test]
fn test_direct_inversion_synthetic() {
    let dx = 0.001;
    let nx = 30;
    let ny = 10;
    let nz = 10;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    let mut displacement = DisplacementField::zeros(nx, ny, nz);

    // Synthetic wave: plane wave along X with speed cs = 3.0 m/s at 100 Hz
    let frequency = 100.0;
    let k_wave = 2.0 * PI * frequency / 3.0;

    for i in 0..nx {
        let x = i as f64 * dx;
        let val = (k_wave * x).cos();

        for j in 0..ny {
            for k in 0..nz {
                displacement.uz[[i, j, k]] = val;
            }
        }
    }

    let result = direct_inversion(&displacement, &grid, 1000.0, frequency);
    assert!(result.is_ok());

    let elasticity_map = result.unwrap();
    let center_val = elasticity_map.shear_wave_speed[[nx / 2, ny / 2, nz / 2]];

    assert!(
        (center_val - 3.0).abs() < 1.0,
        "Expected speed approx 3.0, got {}",
        center_val
    );
}

#[test]
fn test_all_inversion_methods() {
    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let displacement = DisplacementField::zeros(20, 20, 20);

    for method in [
        InversionMethod::TimeOfFlight,
        InversionMethod::PhaseGradient,
        InversionMethod::DirectInversion,
        InversionMethod::VolumetricTimeOfFlight,
        InversionMethod::DirectionalPhaseGradient,
    ] {
        let config = ShearWaveInversionConfig::new(method);
        let inversion = ShearWaveInversion::new(config);
        let result = inversion.reconstruct(&displacement, &grid);
        assert!(
            result.is_ok(),
            "Inversion method {:?} should succeed",
            method
        );
    }
}

#[test]
fn test_volumetric_tof_with_single_peak() {
    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let mut displacement = DisplacementField::zeros(20, 20, 20);
    displacement.uz[[10, 10, 10]] = 5.0; // Single push location

    let result = volumetric_time_of_flight_inversion(&displacement, &grid, 1000.0, 100.0);
    assert!(result.is_ok(), "Volumetric TOF should handle single peak");
}

#[test]
fn test_directional_phase_gradient() {
    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let mut displacement = DisplacementField::zeros(20, 20, 20);

    // Create a gradient pattern
    for i in 0..20 {
        for j in 0..20 {
            for k in 0..20 {
                displacement.uz[[i, j, k]] = (i as f64 / 20.0) * 0.01;
            }
        }
    }

    let result = directional_phase_gradient_inversion(&displacement, &grid, 1000.0, 100.0);
    assert!(result.is_ok(), "Directional phase gradient should succeed");
}

#[test]
fn test_compute_phase_gradient_speed() {
    let profile = vec![0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0];
    let dx = 0.001;

    let speed = compute_phase_gradient_speed(&profile, dx, 100.0);
    assert!(speed.is_some(), "Should compute speed from valid profile");

    let cs = speed.unwrap();
    assert!((0.5..=10.0).contains(&cs), "Speed should be in valid range");
}

#[test]
fn test_compute_phase_gradient_speed_empty() {
    let profile = vec![0.0, 0.0];
    let dx = 0.001;

    let speed = compute_phase_gradient_speed(&profile, dx, 100.0);
    assert!(speed.is_none(), "Should return None for insufficient data");
}

#[test]
fn test_shear_wave_inversion_processor() {
    let config = ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight);
    let processor = ShearWaveInversion::new(config);

    assert_eq!(processor.method(), InversionMethod::TimeOfFlight);
    assert_eq!(processor.config().density, 1000.0);
}
