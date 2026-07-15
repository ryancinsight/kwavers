//! Value-semantic regression tests for linear elastography inversion methods.

use super::direct::direct_inversion;
use super::directional::directional_phase_gradient_inversion;
use super::lfe::local_frequency_estimation_inversion;
use super::phase_gradient::phase_gradient_inversion;
use super::time_of_flight::time_of_flight_inversion;
use super::volumetric::volumetric_time_of_flight_inversion;
use super::ShearWaveInversion;
use crate::inverse::elastography::config::ShearWaveInversionConfig;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::InversionMethod;
use kwavers_physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

#[test]
fn test_time_of_flight_inversion() {
    // Zero displacement → default speed 3.0 everywhere (boundary fill). Map dim must be (20,20,20).
    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let displacement = DisplacementField::zeros(20, 20, 20);

    let map = time_of_flight_inversion(&displacement, &grid, 1000.0, 100.0).unwrap();
    assert_eq!(map.shear_wave_speed.shape(), [20, 20, 20]);
    let center = map.shear_wave_speed[[10, 10, 10]];
    assert!(
        (0.5..=10.0).contains(&center),
        "center shear_wave_speed = {center} must be in [0.5, 10.0]"
    );
}

#[test]
fn test_phase_gradient_inversion() {
    // Zero displacement → default speed 3.0 everywhere. Map dim must be (20,20,20).
    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let displacement = DisplacementField::zeros(20, 20, 20);

    let map = phase_gradient_inversion(&displacement, &grid, 1000.0, 100.0).unwrap();
    assert_eq!(map.shear_wave_speed.shape(), [20, 20, 20]);
    let center = map.shear_wave_speed[[10, 10, 10]];
    assert!(
        (0.5..=10.0).contains(&center),
        "center shear_wave_speed = {center} must be in [0.5, 10.0]"
    );
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
    let k_wave = TWO_PI * frequency / 3.0;

    for i in 0..nx {
        let x = i as f64 * dx;
        let val = (k_wave * x).cos();

        for j in 0..ny {
            for k in 0..nz {
                displacement.uz[[i, j, k]] = val;
            }
        }
    }

    let elasticity_map = direct_inversion(&displacement, &grid, 1000.0, frequency).unwrap();
    let center_val = elasticity_map.shear_wave_speed[[nx / 2, ny / 2, nz / 2]];

    assert!(
        (center_val - 3.0).abs() < 1.0,
        "Expected speed approx 3.0, got {}",
        center_val
    );
}

#[test]
fn test_local_frequency_estimation_recovers_known_speed() {
    // Plane shear wave along X with a known speed cs = 1.0 m/s at 100 Hz.
    // λ = cs/f = 10 mm = 10 cells (dx = 1 mm) → several wavelengths across the
    // domain so the LFE averaging window resolves ⟨|∇u|²⟩/⟨u²⟩ → |k|².
    let dx = 0.001;
    let (nx, ny, nz) = (64, 8, 8);
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    let frequency = 100.0;
    let cs_true = 1.0;
    let k_wave = TWO_PI * frequency / cs_true;

    let mut displacement = DisplacementField::zeros(nx, ny, nz);
    for i in 0..nx {
        let val = (k_wave * (i as f64 * dx)).cos();
        for j in 0..ny {
            for k in 0..nz {
                displacement.uz[[i, j, k]] = val;
            }
        }
    }

    let map =
        local_frequency_estimation_inversion(&displacement, &grid, 1000.0, frequency).unwrap();
    assert_eq!(map.shear_wave_speed.shape(), [nx, ny, nz]);

    let center = map.shear_wave_speed[[nx / 2, ny / 2, nz / 2]];
    // Value-semantic: recovered speed tracks the true 1.0 m/s (not the default
    // fill nor the clamp rails at 0.5 / 20.0 m/s).
    assert!(
        (center - cs_true).abs() < 0.4,
        "LFE center speed = {center} should recover cs_true = {cs_true} (±0.4)"
    );

    // Shear modulus μ = ρ cs² must be consistent with the recovered speed.
    let mu = map.shear_modulus[[nx / 2, ny / 2, nz / 2]];
    let mu_expected = 1000.0 * center * center;
    assert!(
        (mu - mu_expected).abs() / mu_expected < 1e-6,
        "shear modulus {mu} must equal ρ·cs² = {mu_expected}"
    );
}

/// Regression: the LFE inversion works on a 2-D plane-strain field (`nz = 1`),
/// the conventional clinical SWE imaging plane. Before the singleton-axis guard
/// the 3-D interior loop was empty for `nz = 1`, collapsing the estimate to the
/// 20 m/s speed clamp (silent garbage). Same plane shear wave as the 3-D test.
#[test]
fn test_local_frequency_estimation_2d_plane() {
    let dx = 0.001;
    let (nx, ny, nz) = (64, 8, 1);
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();

    let frequency = 100.0;
    let cs_true = 1.0;
    let k_wave = TWO_PI * frequency / cs_true;

    let mut displacement = DisplacementField::zeros(nx, ny, nz);
    for i in 0..nx {
        let val = (k_wave * (i as f64 * dx)).cos();
        for j in 0..ny {
            displacement.uz[[i, j, 0]] = val;
        }
    }

    let map =
        local_frequency_estimation_inversion(&displacement, &grid, 1000.0, frequency).unwrap();
    let center = map.shear_wave_speed[[nx / 2, ny / 2, 0]];
    assert!(
        (center - cs_true).abs() < 0.4,
        "2-D LFE center speed = {center} should recover cs_true = {cs_true} (±0.4), \
         not the 20 m/s clamp"
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
        InversionMethod::LocalFrequencyEstimation,
    ] {
        let config = ShearWaveInversionConfig::new(method);
        let inversion = ShearWaveInversion::new(config);
        let map = inversion
            .reconstruct(&displacement, &grid)
            .unwrap_or_else(|e| panic!("Inversion method {method:?} should succeed; got: {e:?}"));
        assert_eq!(
            map.shear_wave_speed.shape(),
            [20, 20, 20],
            "method {method:?}: shear_wave_speed must be 20×20×20"
        );
    }
}

#[test]
fn test_volumetric_tof_with_single_peak() {
    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let mut displacement = DisplacementField::zeros(20, 20, 20);
    displacement.uz[[10, 10, 10]] = 5.0; // Single push location

    let map = volumetric_time_of_flight_inversion(&displacement, &grid, 1000.0, 100.0).unwrap();
    assert_eq!(
        map.shear_wave_speed.shape(),
        [20, 20, 20],
        "output must span full 20×20×20 grid"
    );
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

    let map = directional_phase_gradient_inversion(&displacement, &grid, 1000.0, 100.0).unwrap();
    let center = map.shear_wave_speed[[10, 10, 10]];
    assert!(
        (0.5..=10.0).contains(&center),
        "directional phase gradient center speed = {center} must be in [0.5, 10.0]"
    );
}

#[test]
fn test_shear_wave_inversion_processor() {
    let config = ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight);
    let processor = ShearWaveInversion::new(config);

    assert_eq!(processor.method(), InversionMethod::TimeOfFlight);
    assert_eq!(processor.config().density, DENSITY_WATER_NOMINAL);
}
