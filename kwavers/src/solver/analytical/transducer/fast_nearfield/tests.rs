use super::core::FastNearfieldSolver;
use super::types::FNMConfig;
use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::domain::source::transducers::rectangular::RectangularTransducer;
use crate::math::fft::Complex64;
use ndarray::Array2;

#[test]
fn test_fnm_solver_creation() {
    let config = FNMConfig::default();
    let solver = FastNearfieldSolver::new(config);
    assert!(solver.is_ok());
}

#[test]
fn test_transducer_setup() {
    let config = FNMConfig::default();
    let mut solver = FastNearfieldSolver::new(config).unwrap();

    let transducer = RectangularTransducer {
        width: 10e-3,
        height: 10e-3,
        frequency: 1e6,
        elements: (32, 32),
    };

    solver.set_transducer(transducer);

    let (elem_width, elem_height) = solver.transducer.as_ref().unwrap().element_size();
    assert!((elem_width - 10e-3 / 32.0).abs() < 1e-9);
    assert!((elem_height - 10e-3 / 32.0).abs() < 1e-9);
}

#[test]
fn test_precompute_factors() {
    let config = FNMConfig::default();
    let mut solver = FastNearfieldSolver::new(config).unwrap();

    let transducer = RectangularTransducer {
        width: 5e-3,
        height: 5e-3,
        frequency: 2e6,
        elements: (16, 16),
    };

    solver.set_transducer(transducer);
    solver.set_medium(SOUND_SPEED_WATER_SIM, DENSITY_WATER_NOMINAL);

    let result = solver.precompute_factors(25e-3); // 25 mm
    assert!(result.is_ok());

    // Check that factors were cached
    assert_eq!(solver.cached_z_distances().len(), 1);
    assert!((solver.cached_z_distances()[0] - 25e-3).abs() < 1e-9);
}

#[test]
fn test_field_computation() {
    let config = FNMConfig {
        angular_spectrum_size: (64, 64), // Smaller for testing
        ..Default::default()
    };
    let mut solver = FastNearfieldSolver::new(config).unwrap();

    let transducer = RectangularTransducer {
        width: 5e-3,
        height: 5e-3,
        frequency: 2e6,
        elements: (16, 16),
    };

    solver.set_transducer(transducer);
    solver.precompute_factors(25e-3).unwrap();

    // Uniform velocity distribution
    let velocity = Array2::<Complex64>::from_elem((16, 16), Complex64::new(1.0, 0.0));

    let pressure = solver.compute_field(&velocity, 25e-3);
    assert!(pressure.is_ok());

    let pressure_field = pressure.unwrap();
    assert_eq!(pressure_field.dim(), (16, 16));

    // Check that result is not zero (basic sanity check)
    let sum: Complex64 = pressure_field.iter().sum();
    assert!(sum.norm() > 0.0);
}

#[test]
fn test_memory_usage() {
    let config = FNMConfig::default();
    let mut solver = FastNearfieldSolver::new(config.clone()).unwrap();

    let transducer = RectangularTransducer {
        width: 10e-3,
        height: 10e-3,
        frequency: 1e6,
        elements: (32, 32),
    };

    solver.set_transducer(transducer);
    solver.precompute_factors(50e-3).unwrap();

    let usage = solver.memory_usage();
    assert!(usage > 0);

    // Clear cache and check memory drops
    solver.clear_cache();
    let usage_after_clear = solver.memory_usage();

    // Usage should not be zero anymore due to precomputed vectors
    // Default config: 512x512 -> kx=512, ky=512 -> 1024 * 8 bytes = 8192 bytes
    let (n_kx, n_ky) = config.angular_spectrum_size;
    let expected_base_usage = (n_kx + n_ky) * std::mem::size_of::<f64>();
    assert_eq!(usage_after_clear, expected_base_usage);
}

/// Solver defaults must equal the canonical water constants so that any code
/// reading `solver.c0` / `solver.rho0` via `set_medium` round-trips correctly.
#[test]
fn test_fast_nearfield_defaults_match_water_constants() {
    let config = FNMConfig::default();
    let solver = FastNearfieldSolver::new(config).unwrap();
    assert!(
        (solver.c0 - SOUND_SPEED_WATER_SIM).abs() < f64::EPSILON,
        "Default c0 ({}) must equal SOUND_SPEED_WATER_SIM ({})",
        solver.c0,
        SOUND_SPEED_WATER_SIM
    );
    assert!(
        (solver.rho0 - DENSITY_WATER_NOMINAL).abs() < f64::EPSILON,
        "Default rho0 ({}) must equal DENSITY_WATER_NOMINAL ({})",
        solver.rho0,
        DENSITY_WATER_NOMINAL
    );
}
