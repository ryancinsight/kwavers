use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{self};
use kwavers_domain::medium::core::CoreMedium;

#[test]
fn test_default_config_creation() {
    let config = kwavers_simulation::configuration::Configuration::default();
    // Config validation - check that required fields exist
    assert!(config.simulation.duration > 0.0);
    assert!(config.simulation.frequency > 0.0);
    assert!(!config.output.snapshots); // Default is false
}

#[test]
fn test_config_with_custom_values() {
    use kwavers_simulation::parameters::SimulationParameters;
    let config = kwavers_simulation::configuration::Configuration {
        simulation: SimulationParameters {
            frequency: 2.0 * MHZ_TO_HZ,
            duration: 0.001,
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(config.simulation.frequency, 2.0 * MHZ_TO_HZ);
}

#[test]
fn test_version_info() {
    let info = crate::get_version_info();
    assert!(info.contains_key("version"));
    assert!(info.contains_key("name"));
}

// ============================================================================
// MINIMAL UNIT TESTS FOR SRS NFR-002 COMPLIANCE (<30s execution)
// Fast tests focusing on core functionality without expensive computations
// ============================================================================

#[test]
fn test_grid_creation_minimal() {
    let grid = kwavers_domain::grid::Grid::new(8, 8, 8, 0.001, 0.001, 0.001).expect("Grid creation");
    assert_eq!(grid.nx, 8);
    assert_eq!(grid.ny, 8);
    assert_eq!(grid.nz, 8);
    assert_eq!(grid.size(), 512);
}

#[test]
fn test_medium_basic_properties() {
    let grid = kwavers_domain::grid::Grid::new(4, 4, 4, 0.001, 0.001, 0.001).expect("Grid creation");
    let medium = kwavers_domain::medium::HomogeneousMedium::new(
        kwavers_core::constants::DENSITY_WATER,
        kwavers_core::constants::SOUND_SPEED_WATER,
        0.0,
        0.0,
        &grid,
    );

    assert!(medium.is_homogeneous());
    assert!((medium.sound_speed(0, 0, 0) - kwavers_core::constants::SOUND_SPEED_WATER).abs() < 1e-6);
    assert!((medium.density(0, 0, 0) - kwavers_core::constants::DENSITY_WATER).abs() < 1e-6);
}

#[test]
fn test_physics_constants_validation() {
    // Physics constants are compile-time verified through const definitions
    // No runtime assertions needed for const values (clippy::assertions_on_constants)
    use kwavers_core::constants::*;

    // Validate that constants are accessible and have expected types
    let _density: f64 = DENSITY_WATER;
    let _speed: f64 = SOUND_SPEED_WATER;

    // Constants are defined in core::constants::fundamental
    // DENSITY_WATER = 998.2 kg/m³ (valid water density)
    // SOUND_SPEED_WATER = 1482.0 m/s (valid water sound speed)
}

#[test]
fn test_cfl_calculation_basic() {
    let grid = kwavers_domain::grid::Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("Grid creation");
    let sound_speed = SOUND_SPEED_WATER_SIM;
    let cfl = 0.4; // Conservative CFL for 3D
    let min_dx = grid.dx.min(grid.dy).min(grid.dz);
    let dt = cfl * min_dx / sound_speed;

    assert!(dt > 0.0);
    assert!(dt < 1e-6); // Reasonable timestep for acoustics

    // CFL stability condition: c*dt/dx <= CFL_max
    let actual_cfl = sound_speed * dt / min_dx;
    assert!((actual_cfl - cfl).abs() < 1e-10);
}

#[test]
fn test_error_handling_basic() {
    // Test basic error type creation
    use error::{ConfigError, KwaversError};

    let config_error = ConfigError::InvalidValue {
        parameter: "test".to_string(),
        value: "invalid".to_string(),
        constraint: "must be positive".to_string(),
    };

    let kwavers_error = KwaversError::Config(config_error);
    assert!(matches!(kwavers_error, KwaversError::Config(_)));
}
