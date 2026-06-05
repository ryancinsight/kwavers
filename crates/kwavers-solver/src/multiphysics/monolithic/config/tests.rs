use super::*;
use kwavers_core::constants::SOUND_SPEED_TISSUE;
use kwavers_core::error::{KwaversError, ValidationError};

#[test]
fn test_newton_krylov_config_default() {
    let config = NewtonKrylovConfig::default();

    assert_eq!(config.max_newton_iterations, 20);
    assert!(config.newton_tolerance < 1e-5);
    assert!(config.line_search_parameter > 0.0 && config.line_search_parameter <= 1.0);
    config.validate().unwrap();
}

#[test]
fn test_newton_krylov_validation_rejects_invalid_alpha() {
    let config = NewtonKrylovConfig {
        line_search_parameter: 1.25,
        ..NewtonKrylovConfig::default()
    };

    let error = config.validate().unwrap_err();

    match error {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter,
            value,
            reason,
        }) => {
            assert_eq!(parameter, "NewtonKrylovConfig::line_search_parameter");
            assert_eq!(value, 1.25);
            assert_eq!(reason, "must be finite and in (0, 1]");
        }
        other => panic!("expected line-search alpha validation error, got {other:?}"),
    }
}

#[test]
fn test_physics_coefficients_default() {
    let c = PhysicsCoefficients::default();

    assert!((c.sound_speed - SOUND_SPEED_TISSUE).abs() < 1e-10);
    assert!(c.thermal_diffusivity() > 0.0);
    assert!(c.optical_diffusion() > 0.0);
    c.validate().unwrap();
}

#[test]
fn test_physics_coefficients_reject_zero_density() {
    let coefficients = PhysicsCoefficients {
        density: 0.0,
        ..PhysicsCoefficients::default()
    };

    let error = coefficients.validate().unwrap_err();

    match error {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter,
            value,
            reason,
        }) => {
            assert_eq!(parameter, "PhysicsCoefficients::density");
            assert_eq!(value, 0.0);
            assert_eq!(reason, "must be finite and positive");
        }
        other => panic!("expected density validation error, got {other:?}"),
    }
}

#[test]
fn test_physics_coefficients_reject_zero_optical_transport() {
    let coefficients = PhysicsCoefficients {
        optical_absorption: 0.0,
        reduced_scattering: 0.0,
        ..PhysicsCoefficients::default()
    };

    let error = coefficients.validate().unwrap_err();

    match error {
        KwaversError::Validation(ValidationError::InvalidParameter { parameter, reason }) => {
            assert_eq!(parameter, "PhysicsCoefficients::optical_transport");
            assert_eq!(
                reason,
                "optical_absorption + reduced_scattering must be positive"
            );
        }
        other => panic!("expected optical transport validation error, got {other:?}"),
    }
}

/// Photoacoustic source term scales linearly with the Gruneisen parameter.
#[test]
fn test_photoacoustic_default_gruneisen_not_one() {
    let c = PhysicsCoefficients::default();
    assert!(
        (c.gruneisen - 1.0).abs() > 0.01,
        "Default Gruneisen parameter ({}) must not be 1.0; water at 37 C ~= 0.12",
        c.gruneisen
    );
    assert!(
        c.gruneisen > 0.0,
        "Gruneisen parameter must be positive, got {}",
        c.gruneisen
    );
}
