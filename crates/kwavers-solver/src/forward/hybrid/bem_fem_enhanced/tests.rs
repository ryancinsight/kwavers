use super::config::EnhancedBemFemConfig;
use super::solver::EnhancedBemFemSolver;
use super::types::{BemFemValidationResult, InterfaceQuality};

#[test]
fn test_enhanced_config_default() {
    let config = EnhancedBemFemConfig::default();
    assert!(config.burton_miller_config.as_ref().unwrap().wavenumber > 0.0);
    assert!(config.adaptive_refinement);
}

#[test]
fn test_enhanced_config_sphere_validation() {
    let config = EnhancedBemFemConfig::for_sphere_validation(0.1, 1000.0);
    assert!(config.burton_miller_config.as_ref().unwrap().frequency > 0.0);
    assert!(config.adaptive_refinement);
    assert_eq!(
        config.validation_frequencies.as_ref().map(|f| f.len() ),
        Some(1)
    );
}

#[test]
fn test_enhanced_config_builder() {
    let config = EnhancedBemFemConfig::default()
        .with_adaptive_refinement(false)
        .with_target_error(1e-6);

    assert!(!config.adaptive_refinement);
    assert!((config.target_interface_error - 1e-6).abs() < 1e-12);
}

#[test]
fn test_solver_creation() {
    let config = EnhancedBemFemConfig::default();
    let solver = EnhancedBemFemSolver::new(config);
    assert!(solver.interface_quality.is_none());
    assert_eq!((solver.refinement_history.len()), 0);
}

#[test]
fn test_validation_result_passed() {
    let quality = InterfaceQuality {
        num_elements: 100,
        avg_element_size: 0.01,
        estimated_error: 1e-6,
        condition_number: Some(1000.0),
        max_local_error: 1e-5,
        spurious_resonance_detected: false,
    };

    let result = BemFemValidationResult {
        frequency: 1000.0,
        spurious_resonance_detected: false,
        burton_miller_used: true,
        interface_error: 1e-6,
        refinement_levels: 2,
        validation_time: 0.5,
        interface_quality: quality,
    };

    assert!(result.passed(1e-5));
    assert!(!result.passed(1e-7));
}

#[test]
fn test_validation_result_summary() {
    let quality = InterfaceQuality {
        num_elements: 100,
        avg_element_size: 0.01,
        estimated_error: 1e-6,
        condition_number: Some(1000.0),
        max_local_error: 1e-5,
        spurious_resonance_detected: false,
    };

    let result = BemFemValidationResult {
        frequency: 1000.0,
        spurious_resonance_detected: false,
        burton_miller_used: true,
        interface_error: 1e-6,
        refinement_levels: 2,
        validation_time: 0.5,
        interface_quality: quality,
    };

    let summary = result.summary();
    assert!(summary.contains("1000.0 Hz"));
    assert!(summary.contains("true"));
    assert!(summary.contains("1.00e-6"));
}

#[test]
fn test_interface_quality_structure() {
    let quality = InterfaceQuality {
        num_elements: 250,
        avg_element_size: 0.005,
        estimated_error: 5e-7,
        condition_number: Some(2000.0),
        max_local_error: 1e-6,
        spurious_resonance_detected: false,
    };

    assert_eq!(quality.num_elements, 250);
    assert!((quality.avg_element_size - 0.005).abs() < 1e-12);
    assert!(!quality.spurious_resonance_detected);
}

#[test]
fn test_burton_miller_validation_computes_interface_quality() {
    let mut config = EnhancedBemFemConfig::for_sphere_validation(0.1, 1000.0);
    config.adaptive_refinement = false;
    config.target_interface_error = 1e-3;
    let mut solver = EnhancedBemFemSolver::new(config);

    let result = solver.validate(1000.0).unwrap();

    assert!(!result.spurious_resonance_detected);
    assert!(result.burton_miller_used);
    assert!(result.interface_error > 0.0);
    assert!(result.interface_error < 1e-6);
    assert!(result.interface_quality.num_elements > 0);
    assert!(result.interface_quality.avg_element_size > 0.0);
    assert!(result.interface_quality.condition_number.unwrap() > 1.0);
}

#[test]
fn test_standard_bem_detects_configured_characteristic_frequency() {
    let config = EnhancedBemFemConfig::for_sphere_validation(0.1, 1000.0)
        .without_burton_miller()
        .with_adaptive_refinement(false)
        .with_target_error(1e-3);
    let mut solver = EnhancedBemFemSolver::new(config);

    let result = solver.validate(1000.0).unwrap();

    assert!(result.spurious_resonance_detected);
    assert!(!result.burton_miller_used);
    assert!(!result.passed(1e-3));
}

#[test]
fn test_adaptive_refinement_reduces_interface_error() {
    let mut config = EnhancedBemFemConfig::for_sphere_validation(0.1, 1000.0);
    config.min_element_size = config.max_element_size / 16.0;
    config.target_interface_error = 1e-9;
    config.max_refinement_level = 3;
    let mut solver = EnhancedBemFemSolver::new(config);

    let result = solver.validate(1000.0).unwrap();
    let history = solver.refinement_history();

    assert_eq!(result.refinement_levels, 3);
    assert_eq!((history.len()), 3);
    assert!(history[0].estimated_error > history[1].estimated_error);
    assert!(history[1].estimated_error > history[2].estimated_error);
    assert_eq!(
        result.interface_quality.estimated_error,
        history.last().unwrap().estimated_error
    );
}

#[test]
fn test_validation_rejects_invalid_frequency_and_mesh_bounds() {
    let mut solver = EnhancedBemFemSolver::new(EnhancedBemFemConfig::default());
    let bad_frequency = solver.validate(0.0).unwrap_err();
    assert!(format!("{bad_frequency}").contains("finite and positive"));

    let mut bad_config = EnhancedBemFemConfig::default();
    bad_config.min_element_size = 0.2;
    bad_config.max_element_size = 0.1;
    let mut solver = EnhancedBemFemSolver::new(bad_config);
    let bad_mesh = solver.validate(1000.0).unwrap_err();
    assert!(format!("{bad_mesh}").contains("element bounds"));
}
