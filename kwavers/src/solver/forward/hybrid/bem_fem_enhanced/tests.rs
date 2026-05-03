use super::config::EnhancedBemFemConfig;
use super::solver::EnhancedBemFemSolver;
use super::types::{InterfaceQuality, ValidationResult};

#[test]
fn test_enhanced_config_default() {
    let config = EnhancedBemFemConfig::default();
    assert!(config.burton_miller_config.is_some());
    assert!(config.adaptive_refinement);
}

#[test]
fn test_enhanced_config_sphere_validation() {
    let config = EnhancedBemFemConfig::for_sphere_validation(0.1, 1000.0);
    assert!(config.burton_miller_config.is_some());
    assert!(config.adaptive_refinement);
    assert_eq!(
        config.validation_frequencies.as_ref().map(|f| f.len()),
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
    assert_eq!(solver.refinement_history.len(), 0);
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

    let result = ValidationResult {
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

    let result = ValidationResult {
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
