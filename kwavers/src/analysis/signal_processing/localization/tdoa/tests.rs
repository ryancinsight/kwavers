use super::processor::TDOAProcessor;
use super::types::TDOAConfig;
use crate::analysis::signal_processing::localization::LocalizationProcessor;

#[test]
fn test_tdoa_processor_creation() {
    let config = TDOAConfig::default();
    let _processor = TDOAProcessor::new(&config).unwrap();
}

#[test]
fn test_tdoa_insufficient_sensors() {
    let processor = TDOAProcessor::new(&TDOAConfig::default()).unwrap();
    let result = processor.localize(&[0.0], &[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]);
    assert!(result.is_err());
}

#[test]
fn test_tdoa_config_builder() {
    let config = TDOAConfig::default()
        .with_refinement_iterations(10)
        .with_convergence_tolerance(1e-8);

    assert_eq!(config.refinement_iterations, 10);
    assert_eq!(config.convergence_tolerance, 1e-8);
}

#[test]
fn test_tdoa_localization() {
    let config = TDOAConfig::default();
    let processor = TDOAProcessor::new(&config).unwrap();

    let sensor_positions = vec![
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ];

    let time_delays = vec![0.0, 0.0001, 0.00015, 0.0002];
    let loc = processor.localize(&time_delays, &sensor_positions).unwrap();
    assert!(loc.position.iter().all(|x| x.is_finite()));
}
