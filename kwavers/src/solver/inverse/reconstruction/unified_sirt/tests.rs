use super::config::{SirtAlgorithm, SirtConfig, SirtResult};
use super::reconstructor::SirtReconstructor;
use ndarray::Array3;

#[test]
fn test_sirt_config_default() {
    let cfg = SirtConfig::default();
    assert_eq!(cfg.algorithm, SirtAlgorithm::Sirt);
    assert_eq!(cfg.max_iterations, 100);
    assert!(cfg.relaxation_factor > 0.0 && cfg.relaxation_factor <= 1.0);
}

#[test]
fn test_sirt_config_builder() {
    let cfg = SirtConfig::default()
        .with_sirt()
        .with_iterations(50)
        .with_relaxation(0.3);

    assert_eq!(cfg.algorithm, SirtAlgorithm::Sirt);
    assert_eq!(cfg.max_iterations, 50);
    assert!((cfg.relaxation_factor - 0.3).abs() < 1e-10);
}

#[test]
fn test_osem_config() {
    let cfg = SirtConfig::default().with_osem(4);
    assert_eq!(cfg.algorithm, SirtAlgorithm::Osem { num_subsets: 4 });
}

#[test]
fn test_art_config() {
    let cfg = SirtConfig::default().with_art();
    assert_eq!(cfg.algorithm, SirtAlgorithm::Art);
}

#[test]
fn test_algorithm_display() {
    assert_eq!(format!("{}", SirtAlgorithm::Sirt), "SIRT");
    assert_eq!(format!("{}", SirtAlgorithm::Art), "ART");
    assert_eq!(
        format!("{}", SirtAlgorithm::Osem { num_subsets: 4 }),
        "OSEM (subsets=4)"
    );
}

#[test]
fn test_reconstructor_creation() {
    let cfg = SirtConfig::default();
    let _reconstructor = SirtReconstructor::new(cfg);
}

#[test]
fn test_relaxation_factor_clamping() {
    let cfg1 = SirtConfig::default().with_relaxation(2.0);
    assert!(cfg1.relaxation_factor <= 1.0);

    let cfg2 = SirtConfig::default().with_relaxation(-0.5);
    assert!(cfg2.relaxation_factor >= 0.001);
}

#[test]
fn test_sirt_result_structure() {
    let result = SirtResult {
        image: Array3::zeros((10, 10, 10)),
        iterations: 42,
        final_residual: 0.001,
        residual_history: vec![1.0, 0.9, 0.8, 0.7],
        converged: true,
        computation_time: 1.23,
    };

    assert_eq!(result.iterations, 42);
    assert!(result.converged);
    assert!(result.computation_time > 0.0);
}
