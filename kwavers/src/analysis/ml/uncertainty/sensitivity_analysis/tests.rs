//! Tests for sensitivity analysis.

use super::analyzer::SensitivityAnalyzer;
use super::config::SensitivityConfig;
use ndarray::Array1;

#[test]
fn test_sensitivity_analyzer_creation() {
    let config = SensitivityConfig {
        num_samples: 100,
        confidence_level: 0.95,
    };
    let analyzer = SensitivityAnalyzer::new(config);
    assert!(analyzer.is_ok());
}

#[test]
fn test_parameter_sample_generation() {
    let config = SensitivityConfig::default();
    let analyzer = SensitivityAnalyzer::new(config).unwrap();

    let parameter_ranges = vec![(0.0, 1.0), (1.0, 2.0)];
    let samples = analyzer.generate_parameter_samples(&parameter_ranges, 10);

    assert!(samples.is_ok());
    let samples_vec = samples.unwrap();
    assert_eq!(samples_vec.len(), 10);

    for sample in &samples_vec {
        assert_eq!(sample.len(), 2);
        assert!(sample[0] >= 0.0 && sample[0] <= 1.0);
        assert!(sample[1] >= 1.0 && sample[1] <= 2.0);
    }
}

#[test]
fn test_sensitivity_analysis() {
    let config = SensitivityConfig::default();
    let analyzer = SensitivityAnalyzer::new(config).unwrap();

    let model_fn = |params: &Array1<f64>| Array1::from_vec(vec![2.0 * params[0] + 0.5 * params[1]]);

    let parameter_ranges = vec![(0.0, 1.0), (0.0, 1.0)];
    let analysis = analyzer.analyze(model_fn, &parameter_ranges, 50);

    assert!(analysis.is_ok());
    let indices = analysis.unwrap();

    let sens1 = indices.total.get("param_0").unwrap_or(&0.0);
    let sens2 = indices.total.get("param_1").unwrap_or(&0.0);

    assert!(sens1 >= sens2, "First parameter should be more sensitive");
}

#[test]
fn test_morris_screening() {
    let config = SensitivityConfig::default();
    let analyzer = SensitivityAnalyzer::new(config).unwrap();

    let model_fn = |params: &Array1<f64>| Array1::from_vec(vec![params[0] + params[1]]);

    let parameter_ranges = vec![(0.0, 1.0), (0.0, 1.0)];
    let morris = analyzer.morris_screening(model_fn, &parameter_ranges, 5, 10);

    assert!(morris.is_ok());
    let results = morris.unwrap();

    assert!(!results.mu.is_empty());
    assert!(!results.sigma.is_empty());
}
