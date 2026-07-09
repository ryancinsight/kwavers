//! Tests for sensitivity analysis.

use super::analyzer::SensitivityAnalyzer;
use super::config::SensitivityConfig;
use leto::Array1;

#[test]
fn test_sensitivity_analyzer_creation() {
    let config = SensitivityConfig {
        num_samples: 100,
        confidence_level: 0.95,
    };
    let analyzer = SensitivityAnalyzer::new(config).unwrap();
    // Verify stored config fields.
    assert_eq!(analyzer.config.num_samples, 100);
    assert!(
        (analyzer.config.confidence_level - 0.95).abs() < 1e-12,
        "confidence_level must be 0.95"
    );
}

#[test]
fn test_parameter_sample_generation() {
    let analyzer = SensitivityAnalyzer::new(SensitivityConfig::default()).unwrap();
    let parameter_ranges = vec![(0.0, 1.0), (1.0, 2.0)];
    let samples = analyzer
        .generate_parameter_samples(&parameter_ranges, 10)
        .unwrap();

    assert_eq!(samples.len(), 10, "must produce exactly 10 samples");
    for sample in &samples {
        assert_eq!(sample.len(), 2, "each sample must have 2 parameters");
        assert!(
            sample[0] >= 0.0 && sample[0] <= 1.0,
            "param_0 = {} outside [0,1]",
            sample[0]
        );
        assert!(
            sample[1] >= 1.0 && sample[1] <= 2.0,
            "param_1 = {} outside [1,2]",
            sample[1]
        );
    }
}

#[test]
fn test_sensitivity_analysis() {
    // f(p) = 2·p₀ + 0.5·p₁  on [0,1]²
    // Variance decomposition (Sobol): V ∝ 4·Var(p₀) + 0.25·Var(p₁) = 4/12 + 0.25/12
    // S₁(p₀) ≈ 4/(4.25) ≈ 0.941, S₁(p₁) ≈ 0.25/(4.25) ≈ 0.059
    // Therefore total_sensitivity(p₀) >> total_sensitivity(p₁).
    let analyzer = SensitivityAnalyzer::new(SensitivityConfig::default()).unwrap();
    let model_fn = |params: &Array1<f64>| Array1::from_vec(vec![2.0 * params[0] + 0.5 * params[1]]);
    let parameter_ranges = vec![(0.0, 1.0), (0.0, 1.0)];

    let indices = analyzer.analyze(model_fn, &parameter_ranges, 50).unwrap();

    let sens0 = *indices.total.get("param_0").unwrap_or(&0.0);
    let sens1 = *indices.total.get("param_1").unwrap_or(&0.0);
    assert!(
        sens0 >= sens1,
        "param_0 total-sensitivity {sens0:.4} must be ≥ param_1 {sens1:.4} (ratio 4:0.25 in model)"
    );
    // Both indices must be non-negative (variance fractions).
    assert!(
        sens0 >= 0.0,
        "total sensitivity must be non-negative, got {sens0}"
    );
    assert!(
        sens1 >= 0.0,
        "total sensitivity must be non-negative, got {sens1}"
    );
}

#[test]
fn test_morris_screening() {
    // f(p) = p₀ + p₁: both parameters equally important, no interactions.
    // Morris μ* should be positive for both; σ should be small (linear model).
    let analyzer = SensitivityAnalyzer::new(SensitivityConfig::default()).unwrap();
    let model_fn = |params: &Array1<f64>| Array1::from_vec(vec![params[0] + params[1]]);
    let parameter_ranges = vec![(0.0, 1.0), (0.0, 1.0)];

    let results = analyzer
        .morris_screening(model_fn, &parameter_ranges, 5, 10)
        .unwrap();

    assert!(
        !results.mu.is_empty(),
        "Morris μ must contain entries for all parameters"
    );
    assert!(
        !results.sigma.is_empty(),
        "Morris σ must contain entries for all parameters"
    );

    // For an additive model μ entries must be non-negative.
    for (idx, &val) in results.mu.iter().enumerate() {
        assert!(
            val >= 0.0,
            "Morris μ[param_{idx}] = {val} must be non-negative for an additive model"
        );
    }
}
