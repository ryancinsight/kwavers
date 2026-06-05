//! Tests for uncertainty quantification framework.

use super::quantifier::UncertaintyQuantifier;
use super::types::{MlUncertaintyConfig, MlUncertaintyMethod};
use ndarray::Array3;

#[test]
fn test_uncertainty_quantifier_creation() {
    // MonteCarloDropout: bayesian=Some, conformal=None, ensemble=None, sensitivity=None
    let config = MlUncertaintyConfig {
        method: MlUncertaintyMethod::MonteCarloDropout,
        num_samples: 10,
        confidence_level: 0.95,
        dropout_rate: 0.1,
        ensemble_size: 5,
        calibration_size: 100,
    };

    let quantifier = UncertaintyQuantifier::new(config).unwrap();
    // MonteCarloDropout activates Bayesian path only.
    assert!(
        quantifier._bayesian.is_some(),
        "MonteCarloDropout method must initialize Bayesian component"
    );
    assert!(
        quantifier._ensemble.is_none(),
        "MonteCarloDropout method must not initialize Ensemble component"
    );
}

#[test]
fn test_beamforming_uncertainty() {
    // confidence_score = signal_quality.clamp(0,1) = 0.8
    // uniform image (all 1.0): all neighbors == center → variance=0 → uncertainty_map=0.0
    let config = MlUncertaintyConfig::default();
    let quantifier = UncertaintyQuantifier::new(config).unwrap();

    let image = Array3::from_elem((32, 32, 16), 1.0_f32);
    let result = quantifier
        .quantify_beamforming_uncertainty(&image, 0.8)
        .unwrap();
    assert_eq!(result.uncertainty_map.dim(), (32, 32, 16));
    let confidence_err = (result.confidence_score - 0.8).abs();
    assert!(
        confidence_err < 1e-10,
        "confidence_score = {} (expected 0.8 = signal_quality.clamp(0,1))",
        result.confidence_score
    );
    // Uniform field: zero spatial variance → uncertainty_map is all zeros.
    let max_unc = result
        .uncertainty_map
        .iter()
        .cloned()
        .fold(0.0_f32, f32::max);
    assert!(
        max_unc < 1e-6,
        "uniform image must produce zero uncertainty_map; got max={max_unc}"
    );
}

#[test]
fn test_confidence_check() {
    let config = MlUncertaintyConfig::default();
    let quantifier = UncertaintyQuantifier::new(config).unwrap();

    let image = Array3::from_elem((16, 16, 8), 1.0);
    let uncertainty = quantifier
        .quantify_beamforming_uncertainty(&image, 0.9)
        .unwrap();

    assert!(quantifier.is_confident(&uncertainty, 0.5));
    assert!(!quantifier.is_confident(&uncertainty, 0.95));
}
