//! Tests for uncertainty quantification framework.

use super::quantifier::UncertaintyQuantifier;
use super::types::{UncertaintyConfig, UncertaintyMethod};
use ndarray::Array3;

#[test]
fn test_uncertainty_quantifier_creation() {
    let config = UncertaintyConfig {
        method: UncertaintyMethod::MonteCarloDropout,
        num_samples: 10,
        confidence_level: 0.95,
        dropout_rate: 0.1,
        ensemble_size: 5,
        calibration_size: 100,
    };

    let quantifier = UncertaintyQuantifier::new(config);
    assert!(quantifier.is_ok());
}

#[test]
fn test_beamforming_uncertainty() {
    let config = UncertaintyConfig::default();
    let quantifier = UncertaintyQuantifier::new(config).unwrap();

    let image = Array3::from_elem((32, 32, 16), 1.0);
    let uncertainty = quantifier.quantify_beamforming_uncertainty(&image, 0.8);

    assert!(uncertainty.is_ok());
    let result = uncertainty.unwrap();
    assert_eq!(result.uncertainty_map.dim(), (32, 32, 16));
    assert!(result.confidence_score >= 0.0 && result.confidence_score <= 1.0);
}

#[test]
fn test_confidence_check() {
    let config = UncertaintyConfig::default();
    let quantifier = UncertaintyQuantifier::new(config).unwrap();

    let image = Array3::from_elem((16, 16, 8), 1.0);
    let uncertainty = quantifier
        .quantify_beamforming_uncertainty(&image, 0.9)
        .unwrap();

    assert!(quantifier.is_confident(&uncertainty, 0.5));
    assert!(!quantifier.is_confident(&uncertainty, 0.95));
}
