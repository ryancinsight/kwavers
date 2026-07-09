//! Tests for uncertainty quantification framework.

use super::quantifier::UncertaintyQuantifier;
use super::types::{MlUncertaintyConfig, MlUncertaintyMethod};
use leto::Array3;

#[cfg(feature = "pinn")]
struct LinearPinnPredictor;

#[cfg(feature = "pinn")]
impl super::PinnUncertaintyPredictor for LinearPinnPredictor {
    fn predict_coordinates(
        &self,
        x: &leto::Array1<f64>,
        t: &leto::Array1<f64>,
    ) -> kwavers_core::error::KwaversResult<leto::Array2<f32>> {
        let mut prediction = leto::Array2::zeros((x.len(), 1));
        for idx in 0..x.len() {
            prediction[[idx, 0]] = (x[idx] + 2.0 * t[idx]) as f32;
        }
        Ok(prediction)
    }
}

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

#[cfg(feature = "pinn")]
#[test]
fn test_pinn_uncertainty_uses_solver_agnostic_predictor() {
    let config = MlUncertaintyConfig {
        method: MlUncertaintyMethod::MonteCarloDropout,
        num_samples: 3,
        confidence_level: 0.95,
        dropout_rate: 0.1,
        ensemble_size: 2,
        calibration_size: 4,
    };
    let quantifier = UncertaintyQuantifier::new(config).unwrap();
    let inputs = leto::array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]];

    let result = quantifier
        .quantify_pinn_uncertainty(&LinearPinnPredictor, &inputs, None)
        .unwrap();

    assert_eq!(result.mean_prediction.dim(), (2, 1));
    assert_eq!(result.mean_prediction[[0, 0]], 5.0);
    assert_eq!(result.mean_prediction[[1, 0]], 11.0);
    assert_eq!(result.uncertainty[[0, 0]], 0.0);
    assert_eq!(result.uncertainty[[1, 0]], 0.0);
    assert_eq!(result.reliability_score, 1.0);
}

#[cfg(feature = "pinn")]
#[test]
fn test_pinn_uncertainty_rejects_missing_time_column() {
    let config = MlUncertaintyConfig {
        method: MlUncertaintyMethod::MonteCarloDropout,
        num_samples: 3,
        confidence_level: 0.95,
        dropout_rate: 0.1,
        ensemble_size: 2,
        calibration_size: 4,
    };
    let quantifier = UncertaintyQuantifier::new(config).unwrap();
    let inputs = leto::Array2::from_elem((2, 1), 1.0_f32);

    let err = quantifier
        .quantify_pinn_uncertainty(&LinearPinnPredictor, &inputs, None)
        .unwrap_err();
    let msg = format!("{err:?}");

    assert!(
        msg.contains("x and t columns"),
        "invalid input error must name the missing coordinate contract; got: {msg}"
    );
}

#[test]
fn test_beamforming_uncertainty() {
    // confidence_score = signal_quality.clamp(0,1) = 0.8
    // uniform image (all 1.0): all neighbors == center → variance=0 → uncertainty_map=0.0
    let config = MlUncertaintyConfig::default();
    let quantifier = UncertaintyQuantifier::new(config).unwrap();

    let image = Array3::from_elem([32, 32, 16], 1.0_f32);
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

    let image = Array3::from_elem([16, 16, 8], 1.0);
    let uncertainty = quantifier
        .quantify_beamforming_uncertainty(&image, 0.9)
        .unwrap();

    assert!(quantifier.is_confident(&uncertainty, 0.5));
    assert!(!quantifier.is_confident(&uncertainty, 0.95));
}
