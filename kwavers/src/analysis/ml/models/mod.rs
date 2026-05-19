//! Machine learning models for acoustic simulation
//!
//! This module provides ML models for various simulation tasks.
//! Each model is in its own submodule for proper separation of concerns.

pub mod anomaly_detector;
pub mod convergence_predictor;
pub mod outcome_predictor;
pub mod parameter_optimizer;
pub mod tissue_classifier;

// Re-export main types
pub use anomaly_detector::AnomalyDetectorModel;
pub use convergence_predictor::ConvergencePredictorModel;
pub use outcome_predictor::OutcomePredictorModel;
pub use parameter_optimizer::ParameterOptimizerModel;
pub use tissue_classifier::TissueClassifierModel;

use crate::core::error::KwaversResult;
use ndarray::Array2;

/// Common trait for all ML models
pub trait MLModel {
    /// Run inference on input data
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn predict(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>>;

    /// Get model accuracy metric
    fn accuracy(&self) -> f64;

    /// Get model name
    fn name(&self) -> &str;
}

// Re-export MlModelMetadata from types
pub use crate::analysis::ml::types::MlModelMetadata;

/// Discriminator for the available ML model types.
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    /// Acoustic tissue classification (multi-class).
    TissueClassifier,
    /// Simulation parameter optimization (regression).
    ParameterOptimizer,
    /// Outlier / anomaly detection (binary output).
    AnomalyDetector,
    /// Solver convergence probability prediction.
    ConvergencePredictor,
    /// Treatment outcome prediction (three-class).
    OutcomePredictor,
}

/// Compile-time constants shared across all ML models.
pub mod constants {
    /// Default mini-batch size for batched inference.
    pub const DEFAULT_BATCH_SIZE: usize = 32;

    /// Maximum input sequence length for temporal models.
    pub const MAX_SEQUENCE_LENGTH: usize = 1000;

    /// Default learning rate used when no explicit rate is provided.
    pub const DEFAULT_LEARNING_RATE: f32 = 0.001;

    /// Loss improvement below which training is considered converged.
    pub const CONVERGENCE_THRESHOLD: f32 = 1e-6;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    // ── AnomalyDetectorModel ─────────────────────────────────────────────────

    // 3-σ detector: |x| > 3.0 → 1.0, |x| ≤ 3.0 → 0.0.
    // Analytic truth: row [4.0] → 1.0; row [1.0] → 0.0.
    #[test]
    fn test_anomaly_detector_threshold_correct() {
        let model = AnomalyDetectorModel::new(3.0);
        let input = array![[4.0_f32], [1.0_f32], [-4.0_f32], [3.0_f32]];
        let output = model.predict(&input).unwrap();

        assert_eq!(output.shape(), &[4, 1]);
        assert!((output[[0, 0]] - 1.0).abs() < 1e-6, "4.0 > 3.0 → 1.0");
        assert!((output[[1, 0]] - 0.0).abs() < 1e-6, "1.0 ≤ 3.0 → 0.0");
        assert!((output[[2, 0]] - 1.0).abs() < 1e-6, "|-4.0| > 3.0 → 1.0");
        assert!((output[[3, 0]] - 0.0).abs() < 1e-6, "3.0 not > 3.0 → 0.0");
    }

    #[test]
    fn test_anomaly_detector_metadata() {
        let model = AnomalyDetectorModel::new(3.0);
        assert_eq!(model.name(), "AnomalyDetector");
        assert!((model.accuracy() - 0.95).abs() < 1e-9);
    }

    // ── ConvergencePredictorModel ─────────────────────────────────────────────

    // Sigmoid is monotone: a positive mean maps to probability > 0.5.
    // For mean=0 the sigmoid equals exactly 0.5 (by definition of sigmoid(0)).
    #[test]
    fn test_convergence_predictor_output_range_and_shape() {
        let model = ConvergencePredictorModel::new();
        // Row of all-zeros → mean=0 → sigmoid(0)=0.5.
        let input = Array2::<f32>::zeros((5, 10));
        let output = model.predict(&input).unwrap();

        assert_eq!(output.shape(), &[5, 1]);
        for val in output.iter() {
            assert!(
                *val >= 0.0 && *val <= 1.0,
                "probability out of [0,1]: {val}"
            );
        }
        // Mean=0 → p=0.5 analytically.
        assert!((output[[0, 0]] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_convergence_predictor_positive_mean_above_half() {
        let model = ConvergencePredictorModel::new();
        let input = Array2::from_elem((1, 10), 1.0_f32); // mean=1.0 > 0 → p > 0.5
        let output = model.predict(&input).unwrap();
        assert!(output[[0, 0]] > 0.5, "positive mean should yield p > 0.5");
    }

    // ── OutcomePredictorModel ─────────────────────────────────────────────────

    // Three output channels; values analytically derived from clamp(mean, 0,1).
    // For input row [0.4, 0.4]: mean=0.4 → ch0=0.4, ch1=0.6, ch2=0.2.
    #[test]
    fn test_outcome_predictor_output_shape_and_values() {
        let model = OutcomePredictorModel::new();
        let input = Array2::from_elem((3, 20), 0.4_f32);
        let output = model.predict(&input).unwrap();

        assert_eq!(output.shape(), &[3, 3]);
        for i in 0..3 {
            assert!((output[[i, 0]] - 0.4).abs() < 1e-5, "ch0 should be 0.4");
            assert!((output[[i, 1]] - 0.6).abs() < 1e-5, "ch1 should be 0.6");
            assert!((output[[i, 2]] - 0.2).abs() < 1e-5, "ch2 should be 0.2");
        }
    }

    #[test]
    fn test_outcome_predictor_empty_input_returns_error() {
        let model = OutcomePredictorModel::new();
        let empty = Array2::<f32>::zeros((0, 20));
        assert!(
            model.predict(&empty).is_err(),
            "empty input must return an error"
        );
    }

    // ── ParameterOptimizerModel ──────────────────────────────────────────────

    // Output shape must be (n_samples, output_dim) with all finite values.
    #[test]
    fn test_parameter_optimizer_output_shape_and_finite() {
        let model = ParameterOptimizerModel::new(8, 3);
        let input = Array2::<f32>::ones((4, 8));
        let output = model.predict(&input).unwrap();

        assert_eq!(output.shape(), &[4, 3]);
        for &v in output.iter() {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    #[test]
    fn test_parameter_optimizer_from_weights_shape() {
        let weights = Array2::<f32>::eye(4);
        let bias = Some(Array1::from_elem(4, 0.1_f32));
        let model = ParameterOptimizerModel::from_weights(weights, bias);

        let input = array![[1.0_f32, 0.0, 0.0, 0.0]];
        let output = model.predict(&input).unwrap();

        assert_eq!(output.shape(), &[1, 4]);
        // W=I, b=0.1: output[0,0] = 1.0 + 0.1 = 1.1
        assert!((output[[0, 0]] - 1.1).abs() < 1e-5);
    }

    // ── TissueClassifierModel ────────────────────────────────────────────────

    // classify() must return a valid class index in [0, n_classes).
    #[test]
    fn test_tissue_classifier_class_index_in_bounds() {
        let model = TissueClassifierModel::with_random_weights(8, 5);
        let features = Array1::<f32>::ones(8);
        let class = model.classify(&features).unwrap();
        assert!(class < 5, "class index {class} out of bounds [0,5)");
    }

    #[test]
    fn test_tissue_classifier_predict_shape_and_finite() {
        let model = TissueClassifierModel::with_random_weights(8, 5);
        let input = Array2::<f32>::ones((3, 8));
        let output = model.predict(&input).unwrap();

        assert_eq!(output.shape(), &[3, 5]);
        for &v in output.iter() {
            assert!(v.is_finite());
        }
    }

    // ModelType must be cloneable and debuggable (derived).
    #[test]
    fn test_model_type_clone_debug() {
        let mt = ModelType::TissueClassifier;
        let cloned = mt;
        assert!(matches!(cloned, ModelType::TissueClassifier));
        let s = format!("{cloned:?}");
        assert!(s.contains("TissueClassifier"));
    }
}
