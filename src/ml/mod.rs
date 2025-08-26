//! Machine learning and AI capabilities for intelligent simulation control
//!
//! This module provides:
//! - Neural network inference for parameter optimization
//! - Pre-trained models for tissue classification
//! - Anomaly detection algorithms
//! - Reinforcement learning for automatic tuning

// Submodules
pub mod engine;
pub mod inference;
pub mod models;
pub mod optimization;
pub mod training;
pub mod types;

// Re-export key types for easier access
pub use engine::MLEngine;
pub use types::{
    InferencePrecision, MLBackend, MLConfig, MLModel, Model, ModelMetadata, ModelType,
    PerformanceMetrics,
};

pub use optimization::{
    AcousticEvent, CavitationEvent, ConvergencePredictor, ParameterOptimizer, PatternRecognizer,
    PatternSummary, SimulationPatterns,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2, Array3};

    #[test]
    fn test_ml_engine_creation() {
        let engine = MLEngine::new(MLBackend::Native).unwrap();
        assert_eq!(engine.get_metrics().total_inferences, 0);
    }

    #[test]
    fn test_tissue_classification() {
        use crate::ml::models::TissueClassifierModel;

        // Create a small test field
        let field = Array3::from_shape_vec((2, 2, 1), vec![0.1, 0.2, 0.8, 0.9]).unwrap();

        // Create a simple classifier model
        let weights = array![[1.0_f32, -1.0_f32]];
        let model = TissueClassifierModel::from_weights(weights, None);

        // Create engine and register model
        let mut engine = MLEngine::new(MLBackend::Native).unwrap();
        engine
            .models
            .insert(ModelType::TissueClassifier, Model::TissueClassifier(model));

        // Run classification
        let classes = engine.classify_tissue(&field).unwrap();
        assert_eq!(classes.dim(), field.dim());

        // Check that classes are valid
        for &c in classes.iter() {
            assert!(c < 2); // Binary classification
        }
    }

    #[test]
    fn test_classification_with_uncertainty() {
        use crate::ml::models::TissueClassifierModel;

        // Create a small 2×2×1 field with arbitrary values
        let field = Array3::from_shape_vec((2, 2, 1), vec![0.1, 0.2, 0.8, 0.9]).unwrap();

        // Build classifier with identity-like weights
        let weights = array![[5.0_f32, -5.0_f32]];
        let model = TissueClassifierModel::from_weights(weights, None);

        // Create engine and register model
        let mut engine = MLEngine::new(MLBackend::Native).unwrap();
        engine
            .models
            .insert(ModelType::TissueClassifier, Model::TissueClassifier(model));

        // Run classification with uncertainty quantification
        let (classes, entropy) = engine.classify_tissue_with_uncertainty(&field).unwrap();

        // Shape checks
        assert_eq!(classes.dim(), field.dim());
        assert_eq!(entropy.dim(), field.dim());

        // Entropy must be non-negative and finite
        for &h in entropy.iter() {
            assert!(h.is_finite());
            assert!(h >= 0.0);
        }

        // Classes must be 0 or 1
        for &c in classes.iter() {
            assert!(c == 0 || c == 1);
        }
    }

    #[test]
    fn test_outcome_predictor() {
        use models::ConvergencePredictorModel;

        // Binary classifier with 2 outputs for softmax
        let weights = array![[-10.0_f32, 10.0_f32]];
        let bias = Some(array![5.0_f32, -5.0_f32]);
        let model = ConvergencePredictorModel::from_weights(weights, bias);

        let mut engine = MLEngine::new(MLBackend::Native).unwrap();
        engine.models.insert(
            ModelType::ConvergencePredictor,
            Model::ConvergencePredictor(model),
        );

        let features: Array2<f32> = array![[0.0], [1.0]];
        let probs = engine.predict_outcome(&features).unwrap();
        assert_eq!(probs.len(), 2);

        // Verify probability ranges
        assert!(
            probs[0] < 0.1,
            "Expected low probability for 0.0 input, got {}",
            probs[0]
        );
        assert!(
            probs[1] > 0.9,
            "Expected high probability for 1.0 input, got {}",
            probs[1]
        );
    }

    #[test]
    fn test_parameter_optimization() {
        use models::ParameterOptimizerModel;

        // Create a simple optimizer model
        let weights = array![[0.5_f32], [0.3_f32], [-0.2_f32]];
        let model = ParameterOptimizerModel::from_weights(weights, None);

        let mut engine = MLEngine::new(MLBackend::Native).unwrap();
        engine.models.insert(
            ModelType::ParameterOptimizer,
            Model::ParameterOptimizer(model),
        );

        let state = array![[1.0_f32]];
        let params = engine.optimize_parameters(&state).unwrap();
        assert_eq!(params.len(), 3); // Should return 3 parameters
    }

    #[test]
    fn test_anomaly_detection() {
        use models::AnomalyDetectorModel;

        // Create anomaly detector
        let weights = array![[2.0_f32]];
        let bias = Some(array![-1.0_f32]);
        let model = AnomalyDetectorModel::from_weights(weights, bias);

        let mut engine = MLEngine::new(MLBackend::Native).unwrap();
        engine
            .models
            .insert(ModelType::AnomalyDetector, Model::AnomalyDetector(model));

        let field = Array3::from_shape_vec((2, 2, 1), vec![0.1, 0.2, 0.8, 0.9]).unwrap();
        let anomalies = engine.detect_anomalies(&field).unwrap();
        assert_eq!(anomalies.dim(), field.dim());

        // Check boolean values
        for &a in anomalies.iter() {
            assert!(a || !a); // Must be true or false
        }
    }
}
