//! Shared type definitions for the ML models module.

/// Metadata describing an ML model's identity, I/O contract, and performance profile.
#[derive(Debug, Clone)]
pub struct MlModelMetadata {
    /// Human-readable model identifier (e.g. `"TissueClassifier"`).
    pub name: String,
    /// Semantic version string (e.g. `"1.0.0"`).
    pub version: String,
    /// Expected input tensor shape excluding the batch dimension.
    pub input_shape: Vec<usize>,
    /// Output tensor shape excluding the batch dimension.
    pub output_shape: Vec<usize>,
    /// Validation accuracy in [0, 1].
    pub accuracy: f64,
    /// Mean single-sample inference latency in milliseconds.
    pub inference_time_ms: f64,
}
