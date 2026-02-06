//! Passive Acoustic Mapping (PAM) Abstraction
//!
//! Domain-level interface for PAM implementations.

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Result from PAM processing
#[derive(Debug, Clone)]
pub struct PAMResult {
    /// Source locations detected
    pub source_locations: Vec<[f64; 3]>,

    /// Confidence scores (0.0-1.0)
    pub confidence: Vec<f64>,

    /// Intensity distribution
    pub intensity_map: Array3<f64>,
}

/// PAM processor trait
pub trait PAMProcessor: Send + Sync {
    /// Compute PAM from sensor signals
    fn compute_pam(&self, sensor_signals: &[Array3<f64>]) -> KwaversResult<PAMResult>;

    /// Get processor name
    fn name(&self) -> &str;
}
