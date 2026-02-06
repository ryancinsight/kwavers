//! Signal Filtering Abstraction
//!
//! Domain-level interface for signal filtering operations.

use crate::core::error::KwaversResult;
use ndarray::Array1;

/// Signal filter processor trait
pub trait FilterProcessor: Send + Sync {
    /// Apply filter to signal
    fn filter(&self, signal: &Array1<f64>) -> KwaversResult<Array1<f64>>;

    /// Get filter name
    fn name(&self) -> &str;
}
