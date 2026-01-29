//! Source Localization Abstraction
//!
//! Domain-level interface for source localization implementations.

use crate::core::error::KwaversResult;

/// Source location result
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// Position [x, y, z] in meters
    pub position: [f64; 3],

    /// Confidence (0.0-1.0)
    pub confidence: f64,

    /// Uncertainty radius [m]
    pub uncertainty: f64,
}

/// Localization algorithm trait
pub trait LocalizationProcessor: Send + Sync {
    /// Localize source from time-delay measurements
    fn localize(
        &self,
        time_delays: &[f64],
        sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<SourceLocation>;

    /// Get processor name
    fn name(&self) -> &str;
}
