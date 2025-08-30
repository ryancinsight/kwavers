//! Configuration for heterogeneous media handling

use crate::constants::numerical::HETEROGENEOUS_SMOOTHING_FACTOR;

/// Configuration for heterogeneous media handling
#[derive(Debug, Clone))]
pub struct HeterogeneousConfig {
    /// Enable Gibbs phenomenon mitigation
    pub mitigate_gibbs: bool,
    /// Smoothing method for interfaces
    pub smoothing_method: SmoothingMethod,
    /// Interface detection threshold (relative change)
    pub interface_threshold: f64,
    /// Smoothing kernel width (in grid points)
    pub smoothing_width: f64,
    /// Use pressure-velocity split formulation
    pub use_pv_split: bool,
    /// Adaptive treatment based on interface sharpness
    pub adaptive_treatment: bool,
}

impl Default for HeterogeneousConfig {
    fn default() -> Self {
        Self {
            mitigate_gibbs: true,
            smoothing_method: SmoothingMethod::Gaussian,
            interface_threshold: 0.1, // 10% change indicates interface
            smoothing_width: 2.0 * HETEROGENEOUS_SMOOTHING_FACTOR,
            use_pv_split: true,
            adaptive_treatment: true,
        }
    }
}

/// Smoothing methods for interface treatment
#[derive(Debug, Clone, Copy, PartialEq))]
pub enum SmoothingMethod {
    /// No smoothing (for comparison)
    None,
    /// Gaussian kernel smoothing
    Gaussian,
    /// Hyperbolic tangent transition
    Tanh,
    /// Polynomial (cubic) transition
    Polynomial,
    /// Spectral filtering (remove high frequencies)
    SpectralFilter,
}
