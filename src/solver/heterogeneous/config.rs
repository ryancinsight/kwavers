//! Configuration for heterogeneous media solvers

use crate::constants::numerical::HETEROGENEOUS_SMOOTHING_FACTOR;

/// Configuration for heterogeneous media handling
#[derive(Debug, Clone)]
pub struct HeterogeneousConfig {
    /// Enable k-space correction for heterogeneous media
    pub enable_kspace_correction: bool,
    
    /// Enable density smoothing at interfaces
    pub enable_density_smoothing: bool,
    
    /// Smoothing factor for interface transitions
    pub smoothing_factor: f64,
    
    /// Use staggered grid for improved accuracy
    pub use_staggered_grid: bool,
    
    /// Enable adaptive interface detection
    pub adaptive_interfaces: bool,
    
    /// Interface detection threshold
    pub interface_threshold: f64,
}

impl Default for HeterogeneousConfig {
    fn default() -> Self {
        Self {
            enable_kspace_correction: true,
            enable_density_smoothing: true,
            smoothing_factor: HETEROGENEOUS_SMOOTHING_FACTOR,
            use_staggered_grid: true,
            adaptive_interfaces: false,
            interface_threshold: 0.1,
        }
    }
}