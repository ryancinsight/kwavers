//! Beamforming Processor Trait
//!
//! Domain-level abstraction that all beamforming implementations must satisfy.
//! This allows different implementations (physics-based, neural network-based, adaptive, etc.)
//! to coexist without coupling clinical workflows to specific algorithm choices.

use super::types::BeamPattern;
use super::{BeamformingConfig, BeamformingResult};
use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Core beamforming processor trait
///
/// All implementations of beamforming (whether physics-based, neural-network-based,
/// or signal-processing-based) must implement this trait.
pub trait BeamformingProcessor: Send + Sync {
    /// Perform beamforming on sensor array signals
    ///
    /// # Arguments
    /// * `sensor_signals` - Array of signals from each sensor element
    /// * `config` - Beamforming configuration (array geometry, frequency, etc.)
    ///
    /// # Returns
    /// Beamformed signal with spatial beam pattern information
    fn beamform(
        &self,
        sensor_signals: &[Array3<f64>],
        config: &BeamformingConfig,
    ) -> KwaversResult<BeamformingResult>;

    /// Get beam pattern (spatial directivity)
    ///
    /// Returns the normalized beam response as a function of direction
    fn get_beam_pattern(&self, config: &BeamformingConfig) -> KwaversResult<BeamPattern>;

    /// Get processor name for diagnostics
    fn name(&self) -> &str;

    /// Get processor version for compatibility checking
    fn version(&self) -> (u32, u32, u32) {
        (0, 1, 0)
    }
}

/// Builder for beamforming implementations
///
/// Allows registration of different beamforming providers at runtime.
pub struct BeamformingRegistry {
    processors: std::collections::HashMap<String, Box<dyn Fn() -> Box<dyn BeamformingProcessor>>>,
}

impl std::fmt::Debug for BeamformingRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BeamformingRegistry")
            .field("num_processors", &self.processors.len())
            .field("processor_names", &self.list_processors())
            .finish()
    }
}

impl BeamformingRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            processors: std::collections::HashMap::new(),
        }
    }

    /// Register a beamforming processor factory
    pub fn register<F>(&mut self, name: impl Into<String>, factory: F)
    where
        F: Fn() -> Box<dyn BeamformingProcessor> + 'static,
    {
        self.processors.insert(name.into(), Box::new(factory));
    }

    /// Get a registered processor by name
    pub fn get(&self, name: &str) -> Option<Box<dyn BeamformingProcessor>> {
        self.processors.get(name).map(|f| f())
    }

    /// List all registered processor names
    pub fn list_processors(&self) -> Vec<&str> {
        self.processors.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for BeamformingRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_basic() {
        let mut registry = BeamformingRegistry::new();
        assert!(registry.list_processors().is_empty());
    }
}
