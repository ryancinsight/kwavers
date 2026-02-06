//! CEUS Orchestration Interface
//!
//! Domain-level abstraction for CEUS (Contrast-Enhanced Ultrasound) simulation orchestration.
//! This allows clinical layer to depend on domain abstractions rather than simulation internals.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;

/// CEUS orchestration interface
///
/// Provides a consistent interface for CEUS simulation regardless of implementation details.
/// This is what the clinical therapy layer should depend on.
pub trait CEUSOrchestrator: Send + Sync + std::fmt::Debug {
    /// Update CEUS simulation with new acoustic field
    ///
    /// # Arguments
    /// * `pressure_field` - Acoustic pressure field
    /// * `time` - Current simulation time [s]
    ///
    /// # Returns
    /// Updated image
    fn update(&mut self, pressure_field: &Array3<f64>, time: f64) -> KwaversResult<Array3<f64>>;

    /// Get perfusion model output
    fn get_perfusion_data(&self) -> KwaversResult<Array3<f64>>;

    /// Get microbubble concentration map
    fn get_concentration_map(&self) -> KwaversResult<Array3<f64>>;

    /// Get name for diagnostics
    fn name(&self) -> &str;
}

/// Factory for creating CEUS orchestrators
///
/// Allows different implementations to be registered and instantiated at runtime.
pub struct CEUSOrchestrators {
    default_creator: Option<
        Box<dyn Fn(&Grid, &dyn Medium, f64, f64) -> KwaversResult<Box<dyn CEUSOrchestrator>>>,
    >,
}

impl std::fmt::Debug for CEUSOrchestrators {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CEUSOrchestrators")
            .field("default_creator", &self.default_creator.is_some())
            .finish()
    }
}

impl CEUSOrchestrators {
    /// Create new factory
    pub fn new() -> Self {
        Self {
            default_creator: None,
        }
    }

    /// Register default CEUS orchestrator factory
    pub fn set_default<F>(&mut self, factory: F)
    where
        F: Fn(&Grid, &dyn Medium, f64, f64) -> KwaversResult<Box<dyn CEUSOrchestrator>> + 'static,
    {
        self.default_creator = Some(Box::new(factory));
    }

    /// Create default CEUS orchestrator
    pub fn create_default(
        &self,
        grid: &Grid,
        medium: &dyn Medium,
        bubble_concentration: f64,
        bubble_size: f64,
    ) -> KwaversResult<Box<dyn CEUSOrchestrator>> {
        self.default_creator.as_ref().ok_or_else(|| {
            crate::core::error::KwaversError::NotImplemented(
                "CEUS orchestrator not registered".to_string(),
            )
        })?(grid, medium, bubble_concentration, bubble_size)
    }
}

impl Default for CEUSOrchestrators {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ceus_orchestrators_creation() {
        let _registry = CEUSOrchestrators::new();
    }
}
