//! CEUS Orchestration Interface
//!
//! Domain-level abstraction for CEUS (Contrast-Enhanced Ultrasound) simulation orchestration.
//! This allows clinical layer to depend on domain abstractions rather than simulation internals.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::Array3;

/// CEUS orchestration interface
///
/// Provides a consistent interface for CEUS simulation regardless of implementation details.
/// This is what the clinical therapy layer should depend on.
pub trait CEUSOrchestrator: Send + Sync + std::fmt::Debug {
    /// Update CEUS simulation with new acoustic field
    ///
    /// # Arguments
    /// * `pressure_field` - Acoustic pressure field
    /// * `time` - Current simulation time (s)
    ///
    /// # Returns
    /// Updated image
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn update(&mut self, pressure_field: &Array3<f64>, time: f64) -> KwaversResult<Array3<f64>>;

    /// Get perfusion model output
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn get_perfusion_data(&self) -> KwaversResult<Array3<f64>>;

    /// Get microbubble concentration map
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn get_concentration_map(&self) -> KwaversResult<Array3<f64>>;

    /// Get name for diagnostics
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn name(&self) -> &str;
}

/// Boxed factory closure constructing a CEUS orchestrator from a grid, medium,
/// and two scalar parameters.
type CeusOrchestratorCreator =
    Box<dyn Fn(&Grid, &dyn Medium, f64, f64) -> KwaversResult<Box<dyn CEUSOrchestrator>>>;

/// Factory for creating CEUS orchestrators
///
/// Allows different implementations to be registered and instantiated at runtime.
pub struct CEUSOrchestrators {
    default_creator: Option<CeusOrchestratorCreator>,
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_creator: None,
        }
    }

    /// Register default CEUS orchestrator factory
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn set_default<F>(&mut self, factory: F)
    where
        F: Fn(&Grid, &dyn Medium, f64, f64) -> KwaversResult<Box<dyn CEUSOrchestrator>> + 'static,
    {
        self.default_creator = Some(Box::new(factory));
    }

    /// Create default CEUS orchestrator
    ///
    /// # Boundary contract
    ///
    /// The domain layer owns the orchestration trait and registry only. Concrete
    /// CEUS simulation implementations are registered by an application or
    /// physics assembly layer. An empty registry is therefore a configuration
    /// state, not a placeholder implementation.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn create_default(
        &self,
        grid: &Grid,
        medium: &dyn Medium,
        bubble_concentration: f64,
        bubble_size: f64,
    ) -> KwaversResult<Box<dyn CEUSOrchestrator>> {
        self.default_creator.as_ref().ok_or_else(|| {
            kwavers_core::error::KwaversError::FeatureNotAvailable(
                "CEUS orchestrator registry has no default factory; register a concrete CEUS \
                 implementation before requesting default orchestration"
                    .to_owned(),
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
    use kwavers_medium::homogeneous::HomogeneousMedium;

    #[test]
    fn test_ceus_orchestrators_creation() {
        let registry = CEUSOrchestrators::new();

        assert!(format!("{registry:?}").contains("default_creator: false"));
    }

    #[test]
    fn create_default_rejects_unregistered_factory_without_placeholder_path() {
        let registry = CEUSOrchestrators::new();
        let grid = Grid::new(2, 2, 2, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium = HomogeneousMedium::water(&grid);

        let error = registry
            .create_default(&grid, &medium, 1.0e6, 2.0e-6)
            .unwrap_err();

        assert!(matches!(
            error,
            kwavers_core::error::KwaversError::FeatureNotAvailable(_)
        ));
        assert!(format!("{error}").contains("no default factory"));
    }
}
