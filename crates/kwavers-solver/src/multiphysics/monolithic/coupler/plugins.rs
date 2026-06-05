use super::super::config::PhysicsCoefficients;
use super::MonolithicCoupler;
use crate::plugin::Plugin;
use kwavers_core::error::KwaversResult;

impl MonolithicCoupler {
    /// Replace physical coefficients used by coupled residual evaluation.
    pub fn set_physics_coefficients(&mut self, coefficients: PhysicsCoefficients) {
        self.physics_coefficients = coefficients;
    }

    /// Register a named physics component for plugin-backed coupling.
    ///
    /// # Errors
    /// - Currently returns `Ok(())`; the `Result` preserves the plugin boundary
    ///   contract for future validation without changing the public method.
    pub fn register_physics(
        &mut self,
        name: String,
        physics: Box<dyn Plugin>,
    ) -> KwaversResult<()> {
        self.physics_components.insert(name, physics);
        Ok(())
    }
}
