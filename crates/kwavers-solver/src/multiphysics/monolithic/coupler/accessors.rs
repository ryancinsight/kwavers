use super::super::config::PhysicsCoefficients;
use super::MonolithicCoupler;

impl MonolithicCoupler {
    /// Return Newton residual norms recorded during the most recent solve.
    #[must_use]
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Return the active coupled-physics coefficients.
    #[must_use]
    pub fn physics_coefficients(&self) -> &PhysicsCoefficients {
        &self.physics_coefficients
    }
}
