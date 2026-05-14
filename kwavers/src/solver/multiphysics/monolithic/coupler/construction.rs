use super::super::config::{NewtonKrylovConfig, PhysicsCoefficients};
use super::MonolithicCoupler;
use crate::solver::integration::nonlinear::GMRESConfig;
use std::collections::HashMap;

impl MonolithicCoupler {
    /// Create a monolithic coupler with default coupled-physics coefficients.
    #[must_use]
    pub fn new(newton_config: NewtonKrylovConfig, gmres_config: GMRESConfig) -> Self {
        Self::with_coefficients(newton_config, gmres_config, PhysicsCoefficients::default())
    }

    /// Create a monolithic coupler with explicit physics coefficients.
    #[must_use]
    pub fn with_coefficients(
        newton_config: NewtonKrylovConfig,
        gmres_config: GMRESConfig,
        coefficients: PhysicsCoefficients,
    ) -> Self {
        Self {
            newton_config,
            gmres_config,
            convergence_history: Vec::new(),
            physics_components: HashMap::new(),
            physics_coefficients: coefficients,
            du_scratch: None,
            u_prev_scratch: None,
            rhs_scratch: None,
            line_search_state_scratch: None,
            jvp_state_scratch: None,
            gmres_solver: None,
            grid_spacing: (1e-3, 1e-3, 1e-3),
        }
    }
}
