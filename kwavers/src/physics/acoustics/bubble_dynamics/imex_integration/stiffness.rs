//! Stiffness estimation and adaptive time step suggestion

use super::integrator::BubbleIMEXIntegrator;
use crate::physics::acoustics::bubble_dynamics::BubbleState;

impl BubbleIMEXIntegrator {
    /// Calculate stiffness ratio based on characteristic time scales
    #[must_use]
    pub fn estimate_stiffness(&self, state: &BubbleState) -> f64 {
        let params = self.solver.params();
        let thermal_diffusivity =
            params.thermal_conductivity / (params.rho_liquid * params.specific_heat_liquid);

        let mechanical_timescale = state.radius / state.wall_velocity.abs().max(1e-10);
        let thermal_timescale = state.radius.powi(2) / thermal_diffusivity;

        mechanical_timescale / thermal_timescale.min(mechanical_timescale)
    }

    /// Suggest time step based on stiffness
    #[must_use]
    pub fn suggest_timestep(&self, state: &BubbleState) -> f64 {
        let stiffness = self.estimate_stiffness(state);

        if stiffness > 100.0 {
            self.config.dt_min * 10.0
        } else if stiffness > 10.0 {
            (self.config.dt_min + self.config.dt_max) / 2.0
        } else {
            self.config.dt_max
        }
    }
}
