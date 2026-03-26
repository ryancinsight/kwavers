//! Core IMEX integrator: step function, Jacobian, state vector conversion
//!
//! ## Algorithm (IMEX Euler)
//!
//! 1. **Explicit step**: Update R, v using mechanical acceleration (non-stiff)
//! 2. **Implicit step**: Newton iteration for T, n_vapor (stiff thermal/mass transfer)
//!
//! Residual: y_final − y_explicit − dt · f_implicit(y_final) = 0

use super::config::BubbleIMEXConfig;
use crate::core::error::{KwaversResult, PhysicsError};
use crate::physics::acoustics::bubble_dynamics::{BubbleState, KellerMiksisModel};
use ndarray::Array1;
use std::sync::Arc;

use crate::core::constants::thermodynamic::VAPOR_DIFFUSION_COEFFICIENT;

/// IMEX integrator for bubble dynamics
#[derive(Debug)]
pub struct BubbleIMEXIntegrator {
    pub(crate) solver: Arc<KellerMiksisModel>,
    pub(crate) config: BubbleIMEXConfig,
}

impl BubbleIMEXIntegrator {
    /// Create a new IMEX integrator for bubble dynamics
    #[must_use]
    pub fn new(solver: Arc<KellerMiksisModel>, config: BubbleIMEXConfig) -> Self {
        Self { solver, config }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults(solver: Arc<KellerMiksisModel>) -> Self {
        Self::new(solver, BubbleIMEXConfig::default())
    }

    /// Update configuration
    pub fn set_config(&mut self, config: BubbleIMEXConfig) {
        self.config = config;
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &BubbleIMEXConfig {
        &self.config
    }

    /// Get solver
    #[must_use]
    pub fn solver(&self) -> &Arc<KellerMiksisModel> {
        &self.solver
    }

    /// Integrate bubble dynamics for one time step using IMEX
    pub fn step(
        &mut self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        dt: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        let y0 = self.state_to_vector(state);

        // Step 1: Explicit update for mechanical terms
        let mut y_explicit = y0.clone();
        {
            let mut temp_state = self.vector_to_state(&y0, state)?;

            let accel =
                self.solver
                    .calculate_acceleration(&mut temp_state, p_acoustic, dp_dt, t)?;

            y_explicit[0] = y0[0] + dt * temp_state.wall_velocity;
            y_explicit[1] = y0[1] + dt * accel;
            y_explicit[2] = y0[2];
            y_explicit[3] = y0[3];
        }

        // Step 2: Implicit update for thermal and mass transfer
        let mut y_final = y_explicit.clone();
        {
            for iter in 0..self.config.max_iter {
                let state_current = self.vector_to_state(&y_final, state)?;

                let (dt_dt, dn_vapor_dt) =
                    self.calculate_thermal_mass_transfer_rates(&state_current)?;

                let residual = Array1::from_vec(vec![
                    0.0,
                    0.0,
                    y_final[2] - y_explicit[2] - dt * dt_dt,
                    y_final[3] - y_explicit[3] - dt * dn_vapor_dt,
                ]);

                let residual_norm = residual.iter().map(|x| x.abs()).fold(0.0, f64::max);
                if residual_norm < self.config.atol {
                    break;
                }

                let jac = self.compute_jacobian_diagonal(&y_final, dt)?;

                for i in 2..4 {
                    if jac[i].abs() > 1e-12 {
                        y_final[i] -= residual[i] / jac[i];
                    }
                }

                if iter == self.config.max_iter - 1 {
                    break;
                }
            }
        }

        *state = self.vector_to_state(&y_final, state)?;

        state.update_compression(self.solver.params().r0);
        state.update_collapse_state();

        Ok(dt)
    }

    /// Compute diagonal Jacobian approximation for implicit solver
    pub(crate) fn compute_jacobian_diagonal(
        &self,
        y: &Array1<f64>,
        dt: f64,
    ) -> KwaversResult<Array1<f64>> {
        let mut jac = Array1::ones(4);
        let r = y[0];
        let t_bubble = y[2];
        let params = self.solver.params();

        jac[0] = 1.0;
        jac[1] = 1.0;

        let thermal_diffusion_rate = 3.0 * params.thermal_conductivity
            / (params.rho_liquid * params.specific_heat_liquid * r * r);
        let mass_transfer_coupling = if t_bubble > 0.0 {
            crate::physics::constants::WATER_LATENT_HEAT_VAPORIZATION * params.accommodation_coeff
                / (params.specific_heat_liquid * t_bubble)
        } else {
            0.0
        };
        jac[2] = 1.0 + dt * (thermal_diffusion_rate + mass_transfer_coupling);

        let vapor_diffusion_rate = if r > 1e-9 {
            3.0 * VAPOR_DIFFUSION_COEFFICIENT / (r * r)
        } else {
            0.0
        };
        jac[3] = 1.0 + dt * vapor_diffusion_rate;

        Ok(jac)
    }

    /// Convert bubble state to vector form
    pub(crate) fn state_to_vector(&self, state: &BubbleState) -> Array1<f64> {
        let mut y = Array1::zeros(4);
        y[0] = state.radius;
        y[1] = state.wall_velocity;
        y[2] = state.temperature;
        y[3] = state.n_vapor;
        y
    }

    /// Convert vector to bubble state
    pub(crate) fn vector_to_state(
        &self,
        y: &Array1<f64>,
        template: &BubbleState,
    ) -> KwaversResult<BubbleState> {
        if y.len() != 4 {
            return Err(PhysicsError::InvalidState {
                field: "state_vector".to_string(),
                value: format!("length {}", y.len()),
                reason: "Expected 4 elements".to_string(),
            }
            .into());
        }

        let mut state = template.clone();
        state.radius = y[0];
        state.wall_velocity = y[1];
        state.temperature = y[2];
        state.n_vapor = y[3];

        Ok(state)
    }
}
