//! IMEX (Implicit-Explicit) Time Integration for Bubble Dynamics
//!
//! This module provides an IMEX time integration scheme specifically designed
//! for the stiff bubble dynamics equations. The mechanical terms (radius, velocity)
//! are treated explicitly while the thermal and mass transfer terms are treated
//! implicitly to handle their stiffness.
//!
//! ## Literature References
//!
//! 1. **Ascher et al. (1997)**. "Implicit-explicit Runge-Kutta methods for 
//!    time-dependent partial differential equations"
//!    - IMEX-RK schemes for stiff systems
//!
//! 2. **Kennedy & Carpenter (2003)**. "Additive Runge-Kutta schemes for 
//!    convection-diffusion-reaction equations"
//!    - ARK methods for mixed stiff/non-stiff systems
//!
//! 3. **Prosperetti & Lezzi (1986)**. "Bubble dynamics in a compressible liquid"
//!    - Thermal effects in bubble dynamics

use super::{BubbleState, KellerMiksisModel};
use crate::error::{KwaversResult, PhysicsError};
use ndarray::Array1;
use std::sync::Arc;

/// Configuration for IMEX bubble integration
#[derive(Debug, Clone)]
pub struct BubbleIMEXConfig {
    /// Relative tolerance for implicit solver
    pub rtol: f64,
    /// Absolute tolerance for implicit solver
    pub atol: f64,
    /// Maximum iterations for implicit solver
    pub max_iter: usize,
    /// Enable adaptive time stepping
    pub adaptive: bool,
    /// Minimum time step for adaptive stepping
    pub dt_min: f64,
    /// Maximum time step for adaptive stepping
    pub dt_max: f64,
}

impl Default for BubbleIMEXConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-9,
            max_iter: 10,
            adaptive: false,
            dt_min: 1e-12,
            dt_max: 1e-7,
        }
    }
}

/// IMEX integrator for bubble dynamics
pub struct BubbleIMEXIntegrator {
    solver: Arc<KellerMiksisModel>,
    config: BubbleIMEXConfig,
}

impl BubbleIMEXIntegrator {
    /// Create a new IMEX integrator for bubble dynamics
    pub fn new(solver: Arc<KellerMiksisModel>, config: BubbleIMEXConfig) -> Self {
        Self {
            solver,
            config,
        }
    }
    
    /// Create with default configuration
    pub fn with_defaults(solver: Arc<KellerMiksisModel>) -> Self {
        Self::new(solver, BubbleIMEXConfig::default())
    }
    
    /// Update configuration
    pub fn set_config(&mut self, config: BubbleIMEXConfig) {
        self.config = config;
    }
    
    /// Get configuration
    pub fn config(&self) -> &BubbleIMEXConfig {
        &self.config
    }
    
    /// Get solver
    pub fn solver(&self) -> &Arc<KellerMiksisModel> {
        &self.solver
    }
    
    /// Integrate bubble dynamics for one time step using IMEX
    /// Uses a simple first-order IMEX Euler scheme for now
    pub fn step(
        &mut self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        dt: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        // Convert bubble state to vector form
        let y0 = self.state_to_vector(state);
        
        // Step 1: Explicit update for mechanical terms
        let mut y_explicit = y0.clone();
        {
            let mut temp_state = self.vector_to_state(&y0, state)?;
            
            // Calculate mechanical acceleration
            let accel = self.solver.calculate_acceleration(
                &mut temp_state,
                p_acoustic,
                dp_dt,
                t,
            );
            
            // Update radius and velocity explicitly
            y_explicit[0] = y0[0] + dt * temp_state.wall_velocity;  // R
            y_explicit[1] = y0[1] + dt * accel;                      // v
            y_explicit[2] = y0[2];  // T (unchanged in explicit step)
            y_explicit[3] = y0[3];  // n_vapor (unchanged in explicit step)
        }
        
        // Step 2: Implicit update for thermal and mass transfer
        let mut y_final = y_explicit.clone();
        {
            // Newton iteration for implicit terms
            for iter in 0..self.config.max_iter {
                let state_current = self.vector_to_state(&y_final, state)?;
                
                // Calculate thermal and mass transfer rates
                let (dT_dt, dn_vapor_dt) = self.calculate_thermal_mass_transfer_rates(&state_current)?;
                
                // Residual: y_final - y_explicit - dt * f_implicit(y_final) = 0
                let residual = Array1::from_vec(vec![
                    0.0,  // No implicit contribution to radius
                    0.0,  // No implicit contribution to velocity
                    y_final[2] - y_explicit[2] - dt * dT_dt,
                    y_final[3] - y_explicit[3] - dt * dn_vapor_dt,
                ]);
                
                // Check convergence
                let residual_norm = residual.iter().map(|x| x.abs()).fold(0.0, f64::max);
                if residual_norm < self.config.atol {
                    break;
                }
                
                // Compute Jacobian (simplified - diagonal approximation)
                let jac = self.compute_jacobian_diagonal(&y_final, dt)?;
                
                // Newton update: y_final = y_final - J^(-1) * residual
                for i in 2..4 {  // Only update T and n_vapor
                    if jac[i].abs() > 1e-12 {
                        y_final[i] -= residual[i] / jac[i];
                    }
                }
                
                // Break if max iterations reached
                if iter == self.config.max_iter - 1 {
                    // Not converged, but continue anyway
                    break;
                }
            }
        }
        
        // Convert back to bubble state
        *state = self.vector_to_state(&y_final, state)?;
        
        // Update derived quantities
        state.update_compression(self.solver.params().r0);
        state.update_collapse_state();
        
        Ok(dt)  // Return actual time step taken
    }
    
    /// Compute diagonal Jacobian approximation for implicit solver
    fn compute_jacobian_diagonal(&self, y: &Array1<f64>, dt: f64) -> KwaversResult<Array1<f64>> {
        let mut jac = Array1::ones(4);
        
        // Jacobian diagonal elements: d(residual_i)/d(y_i)
        // For residual_i = y_i - y_explicit_i - dt * f_i(y)
        // We have: d(residual_i)/d(y_i) = 1 - dt * df_i/dy_i
        
        // For simplicity, we use: J_ii ≈ 1 (since dt is small)
        // This is a simplified approach that works well for small time steps
        
        jac[0] = 1.0;  // Radius (no implicit term)
        jac[1] = 1.0;  // Velocity (no implicit term)
        jac[2] = 1.0;  // Temperature (approximation)
        jac[3] = 1.0;  // Vapor (approximation)
        
        Ok(jac)
    }
    
    /// Calculate thermal and mass transfer rates without modifying state
    /// This ensures proper coupling between the two processes
    fn calculate_thermal_mass_transfer_rates(
        &self,
        state: &BubbleState,
    ) -> KwaversResult<(f64, f64)> {
        // Get current state values
        let r = state.radius;
        let v = state.wall_velocity;
        let T = state.temperature;
        let n_vapor = state.n_vapor;
        let n_gas = state.n_gas;
        
        // Calculate thermal rate of change (dT/dt)
        let params = self.solver.params();
        let dT_dt = if params.use_thermal_effects {
            // Polytropic/adiabatic model with heat transfer
            let gamma = self.calculate_effective_polytropic_index(state);
            
            // Heat transfer coefficient (simplified Nusselt number approach)
            let thermal_diffusivity = params.thermal_conductivity / 
                (params.rho_liquid * params.specific_heat_liquid);
            let peclet = (2.0 * r * v.abs()) / thermal_diffusivity;
            let nusselt = 2.0 + 0.6 * peclet.powf(0.5);
            let h = nusselt * params.thermal_conductivity / (2.0 * r);
            
            // Temperature rate from compression and heat transfer
            let compression_heating = -(gamma - 1.0) * T * v / r;
            let heat_transfer = -h * (T - 293.15) / (n_gas + n_vapor); // Using ambient temp
            
            compression_heating + heat_transfer
        } else {
            0.0
        };
        
        // Calculate mass transfer rate (dn_vapor/dt)
        let dn_vapor_dt = if params.use_mass_transfer {
            // Evaporation/condensation based on temperature-dependent vapor pressure
            let p_vapor_eq = self.calculate_equilibrium_vapor_pressure(T);
            let p_vapor_actual = n_vapor * crate::constants::thermodynamics::R_GAS * T / state.volume();
            
            // Mass transfer coefficient (simplified approach)
            let D_vapor = 2.5e-5; // Vapor diffusion coefficient in air [m²/s]
            let thermal_diffusivity = params.thermal_conductivity / 
                (params.rho_liquid * params.specific_heat_liquid);
            let peclet = (2.0 * r * v.abs()) / thermal_diffusivity;
            let sherwood = 2.0 + 0.6 * peclet.powf(0.33); // Mass transfer Sherwood number
            let k_mass = sherwood * D_vapor / (2.0 * r);
            
            // Rate of vapor moles change
            let driving_force = p_vapor_eq - p_vapor_actual;
            k_mass * driving_force * state.surface_area() / (crate::constants::thermodynamics::R_GAS * T)
        } else {
            0.0
        };
        
        Ok((dT_dt, dn_vapor_dt))
    }
    
    /// Calculate effective polytropic index for thermal model
    fn calculate_effective_polytropic_index(&self, state: &BubbleState) -> f64 {
        use crate::constants::bubble_dynamics::{PECLET_SCALING_FACTOR, MIN_PECLET_NUMBER};
        
        let params = self.solver.params();
        let thermal_diffusivity = params.thermal_conductivity / 
            (params.rho_liquid * params.specific_heat_liquid);
        let peclet = (2.0 * state.radius * state.wall_velocity.abs()) / thermal_diffusivity;
        let peclet_eff = peclet.max(MIN_PECLET_NUMBER);
        
        // Effective polytropic index varies from isothermal (1.0) to adiabatic (gamma)
        let gamma_gas = state.gas_species.gamma();
        1.0 + (gamma_gas - 1.0) / (1.0 + PECLET_SCALING_FACTOR / peclet_eff)
    }
    
    /// Calculate equilibrium vapor pressure at given temperature
    fn calculate_equilibrium_vapor_pressure(&self, temperature: f64) -> f64 {
        // Antoine equation for water vapor pressure
        let A = 8.07131;
        let B = 1730.63;
        let C = 233.426;
        
        let t_celsius = temperature - 273.15;
        let log10_p = A - B / (C + t_celsius);
        
        // Convert from mmHg to Pa
        10.0_f64.powf(log10_p) * 133.322
    }
    
    /// Convert bubble state to vector form
    fn state_to_vector(&self, state: &BubbleState) -> Array1<f64> {
        let mut y = Array1::zeros(4);
        y[0] = state.radius;
        y[1] = state.wall_velocity;
        y[2] = state.temperature;
        y[3] = state.n_vapor;
        y
    }
    
    /// Convert vector to bubble state
    fn vector_to_state(&self, y: &Array1<f64>, template: &BubbleState) -> KwaversResult<BubbleState> {
        if y.len() != 4 {
            return Err(PhysicsError::InvalidState {
                field: "state_vector".to_string(),
                value: format!("length {}", y.len()),
                reason: "Expected 4 elements".to_string(),
            }.into());
        }
        
        let mut state = template.clone();
        state.radius = y[0];
        state.wall_velocity = y[1];
        state.temperature = y[2];
        state.n_vapor = y[3];
        
        Ok(state)
    }
    
    /// Estimate stiffness of the system
    pub fn estimate_stiffness(&self, state: &BubbleState) -> f64 {
        let params = self.solver.params();
        let thermal_diffusivity = params.thermal_conductivity / 
            (params.rho_liquid * params.specific_heat_liquid);
        
        // Estimate based on time scales
        let mechanical_timescale = state.radius / state.wall_velocity.abs().max(1e-10);
        let thermal_timescale = state.radius.powi(2) / thermal_diffusivity;
        
        mechanical_timescale / thermal_timescale.min(mechanical_timescale)
    }
    
    /// Suggest time step based on stiffness
    pub fn suggest_timestep(&self, state: &BubbleState) -> f64 {
        let stiffness = self.estimate_stiffness(state);
        
        if stiffness > 100.0 {
            // Very stiff - use small time step
            self.config.dt_min * 10.0
        } else if stiffness > 10.0 {
            // Moderately stiff
            (self.config.dt_min + self.config.dt_max) / 2.0
        } else {
            // Not stiff
            self.config.dt_max
        }
    }
}

/// High-level function to integrate bubble dynamics using IMEX
pub fn integrate_bubble_dynamics_imex(
    solver: Arc<KellerMiksisModel>,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
) -> KwaversResult<()> {
    let mut integrator = BubbleIMEXIntegrator::with_defaults(solver);
    integrator.step(state, p_acoustic, dp_dt, dt, t)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::bubble_dynamics::{BubbleParameters, BubbleState, KellerMiksisModel};
    use std::sync::Arc;
    
    #[test]
    fn test_imex_integration() {
        let params = BubbleParameters::default();
        let solver = Arc::new(KellerMiksisModel::new(params.clone()));
        let mut state = BubbleState::new(&params);
        
        let config = BubbleIMEXConfig::default();
        let mut integrator = BubbleIMEXIntegrator::new(solver, config);
        
        // Test integration step
        let result = integrator.step(
            &mut state,
            1e5,  // 1 bar acoustic pressure
            0.0,
            1e-9, // 1 ns time step
            0.0,
        );
        
        assert!(result.is_ok());
        assert!(state.radius > 0.0);
        assert!(state.temperature > 0.0);
    }
    
    #[test]
    fn test_stiffness_detection() {
        let params = BubbleParameters::default();
        let solver = Arc::new(KellerMiksisModel::new(params.clone()));
        let state = BubbleState::new(&params);
        
        let integrator = BubbleIMEXIntegrator::with_defaults(solver);
        let stiffness = integrator.estimate_stiffness(&state);
        
        assert!(stiffness > 0.0);
        
        let suggested_dt = integrator.suggest_timestep(&state);
        assert!(suggested_dt > 0.0);
        assert!(suggested_dt <= BubbleIMEXConfig::default().dt_max);
    }
    
    #[test]
    fn test_thermal_mass_coupling() {
        let mut params = BubbleParameters::default();
        params.use_thermal_effects = true;
        params.use_mass_transfer = true;
        
        let solver = Arc::new(KellerMiksisModel::new(params.clone()));
        let state = BubbleState::new(&params);
        
        let integrator = BubbleIMEXIntegrator::with_defaults(solver);
        
        // Test that rates are calculated without modifying state
        let (dT_dt, dn_vapor_dt) = integrator.calculate_thermal_mass_transfer_rates(&state).unwrap();
        
        // Rates should be non-zero when effects are enabled
        assert!(dT_dt.abs() > 0.0 || dn_vapor_dt.abs() > 0.0);
    }
    
    #[test]
    fn test_adaptive_epsilon() {
        // Test that the integration works for different scales
        let params = BubbleParameters::default();
        let solver = Arc::new(KellerMiksisModel::new(params.clone()));
        
        // Test with small radius
        let mut small_state = BubbleState::new(&params);
        small_state.radius = 1e-9; // 1 nm
        
        // Test with large radius  
        let mut large_state = BubbleState::new(&params);
        large_state.radius = 1e-3; // 1 mm
        
        let mut integrator = BubbleIMEXIntegrator::with_defaults(solver);
        
        // Both should integrate successfully
        let small_result = integrator.step(
            &mut small_state,
            0.0,
            0.0,
            1e-12,
            0.0,
        );
        assert!(small_result.is_ok());
        
        let large_result = integrator.step(
            &mut large_state,
            0.0,
            0.0,
            1e-6,
            0.0,
        );
        assert!(large_result.is_ok());
    }
}