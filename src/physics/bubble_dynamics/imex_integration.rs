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

use super::{BubbleState, BubbleParameters, KellerMiksisModel};
use crate::error::{KwaversResult, PhysicsError};
use crate::solver::imex::{IMEXRK, IMEXScheme};
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Configuration for IMEX bubble integration
#[derive(Debug, Clone)]
pub struct BubbleIMEXConfig {
    /// IMEX scheme to use (e.g., ARS222, ARS343)
    pub scheme: IMEXScheme,
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
            scheme: IMEXScheme::ARS222,  // 2nd order, L-stable
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
    imex: IMEXRK,
}

impl BubbleIMEXIntegrator {
    /// Create a new IMEX integrator for bubble dynamics
    pub fn new(solver: Arc<KellerMiksisModel>, config: BubbleIMEXConfig) -> Self {
        let imex = IMEXRK::new(
            config.scheme.clone(),
            config.rtol,
            config.atol,
            config.max_iter,
        );
        
        Self {
            solver,
            config,
            imex,
        }
    }
    
    /// Create with default configuration
    pub fn with_defaults(solver: Arc<KellerMiksisModel>) -> Self {
        Self::new(solver, BubbleIMEXConfig::default())
    }
    
    /// Update configuration
    pub fn set_config(&mut self, config: BubbleIMEXConfig) {
        self.config = config;
        self.imex = IMEXRK::new(
            config.scheme.clone(),
            config.rtol,
            config.atol,
            config.max_iter,
        );
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
    pub fn step(
        &mut self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        dt: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        // Convert bubble state to vector form
        let mut y = self.state_to_vector(state);
        
        // Define explicit RHS (mechanical terms)
        let explicit_rhs = |y_vec: &Array1<f64>, time: f64| -> KwaversResult<Array1<f64>> {
            let mut temp_state = self.vector_to_state(y_vec, &state)?;
            
            // Calculate mechanical acceleration (explicit)
            let accel = self.solver.calculate_acceleration(
                &mut temp_state,
                p_acoustic,
                dp_dt,
                time,
            );
            
            // Build RHS for mechanical terms only
            let mut rhs = Array1::zeros(4);
            rhs[0] = temp_state.wall_velocity;  // dR/dt = v
            rhs[1] = accel;                      // dv/dt = acceleration
            // Temperature and vapor content derivatives are handled implicitly
            rhs[2] = 0.0;
            rhs[3] = 0.0;
            
            Ok(rhs)
        };
        
        // Define implicit RHS (thermal and mass transfer terms) - FIXED VERSION
        let implicit_rhs = |y_vec: &Array1<f64>, _time: f64| -> KwaversResult<Array1<f64>> {
            let temp_state = self.vector_to_state(y_vec, &state)?;
            
            // Calculate rates of change without modifying state
            // This ensures proper coupling between thermal and mass transfer
            let (dT_dt, dn_vapor_dt) = self.calculate_thermal_mass_transfer_rates(&temp_state)?;
            
            // Build RHS for thermal/mass transfer terms
            let mut rhs = Array1::zeros(4);
            rhs[0] = 0.0;  // No contribution to radius
            rhs[1] = 0.0;  // No direct contribution to velocity
            rhs[2] = dT_dt;
            rhs[3] = dn_vapor_dt;
            
            Ok(rhs)
        };
        
        // Define Jacobian for implicit solver with ADAPTIVE EPSILON
        let jacobian = |y_vec: &Array1<f64>, time: f64| -> KwaversResult<Array2<f64>> {
            let n = y_vec.len();
            let mut jac = Array2::zeros((n, n));
            
            // Base epsilon for finite differences
            let sqrt_eps = f64::EPSILON.sqrt(); // ~1.5e-8 for f64
            
            let f0 = implicit_rhs(y_vec, time)?;
            
            for j in 0..n {
                let mut y_perturbed = y_vec.clone();
                
                // Adaptive step size based on the magnitude of the state variable
                // This ensures good numerical accuracy for both small and large values
                let scale = y_vec[j].abs().max(1.0);
                let h = sqrt_eps * scale;
                
                // Use central differences for better accuracy when possible
                let use_central = true; // Could make this configurable
                
                if use_central && y_vec[j] - h > 0.0 {
                    // Central difference: (f(x+h) - f(x-h)) / (2h)
                    y_perturbed[j] = y_vec[j] + h;
                    let f_plus = implicit_rhs(&y_perturbed, time)?;
                    
                    y_perturbed[j] = y_vec[j] - h;
                    let f_minus = implicit_rhs(&y_perturbed, time)?;
                    
                    for i in 0..n {
                        jac[[i, j]] = (f_plus[i] - f_minus[i]) / (2.0 * h);
                    }
                } else {
                    // Forward difference: (f(x+h) - f(x)) / h
                    y_perturbed[j] = y_vec[j] + h;
                    let f_perturbed = implicit_rhs(&y_perturbed, time)?;
                    
                    for i in 0..n {
                        jac[[i, j]] = (f_perturbed[i] - f0[i]) / h;
                    }
                }
            }
            
            Ok(jac)
        };
        
        // Perform IMEX time step
        let dt_actual = self.imex.step(
            &mut y,
            dt,
            t,
            &explicit_rhs,
            &implicit_rhs,
            &jacobian,
        )?;
        
        // Convert back to bubble state
        *state = self.vector_to_state(&y, state)?;
        
        // Update derived quantities
        state.update_compression(state.params.r0);
        state.update_collapse_state();
        
        Ok(dt_actual)
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
        let dT_dt = if state.params.use_thermal_effects {
            // Polytropic/adiabatic model with heat transfer
            let gamma = self.calculate_effective_polytropic_index(state);
            
            // Heat transfer coefficient (simplified Nusselt number approach)
            let thermal_diffusivity = state.params.thermal_diffusivity;
            let peclet = (2.0 * r * v.abs()) / thermal_diffusivity;
            let nusselt = 2.0 + 0.6 * peclet.powf(0.5);
            let h = nusselt * state.params.k_thermal / (2.0 * r);
            
            // Temperature rate from compression and heat transfer
            let compression_heating = -(gamma - 1.0) * T * v / r;
            let heat_transfer = -h * (T - state.params.t_ambient) / (n_gas + n_vapor);
            
            compression_heating + heat_transfer
        } else {
            0.0
        };
        
        // Calculate mass transfer rate (dn_vapor/dt)
        let dn_vapor_dt = if state.params.use_mass_transfer {
            // Evaporation/condensation based on temperature-dependent vapor pressure
            let p_vapor_eq = self.calculate_equilibrium_vapor_pressure(T);
            let p_vapor_actual = n_vapor * crate::constants::thermodynamics::R_GAS * T / state.volume();
            
            // Mass transfer coefficient (simplified approach)
            let D_vapor = 2.5e-5; // Vapor diffusion coefficient in air [m²/s]
            let thermal_diffusivity = state.params.thermal_diffusivity;
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
        
        let peclet = (2.0 * state.radius * state.wall_velocity.abs()) / state.params.thermal_diffusivity;
        let peclet_eff = peclet.max(MIN_PECLET_NUMBER);
        
        // Effective polytropic index varies from isothermal (1.0) to adiabatic (gamma)
        let gamma_gas = state.params.gamma;
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
        // Estimate based on time scales
        let mechanical_timescale = state.radius / state.wall_velocity.abs().max(1e-10);
        let thermal_timescale = state.radius.powi(2) / state.params.thermal_diffusivity;
        
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
        // Test that adaptive epsilon works for different scales
        let params = BubbleParameters::default();
        let solver = Arc::new(KellerMiksisModel::new(params.clone()));
        
        // Test with small radius
        let mut small_state = BubbleState::new(&params);
        small_state.radius = 1e-9; // 1 nm
        
        // Test with large radius  
        let mut large_state = BubbleState::new(&params);
        large_state.radius = 1e-3; // 1 mm
        
        let integrator = BubbleIMEXIntegrator::with_defaults(solver);
        
        // Both should integrate successfully with adaptive epsilon
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