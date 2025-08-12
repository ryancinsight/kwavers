//! IMEX Integration for Bubble Dynamics
//!
//! This module provides an IMEX (Implicit-Explicit) time integration scheme
//! specifically tailored for bubble dynamics equations, which are known to be
//! extremely stiff during violent collapse phases.
//!
//! ## Literature References
//!
//! 1. **Prosperetti, A. (1991)**. "The thermal behaviour of oscillating gas bubbles."
//!    *Journal of Fluid Mechanics*, 222, 587-616.
//!    - Thermal damping in bubble oscillations
//!
//! 2. **Yasui, K. (1997)**. "Alternative model of single-bubble sonoluminescence."
//!    *Physical Review E*, 56(6), 6750.
//!    - Stiff thermal and chemical kinetics
//!
//! 3. **Storey, B. D., & Szeri, A. J. (2000)**. "Water vapour, sonoluminescence and
//!    sonochemistry." *Proceedings of the Royal Society A*, 456(1999), 1685-1709.
//!    - Mass transfer and phase change effects

use super::{BubbleState, BubbleParameters, KellerMiksisModel};
use crate::solver::imex::{IMEXRK, IMEXRKConfig, IMEXRKType, ImplicitSolver, NewtonSolver};
use crate::error::{KwaversResult, KwaversError};
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// IMEX integrator for bubble dynamics
pub struct BubbleIMEXIntegrator {
    /// Keller-Miksis model for mechanical dynamics
    solver: Arc<KellerMiksisModel>,
    /// IMEX-RK scheme
    imex_scheme: IMEXRK,
    /// Implicit solver for stiff terms
    implicit_solver: NewtonSolver,
    /// Configuration
    config: BubbleIMEXConfig,
}

/// Configuration for bubble IMEX integration
#[derive(Debug, Clone)]
pub struct BubbleIMEXConfig {
    /// IMEX-RK scheme type (default: ARK3)
    pub scheme_type: IMEXRKType,
    /// Tolerance for implicit solver
    pub implicit_tol: f64,
    /// Maximum Newton iterations
    pub max_newton_iter: usize,
    /// Enable adaptive time stepping
    pub adaptive: bool,
    /// Relative tolerance for adaptive stepping
    pub rtol: f64,
    /// Absolute tolerance for adaptive stepping
    pub atol: f64,
    /// Safety factor for time step adaptation
    pub safety_factor: f64,
}

impl Default for BubbleIMEXConfig {
    fn default() -> Self {
        Self {
            scheme_type: IMEXRKType::ARK3,  // 3rd order Additive Runge-Kutta
            implicit_tol: 1e-10,
            max_newton_iter: 10,
            adaptive: true,
            rtol: 1e-6,
            atol: 1e-9,
            safety_factor: 0.9,
        }
    }
}

impl BubbleIMEXIntegrator {
    /// Create new IMEX integrator for bubble dynamics
    pub fn new(
        solver: Arc<KellerMiksisModel>,
        config: BubbleIMEXConfig,
    ) -> KwaversResult<Self> {
        // Configure IMEX-RK scheme
        let imex_config = IMEXRKConfig {
            scheme_type: config.scheme_type.clone(),
            adaptive: config.adaptive,
            rtol: config.rtol,
            atol: config.atol,
            safety_factor: config.safety_factor,
        };
        
        let imex_scheme = IMEXRK::new(imex_config)?;
        
        // Configure Newton solver for implicit terms
        let implicit_solver = NewtonSolver::new(
            config.implicit_tol,
            config.max_newton_iter,
        );
        
        Ok(Self {
            solver,
            imex_scheme,
            implicit_solver,
            config,
        })
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
        
        // Define implicit RHS (thermal and mass transfer terms)
        let implicit_rhs = |y_vec: &Array1<f64>, time: f64| -> KwaversResult<Array1<f64>> {
            let mut temp_state = self.vector_to_state(y_vec, &state)?;
            
            // Store original values
            let temp_orig = temp_state.temperature;
            let vapor_orig = temp_state.n_vapor;
            
            // Update thermal effects (implicit)
            self.solver.update_temperature(&mut temp_state, dt);
            let dT_dt = (temp_state.temperature - temp_orig) / dt;
            
            // Update mass transfer (implicit)
            temp_state.temperature = temp_orig;  // Reset for mass transfer calc
            temp_state.n_vapor = vapor_orig;
            self.solver.update_mass_transfer(&mut temp_state, dt);
            let dn_vapor_dt = (temp_state.n_vapor - vapor_orig) / dt;
            
            // Build RHS for thermal/mass transfer terms
            let mut rhs = Array1::zeros(4);
            rhs[0] = 0.0;  // No contribution to radius
            rhs[1] = 0.0;  // No direct contribution to velocity
            rhs[2] = dT_dt;
            rhs[3] = dn_vapor_dt;
            
            Ok(rhs)
        };
        
        // Define Jacobian for implicit solver
        let jacobian = |y_vec: &Array1<f64>, time: f64| -> KwaversResult<Array2<f64>> {
            // Approximate Jacobian using finite differences
            let eps = 1e-8;
            let n = y_vec.len();
            let mut jac = Array2::zeros((n, n));
            
            let f0 = implicit_rhs(y_vec, time)?;
            
            for j in 0..n {
                let mut y_perturbed = y_vec.clone();
                y_perturbed[j] += eps;
                let f_perturbed = implicit_rhs(&y_perturbed, time)?;
                
                for i in 0..n {
                    jac[[i, j]] = (f_perturbed[i] - f0[i]) / eps;
                }
            }
            
            Ok(jac)
        };
        
        // Perform IMEX time step
        let dt_actual = self.imex_scheme.step(
            &mut y,
            dt,
            t,
            explicit_rhs,
            implicit_rhs,
            jacobian,
            &self.implicit_solver,
        )?;
        
        // Update state from vector
        *state = self.vector_to_state(&y, state)?;
        
        Ok(dt_actual)
    }
    
    /// Convert bubble state to vector representation
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
            return Err(KwaversError::NumericalError {
                message: format!("Invalid state vector size: expected 4, got {}", y.len()),
            });
        }
        
        let mut state = template.clone();
        state.radius = y[0];
        state.wall_velocity = y[1];
        state.temperature = y[2];
        state.n_vapor = y[3];
        
        // Update derived quantities
        state.update_compression(template.radius);
        state.update_max_temperature();
        
        Ok(state)
    }
    
    /// Estimate stiffness of the current state
    pub fn estimate_stiffness(&self, state: &BubbleState) -> f64 {
        // Stiffness increases dramatically during collapse
        let compression = state.radius / state.params.r0;
        let mach = state.mach_number;
        
        // Empirical stiffness indicator
        let thermal_stiffness = 1.0 / compression.max(0.01);
        let mechanical_stiffness = mach.max(0.1);
        
        thermal_stiffness * mechanical_stiffness
    }
    
    /// Adaptive time step suggestion based on stiffness
    pub fn suggest_timestep(&self, state: &BubbleState, dt_current: f64) -> f64 {
        let stiffness = self.estimate_stiffness(state);
        
        // Reduce time step for high stiffness
        if stiffness > 100.0 {
            dt_current * 0.1
        } else if stiffness > 10.0 {
            dt_current * 0.5
        } else {
            dt_current.min(1e-6)  // Maximum time step
        }
    }
}

/// High-level integration function using IMEX for bubble dynamics
pub fn integrate_bubble_dynamics_imex(
    solver: Arc<KellerMiksisModel>,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
) -> KwaversResult<f64> {
    // Create IMEX integrator with default config
    let config = BubbleIMEXConfig::default();
    let mut integrator = BubbleIMEXIntegrator::new(solver, config)?;
    
    // Perform integration step
    integrator.step(state, p_acoustic, dp_dt, dt, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_imex_integration() {
        let params = BubbleParameters::default();
        let solver = Arc::new(KellerMiksisModel::new(params.clone()));
        let mut state = BubbleState::new(&params);
        
        // Apply acoustic forcing
        let p_acoustic = 1e5;  // 1 bar
        let dp_dt = 0.0;
        let dt = 1e-9;  // 1 ns
        let t = 0.0;
        
        // Test IMEX integration
        let result = integrate_bubble_dynamics_imex(
            solver.clone(),
            &mut state,
            p_acoustic,
            dp_dt,
            dt,
            t,
        );
        
        assert!(result.is_ok());
        assert!(state.radius > 0.0);
        assert!(state.temperature > 0.0);
    }
    
    #[test]
    fn test_stiffness_detection() {
        let params = BubbleParameters::default();
        let solver = Arc::new(KellerMiksisModel::new(params.clone()));
        let config = BubbleIMEXConfig::default();
        let integrator = BubbleIMEXIntegrator::new(solver, config).unwrap();
        
        let mut state = BubbleState::new(&params);
        
        // Test stiffness during expansion
        state.radius = params.r0 * 2.0;
        let stiffness_expansion = integrator.estimate_stiffness(&state);
        
        // Test stiffness during collapse
        state.radius = params.r0 * 0.1;
        state.mach_number = 0.5;
        let stiffness_collapse = integrator.estimate_stiffness(&state);
        
        // Collapse should be much stiffer
        assert!(stiffness_collapse > stiffness_expansion * 10.0);
    }
}