//! Adaptive time-stepping for bubble dynamics
//!
//! This module provides adaptive integration methods specifically designed
//! for the stiff ODEs in bubble dynamics, particularly during violent collapse.
//!
//! ## Literature References
//!
//! 1. **Hairer & Wanner (1996)**. "Solving Ordinary Differential Equations II: 
//!    Stiff and Differential-Algebraic Problems"
//!    - Adaptive time-stepping strategies for stiff ODEs
//!
//! 2. **Storey & Szeri (2000)**. "Water vapour, sonoluminescence and sonochemistry"
//!    - Time scales in bubble dynamics
//!
//! 3. **Lauterborn & Kurz (2010)**. "Physics of bubble oscillations"
//!    - Numerical challenges in bubble dynamics

use super::{BubbleState, BubbleParameters, KellerMiksisModel};
use crate::error::{KwaversResult, PhysicsError};
use crate::constants::bubble_dynamics::{MIN_RADIUS, MAX_RADIUS};
use crate::constants::adaptive_integration::*;
use std::sync::{Arc, Mutex};

/// Configuration for adaptive bubble integration
#[derive(Debug, Clone)]
pub struct AdaptiveBubbleConfig {
    /// Maximum time step (limited by acoustic period)
    pub dt_max: f64,
    /// Minimum time step (for extreme collapse)
    pub dt_min: f64,
    /// Relative tolerance for error control
    pub rtol: f64,
    /// Absolute tolerance for error control
    pub atol: f64,
    /// Safety factor for time step adjustment (0.8-0.95 typical)
    pub safety_factor: f64,
    /// Maximum factor to increase time step
    pub dt_increase_max: f64,
    /// Maximum factor to decrease time step
    pub dt_decrease_max: f64,
    /// Maximum number of sub-steps per main time step
    pub max_substeps: usize,
    /// Enable stability monitoring
    pub monitor_stability: bool,
}

impl Default for AdaptiveBubbleConfig {
    fn default() -> Self {
        Self {
            dt_max: 1e-7,        // 100 ns (limited by acoustic frequency)
            dt_min: 1e-12,       // 1 ps (for extreme collapse)
            rtol: 1e-6,
            atol: 1e-9,
            safety_factor: 0.9,
            dt_increase_max: 1.5,
            dt_decrease_max: 0.1,
            max_substeps: 1000,
            monitor_stability: true,
        }
    }
}

/// Adaptive integrator for bubble dynamics with sub-cycling
pub struct AdaptiveBubbleIntegrator {
    solver: Arc<Mutex<KellerMiksisModel>>,
    config: AdaptiveBubbleConfig,
    /// Current adaptive time step
    dt_adaptive: f64,
    /// Statistics
    total_substeps: usize,
    rejected_steps: usize,
    min_dt_used: f64,
    max_dt_used: f64,
}

impl AdaptiveBubbleIntegrator {
    /// Create new adaptive integrator
    pub fn new(solver: Arc<Mutex<KellerMiksisModel>>, config: AdaptiveBubbleConfig) -> Self {
        let dt_adaptive = config.dt_max * 0.1; // Start conservatively
        
        Self {
            solver,
            config,
            dt_adaptive,
            total_substeps: 0,
            rejected_steps: 0,
            min_dt_used: f64::MAX,
            max_dt_used: 0.0,
        }
    }
    
    /// Integrate bubble dynamics with adaptive sub-cycling
    /// 
    /// This method takes multiple smaller time steps internally to resolve
    /// the stiff dynamics while advancing by the main simulation dt.
    pub fn integrate_adaptive(
        &mut self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        dt_main: f64,
        t_start: f64,
    ) -> KwaversResult<()> {
        let mut t = t_start;
        let t_end = t_start + dt_main;
        let mut substeps = 0;
        
        // Sub-cycle until we reach the target time
        while t < t_end && substeps < self.config.max_substeps {
            // Limit time step to not exceed target
            let dt = self.dt_adaptive.min(t_end - t);
            
            // Try a step with current dt
            let (success, dt_new) = self.try_step(state, p_acoustic, dp_dt, dt, t)?;
            
            if success {
                // Step accepted, advance time
                t += dt;
                substeps += 1;
                self.total_substeps += 1;
                
                // Update statistics
                self.min_dt_used = self.min_dt_used.min(dt);
                self.max_dt_used = self.max_dt_used.max(dt);
            } else {
                // Step rejected, will retry with smaller dt
                self.rejected_steps += 1;
            }
            
            // Update adaptive time step
            self.dt_adaptive = dt_new.max(self.config.dt_min).min(self.config.dt_max);
        }
        
        if substeps >= self.config.max_substeps {
            return Err(PhysicsError::ConvergenceFailure {
                solver: "AdaptiveBubbleIntegrator".to_string(),
                iterations: substeps,
                residual: (t_end - t).abs(),
            }.into());
        }
        
        Ok(())
    }
    
    /// Try a single time step with error estimation
    fn try_step(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        dt: f64,
        t: f64,
    ) -> KwaversResult<(bool, f64)> {
        // Store original state
        let state_orig = state.clone();
        
        // Take one full step
        let mut state_full = state_orig.clone();
        self.step_rk4(&mut state_full, p_acoustic, dp_dt, dt, t)?;
        
        // Take two half steps for error estimation
        let mut state_half = state_orig.clone();
        let dt_half = dt * 0.5;
        self.step_rk4(&mut state_half, p_acoustic, dp_dt, dt_half, t)?;
        self.step_rk4(&mut state_half, p_acoustic, dp_dt, dt_half, t + dt_half)?;
        
        // Estimate error (Richardson extrapolation)
        let error_r = (state_full.radius - state_half.radius).abs();
        let error_v = (state_full.wall_velocity - state_half.wall_velocity).abs();
        
        // Compute error norm
        let scale_r = self.config.atol + self.config.rtol * state_half.radius.abs();
        let scale_v = self.config.atol + self.config.rtol * state_half.wall_velocity.abs();
        
        let error_norm = ((error_r / scale_r).powi(2) + (error_v / scale_v).powi(2)).sqrt() / 2.0_f64.sqrt();
        
        // Compute new time step based on error
        let dt_new = if error_norm > 0.0 {
            let factor = self.config.safety_factor * (1.0 / error_norm).powf(0.2); // 4th order method
            let factor = factor.min(self.config.dt_increase_max).max(self.config.dt_decrease_max);
            dt * factor
        } else {
            dt * self.config.dt_increase_max
        };
        
        // Accept or reject step
        let accept = error_norm <= 1.0 && self.check_stability(&state_half);
        
        if accept {
            // Use the more accurate half-step result (5th order)
            *state = state_half;
        }
        
        Ok((accept, dt_new))
    }
    
    /// Perform a single RK4 step
    fn step_rk4(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        dp_dt: f64,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        let r0 = self.solver.lock().unwrap().params().r0;
        
        // RK4 integration
        let state0 = state.clone();
        
        // k1
        let k1_a = self.solver.lock().unwrap().calculate_acceleration(state, p_acoustic, dp_dt, t);
        let k1_v = state.wall_velocity;
        
        // k2
        state.radius = state0.radius + 0.5 * dt * k1_v;
        state.wall_velocity = state0.wall_velocity + 0.5 * dt * k1_a;
        let k2_a = self.solver.lock().unwrap().calculate_acceleration(state, p_acoustic, dp_dt, t + 0.5 * dt);
        let k2_v = state.wall_velocity;
        
        // k3
        state.radius = state0.radius + 0.5 * dt * k2_v;
        state.wall_velocity = state0.wall_velocity + 0.5 * dt * k2_a;
        let k3_a = self.solver.lock().unwrap().calculate_acceleration(state, p_acoustic, dp_dt, t + 0.5 * dt);
        let k3_v = state.wall_velocity;
        
        // k4
        state.radius = state0.radius + dt * k3_v;
        state.wall_velocity = state0.wall_velocity + dt * k3_a;
        let k4_a = self.solver.lock().unwrap().calculate_acceleration(state, p_acoustic, dp_dt, t + dt);
        let k4_v = state.wall_velocity;
        
        // Combine
        state.radius = state0.radius + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        state.wall_velocity = state0.wall_velocity + (dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a);
        
        // Update derived quantities
        state.update_compression(r0);
        state.update_collapse_state();
        
        // Update temperature and mass transfer with smaller time step
        {
            let solver = self.solver.lock().unwrap();
            solver.update_temperature(state, dt);
        }
        {
            let mut solver = self.solver.lock().unwrap();
            solver.update_mass_transfer(state, dt);
        }
        
        Ok(())
    }
    
    /// Check if the state is physically stable
    fn check_stability(&self, state: &BubbleState) -> bool {
        if !self.config.monitor_stability {
            return true;
        }
        
        // Check for NaN or Inf
        if !state.radius.is_finite() || !state.wall_velocity.is_finite() {
            return false;
        }
        
        // Check physical bounds (but don't clamp!)
        if state.radius < MIN_RADIUS * MIN_RADIUS_SAFETY_FACTOR || 
           state.radius > MAX_RADIUS * MAX_RADIUS_SAFETY_FACTOR {
            return false;
        }
        
        // Check for extreme velocities (approaching speed of sound)
        if state.wall_velocity.abs() > self.solver.lock().unwrap().params().c_liquid * MAX_VELOCITY_FRACTION {
            return false;
        }
        
        // Check temperature bounds
        if state.temperature < MIN_TEMPERATURE || state.temperature > MAX_TEMPERATURE {
            return false;
        }
        
        true
    }
    
    /// Get integration statistics
    pub fn statistics(&self) -> IntegrationStatistics {
        IntegrationStatistics {
            total_substeps: self.total_substeps,
            rejected_steps: self.rejected_steps,
            min_dt_used: self.min_dt_used,
            max_dt_used: self.max_dt_used,
            rejection_rate: self.rejected_steps as f64 / self.total_substeps.max(1) as f64,
        }
    }
    
    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.total_substeps = 0;
        self.rejected_steps = 0;
        self.min_dt_used = f64::MAX;
        self.max_dt_used = 0.0;
    }
}

/// Integration statistics for monitoring
#[derive(Debug, Clone)]
pub struct IntegrationStatistics {
    pub total_substeps: usize,
    pub rejected_steps: usize,
    pub min_dt_used: f64,
    pub max_dt_used: f64,
    pub rejection_rate: f64,
}

/// Replace the old fixed-timestep integration with adaptive version
pub fn integrate_bubble_dynamics_adaptive(
    solver: Arc<Mutex<KellerMiksisModel>>,
    state: &mut BubbleState,
    p_acoustic: f64,
    dp_dt: f64,
    dt: f64,
    t: f64,
) -> KwaversResult<()> {
    let config = AdaptiveBubbleConfig::default();
    let mut integrator = AdaptiveBubbleIntegrator::new(solver, config);
    integrator.integrate_adaptive(state, p_acoustic, dp_dt, dt, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adaptive_integration() {
        let params = BubbleParameters::default();
        let solver = Arc::new(Mutex::new(KellerMiksisModel::new(params.clone())));
        let mut state = BubbleState::new(&params);
        
        let config = AdaptiveBubbleConfig::default();
        let mut integrator = AdaptiveBubbleIntegrator::new(solver.clone(), config);
        
        // Test integration with acoustic forcing
        let result = integrator.integrate_adaptive(
            &mut state,
            1e5,  // 1 bar acoustic pressure
            0.0,
            1e-6, // 1 microsecond main time step
            0.0,
        );
        
        assert!(result.is_ok());
        assert!(state.radius > 0.0);
        
        // Check that sub-cycling occurred
        let stats = integrator.statistics();
        assert!(stats.total_substeps > 0);
        println!("Integration stats: {:?}", stats);
    }
    
    #[test]
    fn test_stability_check() {
        let params = BubbleParameters::default();
        let solver = Arc::new(Mutex::new(KellerMiksisModel::new(params.clone())));
        let config = AdaptiveBubbleConfig::default();
        let integrator = AdaptiveBubbleIntegrator::new(solver, config);
        
        // Test with stable state
        let mut state = BubbleState::new(&params);
        assert!(integrator.check_stability(&state));
        
        // Test with NaN
        state.radius = f64::NAN;
        assert!(!integrator.check_stability(&state));
        
        // Test with extreme values
        state.radius = 1e10;
        assert!(!integrator.check_stability(&state));
    }
}