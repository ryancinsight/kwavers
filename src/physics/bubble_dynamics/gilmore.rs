//! Gilmore equation for high-amplitude bubble oscillations
//!
//! The Gilmore equation is more accurate than Keller-Miksis for violent
//! bubble collapse and high-amplitude oscillations where compressibility
//! effects are significant.
//!
//! Reference: Gilmore, F. R. (1952). "The growth or collapse of a spherical bubble
//! in a viscous compressible liquid". Hydrodynamics Laboratory Report 26-4,
//! California Institute of Technology.

use super::{BubbleParameters, BubbleState};
use crate::error::KwaversResult;
use crate::physics::constants_physics::*;

/// Gilmore equation solver for high-amplitude bubble dynamics
#[derive(Debug)]
pub struct GilmoreSolver {
    params: BubbleParameters,
    /// Tait equation parameters for water
    tait_b: f64,
    tait_n: f64,
    /// Reference enthalpy
    h_ref: f64,
}

impl GilmoreSolver {
    /// Create a new Gilmore solver
    pub fn new(params: BubbleParameters) -> Self {
        // Tait equation parameters for water (Fujikawa & Akamatsu, 1980)
        let tait_b = 3.046e8; // Pa
        let tait_n = 7.15;

        // Reference enthalpy at ambient pressure
        let h_ref = Self::calculate_enthalpy(params.p0, params.rho_liquid, tait_b, tait_n);

        Self {
            params,
            tait_b,
            tait_n,
            h_ref,
        }
    }

    /// Calculate enthalpy from Tait equation of state
    /// h = ∫(dp/ρ) = (n/(n-1)) * (p+B)/ρ₀ * [(p+B)/(p₀+B)]^((n-1)/n)
    fn calculate_enthalpy(pressure: f64, rho_0: f64, b: f64, n: f64) -> f64 {
        let p0 = ATMOSPHERIC_PRESSURE;
        (n / (n - 1.0)) * (pressure + b) / rho_0 * ((pressure + b) / (p0 + b)).powf((n - 1.0) / n)
    }

    /// Calculate sound speed from Tait equation
    /// c² = (∂p/∂ρ)_s = n(p+B)/ρ
    fn calculate_sound_speed(&self, pressure: f64) -> f64 {
        ((self.tait_n * (pressure + self.tait_b)) / self.params.rho_liquid).sqrt()
    }

    /// Calculate liquid density from Tait equation
    /// ρ = ρ₀ * [(p+B)/(p₀+B)]^(1/n)
    fn calculate_density(&self, pressure: f64) -> f64 {
        let p0 = self.params.p0;
        self.params.rho_liquid
            * ((pressure + self.tait_b) / (p0 + self.tait_b)).powf(1.0 / self.tait_n)
    }

    /// Calculate bubble wall acceleration using Gilmore equation
    ///
    /// The Gilmore equation in standard form:
    /// (1 - u/C) * R * R_ddot + (3/2) * (1 - u/(3C)) * u² =
    ///     (1 + u/C) * H + (1 - u/C) * R/C * dH/dt
    ///
    /// where:
    /// - u = dR/dt is the bubble wall velocity
    /// - C is the sound speed in liquid at the bubble wall
    /// - H is the enthalpy difference between bubble wall and infinity
    pub fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        let r = state.radius;
        let u = state.wall_velocity;

        // Acoustic forcing
        let omega = 2.0 * std::f64::consts::PI * self.params.driving_frequency;
        let p_acoustic_inst = p_acoustic * (omega * t).sin();
        let p_inf = self.params.p0 + p_acoustic_inst;

        // Internal bubble pressure (simplified - should include thermal effects)
        let gamma = state.gas_species.gamma();
        let p_gas = self.params.p0 * (self.params.r0 / r).powf(3.0 * gamma);

        // Pressure at bubble wall (liquid side)
        let p_wall = p_gas - 2.0 * self.params.sigma / r - 4.0 * self.params.mu_liquid * u / r;

        // Enthalpy at bubble wall and infinity
        let h_wall =
            Self::calculate_enthalpy(p_wall, self.params.rho_liquid, self.tait_b, self.tait_n);
        let h_inf =
            Self::calculate_enthalpy(p_inf, self.params.rho_liquid, self.tait_b, self.tait_n);
        let h = h_wall - h_inf;

        // Sound speed at bubble wall
        let c_wall = self.calculate_sound_speed(p_wall);

        // Time derivative of enthalpy (simplified - assumes quasi-static)
        let dh_dt = self.estimate_enthalpy_derivative(state, p_wall, p_acoustic, omega, t);

        // Gilmore equation solved for R_ddot
        let u_c = u / c_wall;

        // Avoid singularity when u approaches c
        if u_c.abs() > 0.99 {
            return Err(crate::error::PhysicsError::NumericalInstability {
                timestep: 0.0,
                cfl_limit: u_c.abs(),
            }
            .into());
        }

        let lhs_coeff = r * (1.0 - u_c);
        let rhs = (1.0 + u_c) * h + (1.0 - u_c) * r / c_wall * dh_dt;
        let nonlinear_term = 1.5 * (1.0 - u_c / 3.0) * u * u;

        let acceleration = (rhs - nonlinear_term) / lhs_coeff;

        Ok(acceleration)
    }

    /// Estimate time derivative of enthalpy
    fn estimate_enthalpy_derivative(
        &self,
        state: &BubbleState,
        p_wall: f64,
        p_acoustic: f64,
        omega: f64,
        t: f64,
    ) -> f64 {
        // Simplified estimate based on acoustic forcing rate
        let dp_inf_dt = p_acoustic * omega * (omega * t).cos();

        // Chain rule: dH/dt = (∂H/∂p) * dp/dt
        // For Tait equation, this involves complex derivatives
        // Simplified approximation:
        let c_wall = self.calculate_sound_speed(p_wall);
        dp_inf_dt / (self.params.rho_liquid * c_wall)
    }

    /// Check if conditions warrant using Gilmore over simpler models
    pub fn should_use_gilmore(&self, state: &BubbleState) -> bool {
        // Use Gilmore when:
        // 1. Wall Mach number > 0.1
        // 2. Pressure ratio > 10
        // 3. Radius compression > 10x

        let mach = state.wall_velocity.abs() / self.params.c_liquid;
        let pressure_ratio = state.pressure_internal / self.params.p0;
        let radius_ratio = self.params.r0 / state.radius;

        mach > 0.1 || pressure_ratio > 10.0 || radius_ratio > 10.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gilmore_initialization() {
        let params = BubbleParameters::default();
        let solver = GilmoreSolver::new(params);

        assert!(solver.tait_b > 0.0);
        assert!(solver.tait_n > 1.0);
        assert!(solver.h_ref > 0.0);
    }

    #[test]
    fn test_sound_speed_calculation() {
        let params = BubbleParameters::default();
        let solver = GilmoreSolver::new(params);

        // At atmospheric pressure, sound speed should be close to 1500 m/s
        let c = solver.calculate_sound_speed(ATMOSPHERIC_PRESSURE);
        assert!(c > 1400.0 && c < 1600.0);

        // At higher pressure, sound speed increases
        let c_high = solver.calculate_sound_speed(10.0 * ATMOSPHERIC_PRESSURE);
        assert!(c_high > c);
    }

    #[test]
    fn test_gilmore_vs_keller_miksis_threshold() {
        let params = BubbleParameters::default();
        let solver = GilmoreSolver::new(params.clone());
        let mut state = BubbleState::new(&params);

        // Low amplitude - shouldn't need Gilmore
        state.wall_velocity = 10.0; // m/s
        assert!(!solver.should_use_gilmore(&state));

        // High amplitude - should use Gilmore
        state.wall_velocity = 200.0; // m/s (Mach > 0.1)
        assert!(solver.should_use_gilmore(&state));
    }
}
