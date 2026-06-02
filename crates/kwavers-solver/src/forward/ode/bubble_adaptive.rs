// Adaptive Runge-Kutta integration for bubble dynamics ODEs
//
// Provides RK4(5) with Richardson extrapolation error control for the stiff
// Keller-Miksis equation. Time step adaptation follows Hairer & Wanner (1996)
// with stability monitoring for violent collapse regimes.
//
// ## Mathematical Foundation
//
// The Keller-Miksis equation is a second-order nonlinear ODE of the form:
//
// ```text
// R̈ = f(R, Ṙ, p_∞, c, ρ, ...)
// ```
//
// which is converted to a first-order system:
//
// ```text
// dy/dt = f(t, y),  y = [R, Ṙ, T, n_v]
// ```
//
// and integrated with adaptive RK4 using local error estimation via
// Richardson extrapolation (full step vs. two half steps).
//
// ## Stability Criterion
//
// The adaptive step size is constrained by:
// - `dt_min ≤ dt ≤ dt_max` (numerical and physical bounds)
// - CFL-like limit: `|Ṙ| < c_liquid * 0.3`
// - Physical bounds: `R_min < R < R_max`

// Re-export from physics layer (canonical location established via DIP)
pub use kwavers_physics::acoustics::bubble_dynamics::adaptive_integration::{
    integrate_bubble_dynamics_adaptive, AdaptiveBubbleConfig, AdaptiveBubbleIntegrator,
    IntegrationStatistics,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = AdaptiveBubbleConfig::default();
        assert!(config.dt_max > 0.0);
        assert!(config.dt_min > 0.0);
        assert!(config.dt_min < config.dt_max);
        assert!(config.rtol > 0.0);
        assert!(config.safety_factor > 0.0 && config.safety_factor <= 1.0);
    }
}
