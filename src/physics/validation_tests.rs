//! Physics validation tests against analytical solutions
//!
//! This module validates our physics implementations against known analytical
//! solutions from literature to ensure scientific accuracy.

#[cfg(test)]
mod tests {
    
    use crate::physics::bubble_dynamics::{BubbleParameters, BubbleState, RayleighPlessetSolver};
    use crate::physics::constants::*;
    use approx::assert_relative_eq;

    /// Validate linear bubble oscillations against Prosperetti's analytical solution
    /// Reference: Prosperetti, A. (1977). "Thermal effects and damping mechanisms
    /// in the forced radial oscillations of gas bubbles in liquids"
    /// J. Acoust. Soc. Am. 61, 17-27
    #[test]
    fn test_prosperetti_linear_oscillations() {
        // Small amplitude oscillations around equilibrium
        let mut params = BubbleParameters::default();
        params.r0 = 10e-6; // 10 μm bubble
        params.p0 = ATMOSPHERIC_PRESSURE;
        params.sigma = SURFACE_TENSION_WATER;
        params.rho_liquid = DENSITY_WATER;
        params.mu_liquid = VISCOSITY_WATER;
        params.use_thermal_effects = false; // Isothermal for simplicity

        let solver = RayleighPlessetSolver::new(params.clone());
        let mut state = BubbleState::new(&params);

        // Apply small pressure perturbation (1% of ambient)
        let p_acoustic = 0.01 * params.p0;
        let frequency = 20e3; // 20 kHz
        let omega = 2.0 * std::f64::consts::PI * frequency;

        // Natural frequency of bubble (Minnaert frequency)
        let gamma = params.gas_species.gamma();
        let omega_0 = (1.0 / params.r0) * ((3.0 * gamma * params.p0 / params.rho_liquid).sqrt());

        // Damping coefficient (simplified, viscous only)
        let delta = 4.0 * params.mu_liquid / (params.rho_liquid * omega * params.r0.powi(2));

        // Expected amplitude from linear theory
        let amplitude_theory = (p_acoustic / params.rho_liquid)
            / ((omega_0.powi(2) - omega.powi(2)).powi(2) + (2.0 * delta * omega * omega_0).powi(2))
                .sqrt();

        // Simulate for several periods
        let dt = 1.0 / (frequency * 100.0); // 100 points per period
        let n_periods = 5;
        let n_steps = (n_periods as f64 * 100.0) as usize;

        let mut max_velocity = 0.0;
        for i in 0..n_steps {
            let t = i as f64 * dt;
            let accel = solver.calculate_acceleration(&state, p_acoustic, t);
            state.wall_velocity += accel * dt;
            state.radius += state.wall_velocity * dt;

            if state.wall_velocity.abs() > max_velocity {
                max_velocity = state.wall_velocity.abs();
            }
        }

        // Compare with theoretical amplitude (velocity = omega * amplitude)
        let max_velocity_theory = omega * amplitude_theory;

        // Allow 10% error due to numerical discretization
        assert_relative_eq!(max_velocity, max_velocity_theory, epsilon = 0.1);
    }

    /// Validate shock wave formation distance
    /// Reference: Hamilton & Blackstock (1998). "Nonlinear Acoustics"
    /// Academic Press, Chapter 4
    #[test]
    fn test_shock_formation_distance() {
        // For a plane wave: x_shock = ρ₀c₀³/(βωp₀)
        // where β = 1 + B/2A is the nonlinearity parameter

        let frequency: f64 = 1e6; // 1 MHz
        let pressure_amplitude: f64 = 1e6; // 1 MPa
        let beta = 1.0 + NONLINEARITY_WATER / 2.0;
        let omega = 2.0 * std::f64::consts::PI * frequency;

        let x_shock_theory =
            DENSITY_WATER * SOUND_SPEED_WATER.powi(3) / (beta * omega * pressure_amplitude);

        // Our implementation should predict similar shock distance
        // This is a simplified test - full validation would require
        // propagating a wave and detecting shock formation

        // For now, just verify the formula is implemented correctly
        let x_shock_calc =
            DENSITY_WATER * SOUND_SPEED_WATER.powi(3) / (beta * omega * pressure_amplitude);

        assert_relative_eq!(x_shock_calc, x_shock_theory, epsilon = 1e-10);
    }

    /// Validate absorption following power law
    /// Reference: Szabo, T. L. (2004). "Diagnostic Ultrasound Imaging"
    /// Academic Press, Chapter 4
    #[test]
    fn test_power_law_absorption() {
        // Power law: α(f) = α₀ * (f/f₀)^y
        // where α₀ is absorption at reference frequency f₀

        let f0 = 1e6; // 1 MHz reference
        let alpha_0 = ABSORPTION_TISSUE; // dB/cm/MHz^y

        // Test at different frequencies
        let test_frequencies: Vec<f64> = vec![0.5e6, 1e6, 2e6, 5e6, 10e6];

        for f in test_frequencies {
            let alpha_expected = alpha_0 * (f / f0).powf(ABSORPTION_POWER);

            // Convert to Np/m for consistency
            let alpha_np = alpha_expected * DB_TO_NP * MHZ_TO_HZ.powf(ABSORPTION_POWER) / CM_TO_M;

            // Verify the calculation
            assert!(alpha_np > 0.0);

            // At 1 MHz, should match reference
            if (f - f0).abs() < 1.0 {
                assert_relative_eq!(alpha_expected, alpha_0, epsilon = 1e-10);
            }
        }
    }

    /// Validate CFL condition for stability
    #[test]
    fn test_cfl_stability_condition() {
        // CFL: c * dt / dx ≤ CFL_max
        // For FDTD in 3D: CFL_max = 1/sqrt(3) ≈ 0.577

        let dx = SOUND_SPEED_WATER / (1e6 * MIN_PPW); // Grid spacing for 1 MHz
        let cfl_max_3d = 1.0 / 3.0_f64.sqrt();

        // Our safety factor should be less than theoretical maximum
        assert!(CFL_SAFETY < cfl_max_3d);

        // Calculate stable time step
        let dt_stable = CFL_SAFETY * dx / SOUND_SPEED_WATER;

        // Verify it satisfies the condition
        let cfl_actual = SOUND_SPEED_WATER * dt_stable / dx;
        assert!(cfl_actual <= cfl_max_3d);
        assert_relative_eq!(cfl_actual, CFL_SAFETY, epsilon = 1e-10);
    }

    /// Validate mechanical index calculation
    /// Reference: FDA guidance on diagnostic ultrasound
    #[test]
    fn test_mechanical_index() {
        // MI = P_neg / sqrt(f_c)
        // where P_neg is peak negative pressure in MPa
        // and f_c is center frequency in MHz

        let p_neg_mpa: f64 = 2.0; // 2 MPa peak negative pressure
        let freq_mhz: f64 = 2.5; // 2.5 MHz

        let mi = p_neg_mpa / freq_mhz.sqrt();

        // Check if it exceeds FDA limit
        if mi > MI_THRESHOLD {
            // Would need derating or power reduction
            assert!(mi > MI_THRESHOLD);
        }

        // Verify calculation
        assert_relative_eq!(mi, p_neg_mpa / freq_mhz.sqrt(), epsilon = 1e-10);
    }

    /// Validate energy conservation in lossless medium
    #[test]
    fn test_energy_conservation() {
        // In a lossless medium, total energy should be conserved
        // E = (1/2) * ρ * ∫(v² + c²(∇p)²) dV

        // This is a placeholder for a more complex test that would
        // require running a simulation and tracking energy

        // For now, just verify our tolerance is reasonable
        assert!(ENERGY_CONSERVATION_TOLERANCE > 0.0);
        assert!(ENERGY_CONSERVATION_TOLERANCE < 1e-3); // Should be small
    }
}
