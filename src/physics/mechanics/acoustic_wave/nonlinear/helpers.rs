// src/physics/mechanics/acoustic_wave/nonlinear/helpers.rs
use super::config::NonlinearWave;
use std::f64;

impl NonlinearWave {
    /// Calculates the phase factor for wave propagation in k-space.
    ///
    /// This method computes a phase factor used in the k-space update step to account for
    /// wave dispersion. The calculation depends on the `k_space_correction_order`
    /// set in the `NonlinearWave` configuration.
    ///
    /// # Arguments
    ///
    /// * `k_val` - The magnitude of the wavevector k.
    /// * `c` - The sound speed at the relevant point in the medium.
    /// * `dt` - The time step size.
    ///
    /// # Returns
    ///
    /// The calculated phase factor (a scalar `f64` value representing the phase angle in radians,
    /// though typically used as `exp(i * phase_factor)` or `cos(phase_factor) + i * sin(phase_factor)`).
    /// The returned value is `omega * dt` where `omega` is the angular frequency, potentially corrected.
    #[inline]
    pub(super) fn calculate_phase_factor(&self, k_val: f64, c: f64, dt: f64) -> f64 {
        match self.k_space_correction_order {
            1 => -c * k_val * dt,  // First order: simple -k*c*dt
            2 => { // Second order correction
                let kc_pi = k_val * c * dt / f64::consts::PI;
                -c * k_val * dt * (1.0 - 0.25 * kc_pi.powi(2))
            },
            3 => { // Third order correction
                let kc_pi = k_val * c * dt / f64::consts::PI;
                let kc_pi_sq = kc_pi.powi(2);
                -c * k_val * dt * (1.0 - 0.25 * kc_pi_sq + 0.05 * kc_pi_sq.powi(2))
            },
            // Default to 4th order (or higher if specified, though current logic effectively caps at 4)
            // This handles any value of k_space_correction_order >= 4 or if it's 0 (though UI asserts 1-4).
            _ => { 
                let kc_pi = k_val * c * dt / f64::consts::PI;
                let kc_pi_sq = kc_pi.powi(2);
                -c * k_val * dt * (1.0 - 0.25 * kc_pi_sq + 0.05 * kc_pi_sq.powi(2) - 0.008 * kc_pi_sq.powi(3))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid; // Use the actual Grid struct
    // ndarray::Array3 is not directly used in this test module scope.

    // Helper to create a basic NonlinearWave instance for testing helpers
    fn create_test_wave_for_helpers() -> NonlinearWave {
        // Use the actual Grid constructor
        let test_grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        NonlinearWave::new(&test_grid)
    }

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_calculate_phase_factor() {
        let mut wave = create_test_wave_for_helpers();
        
        let k_val = 10.0;
        let c = 343.0;
        let dt = 0.001;

        // Order 1
        wave.k_space_correction_order = 1;
        let expected_order1 = -c * k_val * dt;
        let actual_order1 = wave.calculate_phase_factor(k_val, c, dt);
        assert!((actual_order1 - expected_order1).abs() < EPSILON, "Order 1 failed: expected {}, got {}", expected_order1, actual_order1);

        // Order 2
        wave.k_space_correction_order = 2;
        let kc_pi_order2 = k_val * c * dt / f64::consts::PI;
        let expected_order2 = -c * k_val * dt * (1.0 - 0.25 * kc_pi_order2.powi(2));
        let actual_order2 = wave.calculate_phase_factor(k_val, c, dt);
        assert!((actual_order2 - expected_order2).abs() < EPSILON, "Order 2 failed: expected {}, got {}", expected_order2, actual_order2);

        // Order 3
        wave.k_space_correction_order = 3;
        let kc_pi_order3 = k_val * c * dt / f64::consts::PI;
        let kc_pi_sq_order3 = kc_pi_order3.powi(2);
        let expected_order3 = -c * k_val * dt * (1.0 - 0.25 * kc_pi_sq_order3 + 0.05 * kc_pi_sq_order3.powi(2));
        let actual_order3 = wave.calculate_phase_factor(k_val, c, dt);
        assert!((actual_order3 - expected_order3).abs() < EPSILON, "Order 3 failed: expected {}, got {}", expected_order3, actual_order3);

        // Order 4 (and default)
        wave.k_space_correction_order = 4; // Explicitly 4
        let kc_pi_order4 = k_val * c * dt / f64::consts::PI;
        let kc_pi_sq_order4 = kc_pi_order4.powi(2);
        let expected_order4 = -c * k_val * dt * (1.0 - 0.25 * kc_pi_sq_order4 + 0.05 * kc_pi_sq_order4.powi(2) - 0.008 * kc_pi_sq_order4.powi(3));
        let actual_order4 = wave.calculate_phase_factor(k_val, c, dt);
        assert!((actual_order4 - expected_order4).abs() < EPSILON, "Order 4 failed: expected {}, got {}", expected_order4, actual_order4);
        
        // Test default case (e.g. if order was set to 5, it should behave like 4)
        wave.k_space_correction_order = 5; // Invalid, but should hit the `_` arm
        let actual_default_order = wave.calculate_phase_factor(k_val, c, dt);
        assert!((actual_default_order - expected_order4).abs() < EPSILON, "Default order (5) should match order 4: expected {}, got {}", expected_order4, actual_default_order);

        // Test with k_val = 0 (phase factor should be 0)
        wave.k_space_correction_order = 2; // Any order
        let k_val_zero = 0.0;
        let expected_kval_zero = 0.0;
        let actual_kval_zero = wave.calculate_phase_factor(k_val_zero, c, dt);
        assert!((actual_kval_zero - expected_kval_zero).abs() < EPSILON, "k_val=0 failed: expected {}, got {}", expected_kval_zero, actual_kval_zero);
    }
}
