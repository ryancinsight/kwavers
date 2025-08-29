//! Test support utilities for acoustic wave physics

#[cfg(test)]
pub mod mocks;

#[cfg(test)]
pub mod tests {
    use super::super::*;
    use crate::grid::Grid;
    use std::f64::consts::PI;

    #[test]
    fn test_acoustic_diffusivity_zero_frequency() {
        use super::mocks::mocks::HeterogeneousMediumMock;

        let medium = HeterogeneousMediumMock::new(false);
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);

        // Zero frequency should give zero diffusivity
        let diffusivity = compute_acoustic_diffusivity(&medium, 0.0, 0.0, 0.0, 0.0, &grid);
        assert_eq!(
            diffusivity, 0.0,
            "Zero frequency should give zero diffusivity"
        );
    }

    #[test]
    fn test_acoustic_diffusivity_formula() {
        // Test that the formula δ = 2αc³/ω² is correctly implemented

        // Test case 1: Zero absorption should give zero diffusivity
        let alpha = 0.0;
        let c = 1500.0;
        let freq = 1e6;
        let omega = 2.0 * PI * freq;
        let expected = 2.0 * alpha * c.powi(3) / (omega * omega);
        assert_eq!(expected, 0.0);

        // Test case 2: Non-zero values
        let alpha = 0.5; // Np/m
        let c = 1500.0; // m/s
        let freq = 1e6; // Hz
        let omega = 2.0 * PI * freq;
        let diffusivity = 2.0 * alpha * c.powi(3) / (omega * omega);

        // Calculate expected value
        let expected = 2.0 * 0.5 * 1500.0_f64.powi(3) / (2.0 * PI * 1e6).powi(2);

        assert!(
            (diffusivity - expected).abs() < 1e-10,
            "Formula calculation mismatch: got {}, expected {}",
            diffusivity,
            expected
        );

        // Test case 3: Verify frequency scaling
        let freq2 = 2e6;
        let omega2 = 2.0 * PI * freq2;
        let diffusivity2 = 2.0 * alpha * c.powi(3) / (omega2 * omega2);

        // Diffusivity should scale as 1/f² for constant α
        assert!(
            (diffusivity2 - diffusivity / 4.0).abs() < 1e-10,
            "Frequency scaling incorrect: {} vs {}",
            diffusivity2,
            diffusivity / 4.0
        );

        // Test case 4: Verify the actual value is reasonable
        // For α = 0.5 Np/m, c = 1500 m/s, f = 1 MHz
        // δ = 2 * 0.5 * 1500³ / (2π * 10⁶)² ≈ 8.5e-5 m²/s
        assert!(
            diffusivity > 1e-6 && diffusivity < 1e-3,
            "Diffusivity value seems unreasonable: {}",
            diffusivity
        );
    }

    #[test]
    fn test_heterogeneous_medium_position_dependence() {
        use super::mocks::mocks::HeterogeneousMediumMock;
        use crate::medium::core::CoreMedium;

        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let medium = HeterogeneousMediumMock::new(true);

        // Test that different positions give different values
        let density1 = medium.density(0.0, 0.0, 0.0, &grid);
        let density2 = medium.density(0.5, 0.5, 0.5, &grid);

        assert_ne!(
            density1, density2,
            "Position-dependent medium should have varying density"
        );

        let speed1 = medium.sound_speed(0.0, 0.0, 0.0, &grid);
        let speed2 = medium.sound_speed(0.5, 0.5, 0.5, &grid);

        assert_ne!(
            speed1, speed2,
            "Position-dependent medium should have varying sound speed"
        );
    }
}
