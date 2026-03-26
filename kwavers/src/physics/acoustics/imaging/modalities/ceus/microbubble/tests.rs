#[cfg(test)]
mod tests {
    use super::super::dynamics::BubbleDynamics;
    use crate::domain::imaging::ultrasound::ceus::{Microbubble, MicrobubblePopulation};

    #[test]
    fn test_microbubble_creation() {
        let bubble = Microbubble::sono_vue();

        assert!((bubble.radius_eq - 1.5e-6).abs() < 1e-9);
        assert!(bubble.shell_elasticity > 0.0);
        assert!(bubble.validate().is_ok());
    }

    #[test]
    fn test_resonance_frequency() {
        let bubble = Microbubble::new(2.0, 1.0, 0.5); // 2 μm radius
        let freq = bubble.resonance_frequency(101325.0, 1000.0);

        // Typical resonance frequency for 2 μm bubble should be around 2-5 MHz
        assert!(freq > 1e6 && freq < 10e6);
    }

    #[test]
    fn test_population_creation() {
        let population = MicrobubblePopulation::new(1e6, 2.5).unwrap();

        // 1e6 bubbles/mL = 1e6 * 1e6 = 1e12 bubbles/m³
        assert!((population.concentration - 1e12).abs() < 1e10);
        assert!(population.reference_bubble.radius_eq > 0.0);
    }

    #[test]
    fn test_bubble_dynamics() {
        let dynamics = BubbleDynamics::new();
        let bubble = Microbubble::definit_y();

        let response = dynamics
            .simulate_oscillation(
                &bubble, 50_000.0, // 50 kPa
                2e6,      // 2 MHz
                1e-6,     // 1 μs
            )
            .unwrap();

        assert!(!response.time.is_empty());
        assert!(!response.radius.is_empty());
        assert_eq!(response.time.len(), response.radius.len());

        // Bubble should oscillate
        let radius_change = response.max_radius_change();
        assert!(radius_change > 0.0);
    }

    #[test]
    fn test_nonlinear_scattering() {
        let dynamics = BubbleDynamics::new();
        let bubble = Microbubble::sono_vue();

        let efficiency = dynamics.nonlinear_scattering_efficiency(
            &bubble, 100_000.0, // 100 kPa
            3e6,       // 3 MHz
        );

        assert!((0.0..=1.0).contains(&efficiency));
    }

    #[test]
    fn test_invalid_microbubble() {
        let bubble = Microbubble {
            radius_eq: -1.0, // Invalid
            shell_thickness: 0.1e-6,
            shell_elasticity: 1000.0,
            shell_viscosity: 0.5,
            polytropic_index: 1.07,
            surface_tension: 0.072,
        };

        assert!(bubble.validate().is_err());
    }
}
