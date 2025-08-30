//! Tests for power modulation components

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_modulation_schemes() {
        let control = PowerControl::new(ModulationScheme::Pulsed);
        assert_eq!(control.scheme, ModulationScheme::Pulsed);
    }

    #[test]
    fn test_duty_cycle_limits() {
        let mut control = PowerControl::default();
        control.set_duty_cycle(0.0);
        assert!(control.duty_cycle >= MIN_DUTY_CYCLE);

        control.set_duty_cycle(1.0);
        assert!(control.duty_cycle <= MAX_DUTY_CYCLE);
    }

    #[test]
    fn test_amplitude_controller() {
        let mut controller = AmplitudeController::new(0.5);
        controller.set_target(1.0);

        // Should ramp up over time
        let initial = controller.get_amplitude();
        controller.update(0.1);
        assert!(controller.get_amplitude() > initial);
    }

    #[test]
    fn test_safety_limiter() {
        let mut limiter = SafetyLimiter::new();

        // Should limit excessive amplitude
        let limited = limiter.limit(2.0);
        assert!(limited <= 1.0);

        // Should pass through valid amplitude
        limiter.reset();
        let passed = limiter.limit(0.5);
        assert_eq!(passed, 0.5);
    }

    #[test]
    fn test_pulse_sequence() {
        let mut generator = PulseSequenceGenerator::create_burst_sequence(
            3,     // num_pulses
            0.001, // pulse_duration
            0.001, // pulse_delay
            1.0,   // amplitude
            1e6,   // frequency
        );

        assert_eq!(generator.total_duration(), 3.0 * 0.002);

        let pulse = generator.get_current_pulse(0.0);
        assert!(pulse.is_some());
    }
}
