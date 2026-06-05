//! Reflection calculations for wave propagation

/// Reflection calculator
#[derive(Debug)]
pub struct ReflectionCalculator {
    // Implementation details
}

/// Reflection coefficients
#[derive(Debug, Clone)]
pub struct ReflectionCoefficients {
    /// Amplitude reflection coefficient
    pub amplitude: f64,
    /// Phase shift upon reflection \[radians\]
    pub phase: f64,
    /// Energy reflection coefficient (R = |r|²)
    pub energy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// ReflectionCoefficients stores all three fields correctly.
    #[test]
    fn stores_amplitude_phase_energy() {
        let r = ReflectionCoefficients {
            amplitude: 0.3,
            phase: PI,
            energy: 0.09,
        };
        assert!((r.amplitude - 0.3).abs() < 1e-15);
        assert!((r.phase - PI).abs() < 1e-15);
        assert!((r.energy - 0.09).abs() < 1e-15);
    }

    /// Clone produces an independent copy with identical values.
    #[test]
    fn clone_produces_equal_values() {
        let original = ReflectionCoefficients {
            amplitude: 0.5,
            phase: 0.0,
            energy: 0.25,
        };
        let cloned = original.clone();
        assert!((original.amplitude - cloned.amplitude).abs() < 1e-15);
        assert!((original.phase - cloned.phase).abs() < 1e-15);
        assert!((original.energy - cloned.energy).abs() < 1e-15);
    }

    /// energy = amplitude² holds for loss-free boundary (analytical contract).
    ///
    /// At a water–tissue interface with r = 0.4: R = r² = 0.16.
    #[test]
    fn energy_equals_amplitude_squared_for_lossless_interface() {
        let r = 0.4_f64;
        let coeff = ReflectionCoefficients {
            amplitude: r,
            phase: 0.0,
            energy: r * r,
        };
        let computed_energy = coeff.amplitude * coeff.amplitude;
        assert!(
            (coeff.energy - computed_energy).abs() < 1e-15,
            "energy={}, amplitude²={}",
            coeff.energy,
            computed_energy
        );
    }
}
