//! Photoacoustic Physics Implementations
//!
//! This module implements photoacoustic coupling physics, including
//! optical absorption, thermal expansion, and pressure wave generation.

/// Canonical Grüneisen model — temperature-dependent, literature-validated.
///
/// This is the single source of truth for Grüneisen parameter evaluation in kwavers
/// (Sprint 226 SSOT consolidation). Use this type everywhere a Grüneisen value is needed.
pub use crate::photoacoustics::GrueneisenModel;

/// Pulsed laser source for photoacoustic excitation
#[derive(Debug)]
pub struct PulsedLaser {
    /// Peak power (W)
    pub peak_power: f64,
    /// Pulse duration (s)
    pub pulse_duration: f64,
    /// Repetition rate (Hz)
    pub repetition_rate: f64,
    /// Wavelength (m)
    pub wavelength: f64,
    /// Beam profile
    pub beam_profile: BeamProfile,
}

/// Transverse optical-beam intensity profile.
#[derive(Debug, Clone)]
pub enum BeamProfile {
    /// Gaussian profile `exp(−r²/w²)`.
    Gaussian {
        /// 1/e² beam radius w [m].
        beam_radius: f64,
    },
    /// Uniform flat-top (top-hat) profile.
    FlatTop {
        /// Radius of the illuminated disc [m].
        beam_radius: f64,
    },
    /// Non-diffracting Bessel beam.
    Bessel {
        /// Radius of the central lobe [m].
        central_lobe_radius: f64,
    },
}

impl PulsedLaser {
    /// Create a new pulsed laser
    #[must_use]
    pub fn new(peak_power: f64, pulse_duration: f64, wavelength: f64) -> Self {
        Self {
            peak_power,
            pulse_duration,
            repetition_rate: 10.0, // 10 Hz default
            wavelength,
            beam_profile: BeamProfile::Gaussian { beam_radius: 1e-3 },
        }
    }

    /// Compute peak fluence (J/m²)
    #[must_use]
    pub fn peak_fluence(&self) -> f64 {
        match &self.beam_profile {
            BeamProfile::Gaussian { beam_radius } => {
                // For Gaussian beam: Φ₀ = (2E_pulse)/(π w₀²)
                let beam_area = std::f64::consts::PI * beam_radius * beam_radius;
                let pulse_energy = self.peak_power * self.pulse_duration;
                2.0 * pulse_energy / beam_area
            }
            BeamProfile::FlatTop { beam_radius } => {
                let beam_area = std::f64::consts::PI * beam_radius * beam_radius;
                let pulse_energy = self.peak_power * self.pulse_duration;
                pulse_energy / beam_area
            }
            BeamProfile::Bessel {
                central_lobe_radius,
            } => {
                // Simplified for central lobe
                let beam_area = std::f64::consts::PI * central_lobe_radius * central_lobe_radius;
                let pulse_energy = self.peak_power * self.pulse_duration;
                pulse_energy / beam_area
            }
        }
    }

    /// Compute average power (W)
    #[must_use]
    pub fn average_power(&self) -> f64 {
        self.peak_power * self.pulse_duration * self.repetition_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulsed_laser() {
        let laser = PulsedLaser::new(1e6, 10e-9, 800e-9); // 1 MW peak, 10 ns pulse

        let peak_fluence = laser.peak_fluence();
        assert!(peak_fluence > 0.0);

        let avg_power = laser.average_power();
        assert!(avg_power < laser.peak_power); // Average should be less than peak
    }
}
