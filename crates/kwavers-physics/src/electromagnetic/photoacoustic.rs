//! Photoacoustic Physics Implementations
//!
//! This module implements photoacoustic coupling physics, including
//! optical absorption, thermal expansion, and pressure wave generation.

use aequitas::systems::si::{
    quantities::{Energy, EnergyPerArea, Frequency, Length, Power, Time},
    units::{Hertz, Millimeter},
};

/// Canonical Grüneisen model — temperature-dependent, literature-validated.
///
/// This is the single source of truth for Grüneisen parameter evaluation in kwavers
/// (Sprint 226 SSOT consolidation). Use this type everywhere a Grüneisen value is needed.
pub use crate::photoacoustics::GrueneisenModel;

/// Pulsed laser source for photoacoustic excitation.
#[derive(Debug)]
pub struct PulsedLaser {
    /// Peak optical power.
    pub peak_power: Power,
    /// Pulse duration.
    pub pulse_duration: Time,
    /// Pulse repetition rate.
    pub repetition_rate: Frequency,
    /// Optical wavelength.
    pub wavelength: Length,
    /// Beam profile
    pub beam_profile: BeamProfile,
}

/// Transverse optical-beam intensity profile.
#[derive(Debug, Clone)]
pub enum BeamProfile {
    /// Gaussian profile `exp(−r²/w²)`.
    Gaussian {
        /// 1/e² beam radius w.
        beam_radius: Length,
    },
    /// Uniform flat-top (top-hat) profile.
    FlatTop {
        /// Radius of the illuminated disc.
        beam_radius: Length,
    },
    /// Non-diffracting Bessel beam.
    Bessel {
        /// Radius of the central lobe.
        central_lobe_radius: Length,
    },
}

impl PulsedLaser {
    /// Create a new pulsed laser.
    #[must_use]
    pub fn new(peak_power: Power, pulse_duration: Time, wavelength: Length) -> Self {
        Self {
            peak_power,
            pulse_duration,
            repetition_rate: Frequency::from_unit::<Hertz>(10.0),
            wavelength,
            beam_profile: BeamProfile::Gaussian {
                beam_radius: Length::from_unit::<Millimeter>(1.0),
            },
        }
    }

    /// Compute peak fluence.
    #[must_use]
    pub fn peak_fluence(&self) -> EnergyPerArea {
        let pulse_energy: Energy = self.peak_power * self.pulse_duration;
        match &self.beam_profile {
            BeamProfile::Gaussian { beam_radius } => {
                // For Gaussian beam: Φ₀ = (2E_pulse)/(π w₀²)
                let beam_area = *beam_radius * *beam_radius;
                pulse_energy / beam_area * (2.0 / std::f64::consts::PI)
            }
            BeamProfile::FlatTop { beam_radius } => {
                let beam_area = *beam_radius * *beam_radius;
                pulse_energy / beam_area * (1.0 / std::f64::consts::PI)
            }
            BeamProfile::Bessel {
                central_lobe_radius,
            } => {
                // Simplified for central lobe
                let beam_area = *central_lobe_radius * *central_lobe_radius;
                pulse_energy / beam_area * (1.0 / std::f64::consts::PI)
            }
        }
    }

    /// Compute average power.
    #[must_use]
    pub fn average_power(&self) -> Power {
        let pulse_energy: Energy = self.peak_power * self.pulse_duration;
        pulse_energy * self.repetition_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::Watt;

    #[test]
    fn test_pulsed_laser() {
        let laser = PulsedLaser::new(
            Power::from_unit::<Watt>(1e6),
            Time::from_base(10e-9),
            Length::from_base(800e-9),
        );

        let peak_fluence = laser.peak_fluence();
        assert!(*peak_fluence.as_base() > 0.0);

        let avg_power = laser.average_power();
        eunomia::assert_relative_eq!(*avg_power.as_base(), 0.1, epsilon = 1e-12);
        assert!(avg_power < laser.peak_power); // Average should be less than peak
    }

    #[test]
    fn beam_profile_fluence_matches_closed_forms() {
        let mut laser = PulsedLaser::new(
            Power::from_unit::<Watt>(1e6),
            Time::from_base(10e-9),
            Length::from_base(800e-9),
        );
        let pulse_energy = 1e6 * 10e-9;
        let radius = 1e-3;
        let expected_gaussian = 2.0 * pulse_energy / (std::f64::consts::PI * radius * radius);
        let gaussian_fluence = laser.peak_fluence();
        eunomia::assert_relative_eq!(
            *gaussian_fluence.as_base(),
            expected_gaussian,
            epsilon = 1e-12
        );

        laser.beam_profile = BeamProfile::FlatTop {
            beam_radius: Length::from_unit::<Millimeter>(1.0),
        };
        let expected_flat_top = pulse_energy / (std::f64::consts::PI * radius * radius);
        let flat_top_fluence = laser.peak_fluence();
        eunomia::assert_relative_eq!(
            *flat_top_fluence.as_base(),
            expected_flat_top,
            epsilon = 1e-12
        );

        laser.beam_profile = BeamProfile::Bessel {
            central_lobe_radius: Length::from_unit::<Millimeter>(1.0),
        };
        let bessel_fluence = laser.peak_fluence();
        eunomia::assert_relative_eq!(
            *bessel_fluence.as_base(),
            expected_flat_top,
            epsilon = 1e-12
        );
    }
}
