//! Therapy modality definitions and implementations
//!
//! Provides different ultrasound therapy modalities with their specific characteristics.

/// Therapy modality types
#[derive(Debug, Clone, Copy, PartialEq))]
pub enum TherapyModality {
    /// High-Intensity Focused Ultrasound (thermal ablation)
    HIFU,
    /// Low-Intensity Focused Ultrasound (neuromodulation)
    LIFU,
    /// Histotripsy (mechanical ablation)
    Histotripsy,
    /// Blood-Brain Barrier opening
    BBBOpening,
    /// Sonodynamic therapy (with sonosensitizers)
    Sonodynamic,
    /// Sonoporation (cell membrane permeabilization)
    Sonoporation,
    /// Microbubble-mediated therapy
    MicrobubbleTherapy,
}

/// Therapy mechanism types
#[derive(Debug, Clone, Copy, PartialEq))]
pub enum TherapyMechanism {
    /// Thermal effects (hyperthermia, ablation)
    Thermal,
    /// Mechanical effects (cavitation, radiation force)
    Mechanical,
    /// Chemical effects (ROS generation, drug activation)
    Chemical,
    /// Combined effects
    Combined,
}

impl TherapyModality {
    /// Get the primary mechanism for this modality
    pub fn primary_mechanism(&self) -> TherapyMechanism {
        match self {
            Self::HIFU => TherapyMechanism::Thermal,
            Self::LIFU => TherapyMechanism::Mechanical,
            Self::Histotripsy => TherapyMechanism::Mechanical,
            Self::BBBOpening => TherapyMechanism::Mechanical,
            Self::Sonodynamic => TherapyMechanism::Chemical,
            Self::Sonoporation => TherapyMechanism::Mechanical,
            Self::MicrobubbleTherapy => TherapyMechanism::Combined,
        }
    }

    /// Check if thermal effects are significant
    pub fn has_thermal_effects(&self) -> bool {
        matches!(self, Self::HIFU | Self::MicrobubbleTherapy)
    }

    /// Check if cavitation is expected
    pub fn has_cavitation(&self) -> bool {
        matches!(
            self,
            Self::Histotripsy | Self::BBBOpening | Self::MicrobubbleTherapy | Self::Sonoporation
        )
    }

    /// Get typical frequency range [Hz]
    pub fn frequency_range(&self) -> (f64, f64) {
        match self {
            Self::HIFU => (0.5e6, 3.0e6),
            Self::LIFU => (0.2e6, 1.0e6),
            Self::Histotripsy => (0.5e6, 3.0e6),
            Self::BBBOpening => (0.2e6, 1.5e6),
            Self::Sonodynamic => (0.5e6, 3.0e6),
            Self::Sonoporation => (0.5e6, 3.0e6),
            Self::MicrobubbleTherapy => (0.2e6, 2.0e6),
        }
    }

    /// Get typical pressure range [Pa]
    pub fn pressure_range(&self) -> (f64, f64) {
        match self {
            Self::HIFU => (1e6, 10e6),
            Self::LIFU => (0.05e6, 0.5e6),
            Self::Histotripsy => (10e6, 100e6),
            Self::BBBOpening => (0.1e6, 0.5e6),
            Self::Sonodynamic => (0.1e6, 1.0e6),
            Self::Sonoporation => (0.1e6, 1.0e6),
            Self::MicrobubbleTherapy => (0.05e6, 1.0e6),
        }
    }
}
