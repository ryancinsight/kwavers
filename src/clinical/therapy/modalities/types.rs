//! Therapy modalities

/// Therapeutic ultrasound mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TherapyMechanism {
    Thermal,
    Mechanical,
    Combined,
}

/// Therapy modality type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TherapyModality {
    HIFU,
    LIFU,
    Histotripsy,
    BBBOpening,
    Sonodynamic,
    Sonoporation,
}

impl TherapyModality {
    /// Check if modality has thermal effects
    pub fn has_thermal_effects(&self) -> bool {
        matches!(self, Self::HIFU | Self::Sonodynamic)
    }

    /// Check if modality has cavitation effects
    pub fn has_cavitation(&self) -> bool {
        matches!(
            self,
            Self::Histotripsy | Self::BBBOpening | Self::Sonoporation | Self::Sonodynamic
        )
    }

    /// Get primary mechanism
    pub fn primary_mechanism(&self) -> TherapyMechanism {
        match self {
            Self::HIFU => TherapyMechanism::Thermal,
            Self::Histotripsy | Self::BBBOpening | Self::Sonoporation => {
                TherapyMechanism::Mechanical
            }
            Self::LIFU | Self::Sonodynamic => TherapyMechanism::Combined,
        }
    }
}
