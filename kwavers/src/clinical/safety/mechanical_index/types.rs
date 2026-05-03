//! Types for mechanical index safety calculation.

/// Tissue types with different MI safety thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TissueType {
    /// General soft tissue (MI < 1.9)
    SoftTissue,
    /// Ophthalmic tissue (MI < 0.23)
    Ophthalmic,
    /// Lung tissue with gas bodies (MI < 0.7)
    Lung,
    /// Bowel tissue with gas bodies (MI < 0.7)
    Bowel,
    /// Fetal tissue (MI < 1.0 recommended)
    Fetal,
    /// Brain tissue (MI < 1.5)
    Brain,
}

impl TissueType {
    /// Get FDA/WFUMB recommended MI limit for this tissue type
    #[must_use]
    pub fn safety_limit(&self) -> f64 {
        match self {
            Self::SoftTissue => 1.9,  // FDA diagnostic limit
            Self::Ophthalmic => 0.23, // FDA ophthalmic limit
            Self::Lung => 0.7,        // WFUMB gas-body tissue
            Self::Bowel => 0.7,       // WFUMB gas-body tissue
            Self::Fetal => 1.0,       // Conservative fetal limit
            Self::Brain => 1.5,       // Transcranial limit
        }
    }

    /// Get cavitation threshold estimate for this tissue
    #[must_use]
    pub fn cavitation_threshold(&self) -> f64 {
        match self {
            Self::SoftTissue => 0.6,
            Self::Ophthalmic => 0.3,
            Self::Lung => 0.4,   // Lower due to gas bodies
            Self::Bowel => 0.4,  // Lower due to gas bodies
            Self::Fetal => 0.5,  // Conservative estimate
            Self::Brain => 0.55, // Slightly lower than soft tissue
        }
    }
}
/// Mechanical Index calculation result
#[derive(Debug, Clone)]
pub struct MechanicalIndexResult {
    /// Calculated MI value
    pub mi: f64,
    /// Peak rarefactional pressure (MPa)
    pub peak_rarefactional_pressure_mpa: f64,
    /// Center frequency (MHz)
    pub center_frequency_mhz: f64,
    /// Safety assessment
    pub safety_status: SafetyStatus,
    /// Distance from transducer where MI is calculated (cm)
    pub focal_distance_cm: f64,
    /// Tissue-specific safety limit
    pub safety_limit: f64,
}

/// Safety status based on MI value
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SafetyStatus {
    /// MI well below safety limits
    Safe,
    /// MI approaching safety limits (>80% of limit)
    Caution,
    /// MI exceeds recommended limits
    Unsafe,
    /// MI at cavitation threshold
    CavitationRisk,
}


impl MechanicalIndexResult {
    /// Check if MI is within safety limits
    #[must_use]
    pub fn is_safe(&self) -> bool {
        matches!(self.safety_status, SafetyStatus::Safe)
    }

    /// Get safety margin as percentage below limit
    #[must_use]
    pub fn safety_margin_percent(&self) -> f64 {
        ((self.safety_limit - self.mi) / self.safety_limit) * 100.0
    }

    /// Format result for display
    #[must_use]
    pub fn format_report(&self) -> String {
        format!(
            "Mechanical Index Report\n\
             ========================\n\
             MI Value: {:.3}\n\
             Peak Rarefactional Pressure: {:.3} MPa\n\
             Center Frequency: {:.2} MHz\n\
             Focal Distance: {:.1} cm\n\
             Safety Limit: {:.2}\n\
             Safety Status: {:?}\n\
             Safety Margin: {:.1}%\n",
            self.mi,
            self.peak_rarefactional_pressure_mpa,
            self.center_frequency_mhz,
            self.focal_distance_cm,
            self.safety_limit,
            self.safety_status,
            self.safety_margin_percent()
        )
    }
}
