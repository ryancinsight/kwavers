//! Types for thermal-index safety calculation.

/// Thermal-index model specified by the acoustic-output display standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalIndexModel {
    /// Soft-tissue thermal index.
    SoftTissue,
    /// Bone thermal index.
    Bone,
    /// Cranial-bone thermal index.
    CranialBone,
}

impl ThermalIndexModel {
    /// Canonical display label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::SoftTissue => "TIS",
            Self::Bone => "TIB",
            Self::CranialBone => "TIC",
        }
    }
}

/// Thermal-index safety status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalIndexStatus {
    /// TI is below the caution threshold.
    Safe,
    /// TI is at or above 80% of the configured limit.
    Caution,
    /// TI exceeds the configured limit.
    Unsafe,
}

/// Thermal-index calculation result.
#[derive(Debug, Clone)]
pub struct ThermalIndexResult {
    /// Thermal-index model.
    pub model: ThermalIndexModel,
    /// Thermal index, dimensionless.
    pub thermal_index: f64,
    /// Acoustic power after attenuation derating, W.
    pub derated_acoustic_power_w: f64,
    /// Model reference power for a 1 °C rise, W.
    pub reference_power_w: f64,
    /// Center frequency, MHz.
    pub center_frequency_mhz: f64,
    /// Evaluation depth, cm.
    pub depth_cm: f64,
    /// Configured safety limit.
    pub safety_limit: f64,
    /// Safety status.
    pub safety_status: ThermalIndexStatus,
}

impl ThermalIndexResult {
    /// Check if the thermal index is below warning and unsafe thresholds.
    #[must_use]
    pub fn is_safe(&self) -> bool {
        self.safety_status == ThermalIndexStatus::Safe
    }
}
