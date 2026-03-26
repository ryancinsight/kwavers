use crate::core::error::{KwaversError, KwaversResult};

/// Acoustic radiation force push pulse parameters
///
/// # Clinical Values
///
/// - Push duration: 50-400 μs (typ. 100-200 μs)
/// - Push frequency: 3-8 MHz (typ. 5 MHz)
/// - Focus depth: 20-80 mm
/// - F-number: 1.5-3.0 (typ. 2.0)
#[derive(Debug, Clone)]
pub struct PushPulseParameters {
    /// Push pulse frequency (Hz)
    pub frequency: f64,
    /// Push pulse duration (s)
    pub duration: f64,
    /// Peak acoustic intensity (W/m²)
    pub intensity: f64,
    /// Focal depth (m)
    pub focal_depth: f64,
    /// F-number (focal_depth / aperture_width)
    pub f_number: f64,
}

impl Default for PushPulseParameters {
    fn default() -> Self {
        Self {
            frequency: 5.0e6,  // 5 MHz
            duration: 150e-6,  // 150 μs
            intensity: 1.0e6,  // 1 MW/m²
            focal_depth: 0.04, // 40 mm
            f_number: 2.0,
        }
    }
}

impl PushPulseParameters {
    /// Create custom push pulse parameters
    ///
    /// # Arguments
    ///
    /// * `frequency` - Push frequency in Hz
    /// * `duration` - Push duration in seconds
    /// * `intensity` - Peak intensity in W/m²
    /// * `focal_depth` - Focal depth in meters
    /// * `f_number` - F-number (dimensionless)
    pub fn new(
        frequency: f64,
        duration: f64,
        intensity: f64,
        focal_depth: f64,
        f_number: f64,
    ) -> KwaversResult<Self> {
        if frequency <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidValue {
                    parameter: "frequency".to_string(),
                    value: frequency,
                    reason: "must be positive".to_string(),
                },
            ));
        }
        if duration <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidValue {
                    parameter: "duration".to_string(),
                    value: duration,
                    reason: "must be positive".to_string(),
                },
            ));
        }
        if intensity <= 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidValue {
                    parameter: "intensity".to_string(),
                    value: intensity,
                    reason: "must be positive".to_string(),
                },
            ));
        }

        Ok(Self {
            frequency,
            duration,
            intensity,
            focal_depth,
            f_number,
        })
    }
}
