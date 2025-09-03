//! Time-Reversal Configuration Module
//!
//! Provides configuration structures for time-reversal reconstruction algorithms.

use crate::error::{KwaversError, KwaversResult, ValidationError};

/// Configuration for time-reversal reconstruction
#[derive(Debug, Clone)]
pub struct TimeReversalConfig {
    /// Whether to apply frequency filtering during reconstruction
    pub apply_frequency_filter: bool,

    /// Frequency range for filtering (Hz)
    pub frequency_range: Option<(f64, f64)>,

    /// Whether to use amplitude correction
    pub amplitude_correction: bool,

    /// Maximum amplification factor for amplitude correction
    pub max_amplification: f64,

    /// Time window for reconstruction (seconds)
    pub time_window: Option<(f64, f64)>,

    /// Whether to apply spatial windowing
    pub spatial_windowing: bool,

    /// Number of iterations for iterative reconstruction
    pub iterations: usize,

    /// Convergence tolerance for iterative methods
    pub tolerance: f64,

    /// Whether to use phase conjugation
    pub phase_conjugation: bool,

    /// Whether to apply dispersion correction
    pub dispersion_correction: bool,

    /// Reference sound speed for dispersion correction (m/s)
    pub reference_sound_speed: f64,
}

impl Default for TimeReversalConfig {
    fn default() -> Self {
        Self {
            apply_frequency_filter: false,
            frequency_range: None,
            amplitude_correction: false,
            max_amplification: 10.0,
            time_window: None,
            spatial_windowing: false,
            iterations: 1,
            tolerance: 1e-6,
            phase_conjugation: true,
            dispersion_correction: false,
            reference_sound_speed: 1500.0,
        }
    }
}

impl TimeReversalConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> KwaversResult<()> {
        if self.iterations == 0 {
            return Err(KwaversError::Validation(
                ValidationError::OutOfRange {
                    value: self.iterations as f64,
                    min: 1.0,
                    max: usize::MAX as f64,
                }, /* field: iterations */
            ));
        }

        if self.tolerance <= 0.0 || self.tolerance >= 1.0 {
            return Err(KwaversError::Validation(
                ValidationError::OutOfRange {
                    value: self.tolerance,
                    min: 0.0,
                    max: 1.0,
                }, /* field: tolerance */
            ));
        }

        if let Some((f_min, f_max)) = self.frequency_range {
            if f_min >= f_max || f_min < 0.0 {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "frequency_range".to_string(),
                    value: format!("({f_min}, {f_max})"),
                    constraint: "min must be less than max and non-negative".to_string(),
                }));
            }
        }

        if let Some((t_min, t_max)) = self.time_window {
            if t_min >= t_max || t_min < 0.0 {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "time_window".to_string(),
                    value: format!("({t_min}, {t_max})"),
                    constraint: "min must be less than max and non-negative".to_string(),
                }));
            }
        }

        if self.max_amplification <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::OutOfRange {
                    value: self.max_amplification,
                    min: 0.0,
                    max: usize::MAX as f64,
                }, /* field: max_amplification */
            ));
        }

        if self.reference_sound_speed <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::OutOfRange {
                    value: self.reference_sound_speed,
                    min: 0.0,
                    max: usize::MAX as f64,
                }, /* field: reference_sound_speed */
            ));
        }

        Ok(())
    }
}
