//! Medical and bioeffects constants

/// FDA derating factor for in situ intensity
pub const FDA_DERATING_FACTOR: f64 = 0.3;

/// Typical diagnostic ultrasound frequency (Hz)
pub const DIAGNOSTIC_FREQUENCY: f64 = 3.5e6;

/// Typical therapeutic ultrasound frequency (Hz)
pub const THERAPEUTIC_FREQUENCY: f64 = 1e6;

/// HIFU frequency range minimum (Hz)
pub const HIFU_FREQUENCY_MIN: f64 = 0.5e6;

/// HIFU frequency range maximum (Hz)
pub const HIFU_FREQUENCY_MAX: f64 = 5e6;

/// Spatial peak temporal average intensity limit (W/cm²)
pub const ISPTA_LIMIT: f64 = 720.0;

/// Spatial peak pulse average intensity limit (W/cm²)
pub const ISPPA_LIMIT: f64 = 190.0;

/// Thermal dose threshold for tissue damage (CEM43)
pub const THERMAL_DOSE_THRESHOLD: f64 = 240.0;

/// Perfusion rate in tissue (1/s)
pub const TISSUE_PERFUSION_RATE: f64 = 5e-4;

/// Blood specific heat capacity (J/(kg·K))
pub const BLOOD_SPECIFIC_HEAT: f64 = 3617.0;

/// Typical HIFU focal intensity (W/cm²)
pub const HIFU_FOCAL_INTENSITY: f64 = 1000.0;

/// Default ultrasound frequency (Hz)
pub const DEFAULT_ULTRASOUND_FREQUENCY: f64 = 1e6;

/// Standard pressure amplitude (Pa)
pub const STANDARD_PRESSURE_AMPLITUDE: f64 = 1e6;

/// Standard beam width (m)
pub const STANDARD_BEAM_WIDTH: f64 = 0.01;