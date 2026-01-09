//! Physical constants for heterogeneous media

/// Minimum physical density to prevent division by zero (kg/m³)
pub const MIN_PHYSICAL_DENSITY: f64 = 1.0;

/// Minimum physical sound speed to ensure stability (m/s)
pub const MIN_PHYSICAL_SOUND_SPEED: f64 = 100.0;

/// Minimum physical viscosity (Pa·s)
pub const MIN_PHYSICAL_VISCOSITY: f64 = 1e-6;

/// Minimum surface tension (N/m)
pub const MIN_SURFACE_TENSION: f64 = 1e-6;

/// Default polytropic index for air
pub const DEFAULT_POLYTROPIC_INDEX: f64 = 1.4;
