//! Optical constants and properties

/// Refractive index of vacuum
pub const VACUUM_REFRACTIVE_INDEX: f64 = 1.0;

/// Refractive index of water at 20°C
pub const WATER_REFRACTIVE_INDEX: f64 = 1.333;

/// Refractive index of tissue (average)
pub const TISSUE_REFRACTIVE_INDEX: f64 = 1.4;

/// Scattering coefficient of tissue [1/m]
pub const TISSUE_SCATTERING: f64 = 10000.0;

/// Absorption coefficient of tissue [1/m]
pub const TISSUE_ABSORPTION: f64 = 100.0;

/// Anisotropy factor for tissue
pub const TISSUE_ANISOTROPY: f64 = 0.9;

/// Reduced scattering coefficient [1/m]
pub const TISSUE_REDUCED_SCATTERING: f64 = TISSUE_SCATTERING * (1.0 - TISSUE_ANISOTROPY);

/// Photoacoustic Grüneisen parameter
pub const GRUNEISEN_TISSUE: f64 = 0.12;
pub const GRUNEISEN_WATER: f64 = 0.11;
pub const GRUNEISEN_BLOOD: f64 = 0.13;

/// Tissue absorption coefficient [1/m]
pub const TISSUE_ABSORPTION_COEFFICIENT: f64 = TISSUE_ABSORPTION;

/// Tissue diffusion coefficient [m²/s]
pub const TISSUE_DIFFUSION_COEFFICIENT: f64 = 1.0 / (3.0 * TISSUE_REDUCED_SCATTERING);

/// Default polarization factor
pub const DEFAULT_POLARIZATION_FACTOR: f64 = 1.0;

/// Laplacian center coefficient
pub const LAPLACIAN_CENTER_COEFF: f64 = -6.0;
