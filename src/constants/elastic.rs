//! Elastic wave constants

/// Young's modulus for soft tissue [Pa]
pub const TISSUE_YOUNGS_MODULUS: f64 = 50e3;

/// Poisson's ratio for soft tissue
pub const TISSUE_POISSON_RATIO: f64 = 0.499; // Nearly incompressible

/// Shear modulus for soft tissue [Pa]
pub const TISSUE_SHEAR_MODULUS: f64 = TISSUE_YOUNGS_MODULUS / (2.0 * (1.0 + TISSUE_POISSON_RATIO));

/// Bulk modulus for soft tissue [Pa]
pub const TISSUE_BULK_MODULUS: f64 =
    TISSUE_YOUNGS_MODULUS / (3.0 * (1.0 - 2.0 * TISSUE_POISSON_RATIO));

/// Lamé parameters
pub const TISSUE_LAME_LAMBDA: f64 = TISSUE_BULK_MODULUS - 2.0 * TISSUE_SHEAR_MODULUS / 3.0;
pub const TISSUE_LAME_MU: f64 = TISSUE_SHEAR_MODULUS;

/// Shear wave speed in tissue [m/s]
pub const TISSUE_SHEAR_WAVE_SPEED: f64 = 3.0; // sqrt(μ/ρ) ≈ 3 m/s

/// Compression wave speed in tissue [m/s]
pub const TISSUE_COMPRESSION_WAVE_SPEED: f64 = 1540.0; // Same as acoustic

/// Bond transformation factor for stiffness tensor
pub const BOND_TRANSFORM_FACTOR: f64 = 1.0;

/// Lamé to stiffness conversion factor
pub const LAME_TO_STIFFNESS_FACTOR: f64 = 1.0;

/// Symmetry tolerance for tensor operations
pub const SYMMETRY_TOLERANCE: f64 = 1e-10;
