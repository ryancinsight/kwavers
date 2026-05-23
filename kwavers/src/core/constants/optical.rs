//! Optical constants for sonoluminescence calculations

/// Wien's displacement constant (m·K)
pub const WIEN_CONSTANT: f64 = 2.897771955e-3;

/// Blackbody radiation constant (W·m²)
pub const BLACKBODY_CONSTANT: f64 = 3.741771852e-16;

/// Fine structure constant (dimensionless)
pub const FINE_STRUCTURE: f64 = 7.2973525693e-3;

/// Thomson scattering cross section (m²)
pub const THOMSON_CROSS_SECTION: f64 = 6.6524587321e-29;

/// Classical electron radius (m)
pub const ELECTRON_RADIUS: f64 = 2.8179403262e-15;

/// Rydberg constant (1/m)
pub const RYDBERG: f64 = 10973731.568160;

/// Bohr radius (m)
pub const BOHR_RADIUS: f64 = 5.29177210903e-11;

// ============================================================================
// Optical Properties for Tissue
// ============================================================================

/// Tissue absorption coefficient (1/cm)
pub const TISSUE_ABSORPTION_COEFFICIENT: f64 = 0.1;

/// Tissue diffusion coefficient (cm)
pub const TISSUE_DIFFUSION_COEFFICIENT: f64 = 0.03;

/// Default polarization factor
pub const DEFAULT_POLARIZATION_FACTOR: f64 = 1.0;

/// Laplacian center coefficient for optical calculations
pub const LAPLACIAN_CENTER_COEFF: f64 = -2.0;

// ============================================================================
// Biological Tissue Refractive Indices (NIR wavelengths, ~700–900 nm)
// ============================================================================

/// Refractive index of soft biological tissue (dimensionless) in the NIR window.
///
/// This value of 1.4 is the widely adopted approximation for soft tissue
/// at near-infrared wavelengths used in photoacoustic and diffuse-optics models.
///
/// References:
/// - Graaff R et al. (1992). Appl. Opt. 31(10), 1370–1376.
/// - Duck F A (1990). Physical Properties of Tissue. Academic Press. §5.
pub const REFRACTIVE_INDEX_SOFT_TISSUE: f64 = 1.4;
