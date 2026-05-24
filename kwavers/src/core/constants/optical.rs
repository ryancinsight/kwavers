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

/// Refractive index of pure water (dimensionless) at ~589 nm (sodium D-line), 20 °C.
///
/// Value: 1.333 is the most widely cited approximation; 1.33 is used in photoacoustic
/// and diffuse-optics models for biological fluids (blood plasma, interstitial fluid).
///
/// References:
/// - Hale G M & Querry M R (1973). Appl. Opt. 12(3), 555–563.
/// - Daimon M & Masumura A (2007). Appl. Opt. 46(18), 3811–3820.
pub const REFRACTIVE_INDEX_WATER: f64 = 1.33;

/// Refractive index of soft biological tissue (dimensionless) in the NIR window.
///
/// This value of 1.4 is the widely adopted approximation for soft tissue
/// at near-infrared wavelengths used in photoacoustic and diffuse-optics models.
///
/// References:
/// - Graaff R et al. (1992). Appl. Opt. 31(10), 1370–1376.
/// - Duck F A (1990). Physical Properties of Tissue. Academic Press. §5.
pub const REFRACTIVE_INDEX_SOFT_TISSUE: f64 = 1.4;

/// Refractive index of blood plasma and biological fluids (dimensionless, NIR).
///
/// Value: 1.335 — characteristic of protein-containing aqueous solutions such as
/// blood plasma, whole blood, urine, synovial fluid, and similar biological fluids.
/// Slightly higher than pure water (1.33) due to dissolved proteins and electrolytes.
///
/// References:
/// - Faber D J et al. (2004). Opt. Lett. 29(22), 2641–2643 (blood plasma).
/// - Duck, F. A. (1990). Physical Properties of Tissue. Academic Press, Table 7.1.
pub const REFRACTIVE_INDEX_BIOLOGICAL_FLUID: f64 = 1.335;

/// Refractive index of brain parenchyma (dimensionless, NIR ~700–900 nm).
///
/// Value: 1.37 — representative mean for brain tissue in the near-infrared.
/// White matter typically ranges 1.38–1.41 (higher due to myelin lipid content)
/// while grey matter is closer to 1.36–1.37 (higher water fraction).  The value
/// 1.37 is used as a bulk-tissue approximation when sub-region differentiation is
/// not required.  Used in photoacoustic and transcranial optical propagation models.
///
/// References:
/// - Tuchin V V (2007). *Tissue Optics*. SPIE Press, Table 1.1.
/// - Duck, F. A. (1990). *Physical Properties of Tissue*, §7.
/// - Giannios P et al. (2016). J. Biophotonics 9(1–2):71–78 (white/grey matter
///   refractive index measured separately at 1064 nm).
pub const REFRACTIVE_INDEX_BRAIN_TISSUE: f64 = 1.37;

/// Refractive index of cerebrospinal fluid (dimensionless, NIR).
///
/// Value: 1.333 — close to pure water due to very low protein content.
/// Used in transcranial optical and photoacoustic models.
///
/// Reference: Duck, F. A. (1990). Physical Properties of Tissue, Table 7.1.
pub const REFRACTIVE_INDEX_CSF: f64 = 1.333;

/// Refractive index of parenchymal soft tissue in the NIR window (dimensionless).
///
/// Value: 1.38 — representative of liver parenchyma, skeletal muscle, and
/// general soft tissue at NIR wavelengths (~700–900 nm). More precise than the
/// rounded `REFRACTIVE_INDEX_SOFT_TISSUE = 1.4` approximation.
///
/// References:
/// - Graaff R et al. (1992). Appl. Opt. 31(10), 1370–1376.
/// - Tuchin V V (2007). Tissue Optics. SPIE Press. Table 1.1.
pub const REFRACTIVE_INDEX_SOFT_TISSUE_NIR: f64 = 1.38;
