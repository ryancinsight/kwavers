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

/// Tissue optical absorption coefficient at NIR wavelengths (~700–900 nm) [m⁻¹].
///
/// Representative broadband NIR value for generic soft tissue (37°C).
/// Equivalent to 0.1 cm⁻¹ (× 100 m/cm).  Actual values vary with wavelength:
/// ~2 m⁻¹ at 700 nm to ~40 m⁻¹ at 900 nm for tissue with normal blood content.
///
/// Reference: Tuchin V V (2007). *Tissue Optics*. SPIE Press. Table 1.1;
/// Cheong W et al. (1990). *IEEE J. Quantum Electron.* 26(12):2166–2185.
pub const OPTICAL_ABSORPTION_TISSUE_NIR_M: f64 = 10.0; // m⁻¹

/// Tissue optical reduced scattering coefficient at NIR wavelengths [m⁻¹].
///
/// Representative broadband NIR value for generic soft tissue.
/// Equivalent to 1.0 cm⁻¹.  Actual values are wavelength-dependent and
/// tissue-type-dependent; 100 m⁻¹ is a conservative lower-bound estimate.
///
/// Reference: Tuchin V V (2007). *Tissue Optics*. SPIE Press. Table 1.1;
/// Cheong W et al. (1990). *IEEE J. Quantum Electron.* 26(12):2166–2185.
pub const OPTICAL_SCATTERING_REDUCED_TISSUE_NIR_M: f64 = 100.0; // m⁻¹

/// Tissue absorption coefficient (1/cm)
///
/// Deprecated alias — use `OPTICAL_ABSORPTION_TISSUE_NIR_M` (in m⁻¹) for new code.
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

/// Refractive index of water at 589 nm (sodium D line) at 20°C (dimensionless).
///
/// 1.333 is the standard value for liquid water at visible wavelengths.
/// Use REFRACTIVE_INDEX_WATER (1.33) for broadband or NIR approximations.
///
/// Reference: CRC Handbook of Chemistry and Physics, 103rd ed., Table 10-249.
pub const REFRACTIVE_INDEX_WATER_VIS: f64 = 1.333;

/// Refractive index of borosilicate glass (BK7) at 589 nm (dimensionless).
///
/// Common value for standard optical glass; varies ±0.05 by formulation.
///
/// Reference: Schott Optical Glass Data Sheet; Palik ED (1998) Handbook of
/// Optical Constants of Solids. Academic Press.
pub const REFRACTIVE_INDEX_GLASS: f64 = 1.5;

// ── Electromagnetic free-space constants ──────────────────────────────────────

/// Impedance of free space Z₀ = μ₀c (Ω).
///
/// Exact value: √(μ₀/ε₀) = 376.730 313 668... Ω.
/// Optical impedance of a non-magnetic medium: Z = Z₀/n.
///
/// Reference: BIPM SI Brochure, 9th ed. (2019); NIST CODATA 2018 (Z₀ is not
/// exact after the 2019 SI redefinition but deviates from 376.730 313 668 by
/// < 2×10⁻¹⁰ Ω).
pub const VACUUM_IMPEDANCE: f64 = 376.730_313_668; // Ω

// ── Gold Drude model parameters ───────────────────────────────────────────────
//
// Reference: Johnson PB & Christy RW (1972). "Optical constants of the noble
// metals." Phys. Rev. B 6(12):4370–4379. DOI: 10.1103/PhysRevB.6.4370.
// Drude fit values from Sönnichsen C (2001) PhD thesis, LMU München.

/// High-frequency dielectric permittivity ε∞ of gold (dimensionless).
///
/// Drude model: ε(ω) = ε∞ − ωₚ²/(ω² + iγω)
///
/// Reference: Sönnichsen C (2001). PhD thesis, LMU München.
pub const GOLD_EPS_INF: f64 = 9.84;

/// Plasma frequency ωₚ of gold [rad/s].
///
/// Reference: Sönnichsen C (2001). PhD thesis, LMU München.
pub const GOLD_PLASMA_FREQUENCY_RAD_S: f64 = 1.369e16; // rad/s

/// Drude damping constant γ of gold [rad/s].
///
/// Reference: Sönnichsen C (2001). PhD thesis, LMU München.
pub const GOLD_DRUDE_DAMPING_RAD_S: f64 = 1.079e14; // rad/s
