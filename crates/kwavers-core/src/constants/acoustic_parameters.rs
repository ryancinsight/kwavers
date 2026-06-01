//! Acoustic material parameters with literature references
//!
//! All constants are documented with their sources for scientific validation.

/// Water absorption coefficient α₀ at reference frequency
/// Value: 0.0022 dB/(MHz^y cm)
/// Reference: Duck, F. A. (1990). "Physical properties of tissue"
pub const WATER_ABSORPTION_ALPHA_0: f64 = 0.0022;

/// Water absorption frequency power law exponent
/// Value: 1.05 (slightly superlinear)
/// Reference: Szabo, T. L. (1994). "Time domain wave equations for lossy media"
pub const WATER_ABSORPTION_POWER: f64 = 1.05;

/// Nonlinearity parameter B/A for water at 20°C
/// Value: 5.0 (Beyer 1960 polynomial evaluated at 20°C ≈ 4.965)
/// Reference: Beyer, R. T. (1960). "Parameter of nonlinearity in fluids"
pub const WATER_NONLINEARITY_B_A: f64 = 5.0;

/// Nonlinearity parameter B/A for blood at 37°C
/// Value: 6.1
/// Reference: Law, W. K., et al. (1985). "Nonlinearity parameter B/A for biological fluids"
pub const BLOOD_NONLINEARITY_B_A: f64 = 6.1;

/// Nonlinearity parameter B/A for soft tissue
/// Value: 7.0 (average)
/// Reference: Gong, X. F., et al. (1989). "Ultrasonic nonlinearity parameter in biological media"
pub const TISSUE_NONLINEARITY_B_A: f64 = 7.0;

/// Water viscosity at 20°C (Pa·s)
/// Value: 1.002e-3
/// Reference: CRC Handbook of Chemistry and Physics
pub const WATER_VISCOSITY_20C: f64 = 1.002e-3;

/// Blood viscosity at 37°C (Pa·s)
/// Value: 3.5e-3
/// Reference: Rosenson, R. S., et al. (1996). "Distribution of blood viscosity values"
pub const BLOOD_VISCOSITY_37C: f64 = 3.5e-3;

/// Blood plasma viscosity at 37°C (Pa·s)
///
/// Value: 1.2e-3 Pa·s.  Plasma lacks red blood cells, so it behaves as a
/// Newtonian fluid with viscosity ~1.2 mPa·s at body temperature.
///
/// Reference: Duck, F. A. (1990). *Physical Properties of Tissue*. Academic Press,
/// London, Chapter 4 (viscosity section, p. 112); Mills, C. J. (1966) Phys. Med. Biol.
/// 11(4):641–646 (plasma viscometry at 37°C).
pub const BLOOD_PLASMA_VISCOSITY_37C: f64 = 1.2e-3;

/// Effective shear viscosity of generic soft tissue at 37°C (Pa·s)
///
/// Value: 3.0×10⁻³ Pa·s — empirical effective viscosity for acoustic absorption
/// modeling in soft tissue. Approximately 3× water viscosity, reflecting
/// restricted molecular motion in the tissue matrix.
///
/// Reference: Duck FA (1990). Physical Properties of Tissue. Academic Press,
/// Table 5.3.
pub const VISCOSITY_SOFT_TISSUE: f64 = 3.0e-3;

/// Shear viscosity of parenchymal tissues at 37°C (Pa·s)
///
/// Value: 2.0×10⁻³ Pa·s
/// Empirical effective shear viscosity characteristic of major organ parenchyma
/// (brain white/gray matter, liver, kidney cortex, kidney medulla, skeletal
/// muscle, adipose tissue). Approximately 2× water viscosity.
/// Reference: Duck, F. A. (1990). Physical Properties of Tissue, Tables 5.2–5.3.
pub const VISCOSITY_PARENCHYMAL_TISSUE: f64 = 2.0e-3;

/// Water vapor pressure at 20°C (Pa)
/// Value: 2339
/// Reference: Wagner & Pruss (2002). "The IAPWS formulation for water"
pub const WATER_VAPOR_PRESSURE_20C: f64 = 2339.0;

/// Reference frequency for absorption calculations (Hz).
///
/// Value: `1.0e6 Hz` (1 MHz) — the standard reference frequency in medical
/// ultrasound per Pinkerton (1949) and IEC 62127-1.
pub const REFERENCE_FREQUENCY_HZ: f64 = 1.0e6;

/// Minimum time step for acoustic simulations (s)
/// Value: 1e-10
/// Based on CFL condition for typical ultrasound speeds
pub const MIN_ACOUSTIC_TIME_STEP: f64 = 1e-10;

// MAX_TIME_STEP moved to numerical.rs to avoid duplication

/// Rayleigh collapse time coefficient
/// Value: 0.915
/// Reference: Rayleigh (1917). "On the pressure developed in a liquid"
pub const RAYLEIGH_COLLAPSE_COEFFICIENT: f64 = 0.915;

/// Air specific heat at constant pressure (J/(kg·K))
/// Value: 1005
/// Reference: NIST
pub const AIR_SPECIFIC_HEAT_CP: f64 = 1005.0;

/// Mole fraction of molecular oxygen (O₂) in dry air [-].
///
/// Standard dry-air composition: 20.946% O₂ by mole (NIST SRM 2659a).
/// Used for ROS generation models and gas-transport calculations.
///
/// Reference: NIST (2020) *Composition of dry air at sea level*.
pub const AIR_O2_FRACTION: f64 = 0.2095;

/// Air thermal conductivity at 20°C (W/(m·K))
/// Value: 0.0257
/// Reference: NIST
pub const AIR_THERMAL_CONDUCTIVITY: f64 = 0.0257;

/// Polytropic index for air [-]
///
/// γ = Cp/Cv = 7/5 = 1.4 for diatomic ideal gases (Air, N₂, O₂) at ambient conditions.
/// SSOT: delegates to `HEAT_CAPACITY_RATIO_DIATOMIC` in `thermodynamic` constants.
pub const AIR_POLYTROPIC_INDEX: f64 =
    crate::constants::thermodynamic::HEAT_CAPACITY_RATIO_DIATOMIC;

/// Classical acoustic absorption coefficient for air (m·s²).
///
/// The Stokes–Kirchhoff classical absorption follows α = A·f², where
/// A is the absorption constant. At standard conditions (20°C, 1 atm):
/// A = 2·ω²·(η_s + 3η_b/4 + κ·(1/cv − 1/cp)·M/R) / (3·ρ₀·c₀³)
///
/// Empirical fit to ISO 9613-1:1993 data at 293.15 K, 1 atm, 50% RH:
/// A ≈ 1.84×10⁻¹¹ m/Hz²  (absorption per meter, per square Hertz).
///
/// Reference: ISO 9613-1:1993. "Acoustics — Attenuation of sound during
/// propagation outdoors — Part 1: Calculation of the absorption of sound
/// by the atmosphere."
pub const AIR_ABSORPTION_ALPHA_0: f64 = 1.84e-11;

/// Classical absorption power-law exponent for air (dimensionless).
///
/// Air absorption follows the classical quadratic frequency law (f²) due to
/// viscous and thermal conduction mechanisms (Stokes–Kirchhoff). The exponent
/// is exactly 2 for the classical regime at audio and low ultrasonic frequencies.
///
/// Molecular relaxation corrections are significant above ~100 kHz but are
/// captured by frequency-dependent ISO 9613-1 models, not this exponent.
///
/// Reference: Pierce AD (1989). *Acoustics: An Introduction to Its Physical
/// Principles and Applications*. ASA, Chapter 10.
pub const AIR_ABSORPTION_POWER: f64 = 2.0;

/// Default grid spacing for medical ultrasound (m)
/// Value: 1e-4 (0.1 mm)
/// Based on λ/10 criterion at 1.5 `MHz`
pub const DEFAULT_GRID_SPACING: f64 = 1e-4;

/// Acoustic shock detection threshold for gradient-based detectors
/// Value: 0.5 (relative gradient)
/// Empirically determined for robust shock capture
pub const ACOUSTIC_SHOCK_DETECTION_THRESHOLD: f64 = 0.5;

/// Modal decay threshold for smooth solution detection
/// Value: 1e-3
/// High-order modes should be < 0.1% for smooth solutions
pub const MODAL_DECAY_THRESHOLD: f64 = 1e-3;

/// Base sound speed for the Hounsfield-unit-to-acoustic-speed soft-tissue model (m/s).
///
/// This is the empirical model intercept: at HU = 0 (water-equivalent tissue),
/// the linear HU model `c = 1480 + 0.18·clamp(HU, −150, 250)` evaluates to exactly
/// 1480 m/s.  This value is a calibration parameter of the soft-tissue tissue-acoustic
/// model, **not** the speed of sound in pure water (`SOUND_SPEED_WATER = 1482 m/s`).
/// Substituting `SOUND_SPEED_WATER` for this constant would introduce a 2 m/s bias
/// in the body-tissue speed map and shift the model off its calibrated domain.
///
/// # Reference
/// Schneider U et al. (1996). "The calibration of CT Hounsfield units for
/// radiotherapy treatment planning." *Phys. Med. Biol.* 41(1), 111–124.
/// DOI: 10.1088/0031-9155/41/1/009.
pub const SOFT_TISSUE_HU_BASE_SPEED_M_S: f64 = 1480.0;

// ============================================================================
// Medical Ultrasound Frequency Ranges
// ============================================================================

/// Minimum diagnostic ultrasound frequency (Hz)
/// Value: 1 MHz
pub const DIAGNOSTIC_FREQ_MIN: f64 = 1e6;

/// Maximum diagnostic ultrasound frequency (Hz)
/// Value: 20 MHz
pub const DIAGNOSTIC_FREQ_MAX: f64 = 20e6;

/// Minimum therapeutic ultrasound frequency (Hz)
/// Value: 0.5 MHz
pub const THERAPEUTIC_FREQ_MIN: f64 = 0.5e6;

/// Maximum therapeutic ultrasound frequency (Hz)
/// Value: 5 MHz
pub const THERAPEUTIC_FREQ_MAX: f64 = 5e6;

// ============================================================================
// HIFU (High-Intensity Focused Ultrasound) Parameters
// ============================================================================

/// Typical HIFU frequency (Hz)
/// Value: 1 MHz
pub const HIFU_FREQUENCY: f64 = 1e6;

/// Minimum HIFU intensity (W/cm²)
/// Value: 100 W/cm²
pub const HIFU_INTENSITY_MIN: f64 = 100.0;

/// Maximum HIFU intensity (W/cm²)
/// Value: 10000 W/cm²
pub const HIFU_INTENSITY_MAX: f64 = 10000.0;

// ============================================================================
// Bone Properties
// ============================================================================

/// Sound speed in dense cortical bone (m/s)
///
/// Value: 3500 m/s — compressional wave speed in bovine cortical bone.
///
/// Reference: Hosokawa & Otani (1997). "Ultrasonic wave propagation in bovine cancellous bone."
/// J. Acoust. Soc. Am. 101(1), 558–562. DOI: 10.1121/1.418118.
pub const BONE_SOUND_SPEED: f64 = 3500.0;

/// Sound speed in human skull bone (m/s)
///
/// Value: 2900 m/s — effective longitudinal wave speed in the human skull,
/// which has a layered diploe structure (inner cortex + cancellous diploe +
/// outer cortex) that produces a lower effective speed than dense cortical bone.
///
/// Used in transcranial ultrasound propagation models where skull bone is the
/// primary acoustic barrier. Distinct from `BONE_SOUND_SPEED` (3500 m/s for
/// dense cortical), which applies to long bones with minimal cancellous content.
///
/// References:
/// - Fry & Barger (1978). "Acoustical properties of the human skull."
///   J. Acoust. Soc. Am. 63(5), 1576–1590. DOI: 10.1121/1.381852.
/// - Marsac et al. (2017). "MR-guided adaptive focusing of therapeutic
///   ultrasound beams in the human head." Med. Phys. 39(2), 1141–1149.
pub const SOUND_SPEED_SKULL: f64 = 2900.0;

/// Skull-bone attenuation upper bound in the Marsac et al. 2017 linear porosity model [Np/(m·MHz)].
///
/// The Marsac porosity-blend formula interpolates between soft tissue and this value:
/// ```text
/// α(φ) = α_soft·(1 − φ) + 70·φ   [Np/(m·MHz)]
/// ```
/// At φ = 1 (fully cortical bone), α = 70 Np/(m·MHz) ≈ 6.1 dB/(cm·MHz).
///
/// Conversion: α[dB/(cm·MHz)] = 70 × `NP_TO_DB` / 100 ≈ 6.08.
///
/// Reference: Marsac L et al. (2017). "MR-guided adaptive focusing of therapeutic
/// ultrasound beams in the human head." *Med. Phys.* 39(2), 1141–1149.
/// DOI: 10.1002/mp.12168.
pub const SKULL_ATTENUATION_MARSAC_MAX_NP_PER_M_MHZ: f64 = 70.0;

/// Bone density (kg/m³)
/// Value: 1900 kg/m³
/// Reference: Duck, F. A. (1990). "Physical properties of tissue"
pub const BONE_DENSITY: f64 = 1900.0;

/// Skull (cortical bone) density (kg/m³).
///
/// Cortical bone of the human skull is slightly denser than the generic bone
/// value (BONE_DENSITY = 1900 kg/m³) due to its compact lamellar structure.
/// Value: 1920 kg/m³.
/// Reference: Duck (1990) Table 3.3.
pub const DENSITY_SKULL: f64 = 1920.0;

/// Skull cortical bone longitudinal sound speed [m/s] — Pinton et al. (2012) model.
///
/// Value: 3100 m/s — measured longitudinal wave speed for dense cortical skull bone
/// at body temperature. Used in the 3-layer skull model (cortical/trabecular/cortical)
/// for transcranial focused ultrasound.
///
/// Distinct from `SOUND_SPEED_SKULL = 2900 m/s` (Marsac 2017 mean across porosity range).
///
/// Reference: Pinton G et al. (2012). "Attenuation, scattering, and absorption of
/// ultrasound in the skull bone." *Med. Phys.* 39(1):299–307. DOI: 10.1118/1.3668316.
pub const SOUND_SPEED_SKULL_CORTICAL: f64 = 3100.0;

/// Skull cortical bone shear wave speed [m/s].
///
/// Value: 1600 m/s — measured shear wave speed in cortical skull bone.
/// Range reported in the literature: 1400–1800 m/s depending on orientation and
/// bone quality.
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307.
pub const SHEAR_SPEED_SKULL_CORTICAL: f64 = 1600.0;

/// Skull trabecular bone longitudinal sound speed [m/s].
///
/// Value: 2400 m/s — longitudinal speed in the porous trabecular (diploe) layer of
/// the skull, which is slower than compact cortical bone due to the higher porosity.
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307.
pub const SOUND_SPEED_SKULL_TRABECULAR: f64 = 2400.0;

/// Skull trabecular bone density [kg/m³].
///
/// Value: 1600 kg/m³ — lower than cortical skull (1920 kg/m³) due to trabecular
/// porosity. Used in the 3-layer skull transmission model.
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307.
pub const DENSITY_SKULL_TRABECULAR: f64 = 1600.0;

/// Skull trabecular bone shear wave speed [m/s].
///
/// Value: 1200 m/s — lower than cortical shear speed due to trabecular porosity.
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307.
pub const SHEAR_SPEED_SKULL_TRABECULAR: f64 = 1200.0;

/// Skull suture (fibrous joint) longitudinal sound speed [m/s].
///
/// Cranial sutures are connective tissue joints between skull bones; their acoustic
/// speed is intermediate between soft tissue (~1540 m/s) and trabecular bone (~2400 m/s).
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307;
/// Marquet F et al. (2009). *Phys. Med. Biol.* 54(9):2895–2916.
pub const SOUND_SPEED_SKULL_SUTURE: f64 = 1800.0;

/// Skull suture (fibrous joint) density [kg/m³].
///
/// Suture tissue is predominantly fibrous connective tissue; density is close to
/// soft tissue (~1040–1100 kg/m³) but elevated by mineralisation at suture margins.
///
/// Reference: Pinton G et al. (2012). *Med. Phys.* 39(1):299–307.
pub const DENSITY_SKULL_SUTURE: f64 = 1200.0;

/// Bone nonlinearity parameter (B/A)
/// Value: 8.0
/// Reference: Estimated from tissue properties
pub const BONE_NONLINEARITY: f64 = 8.0;

/// Bone attenuation coefficient [dB/(MHz·cm)]
/// Value: 20.0
/// Reference: Wear, K. A. (2000). "Measurements of phase velocity and group velocity in bone"
pub const BONE_ATTENUATION: f64 = 20.0;

// ============================================================================
// Reference Frequencies
// ============================================================================

/// Reference frequency for absorption calculations (Hz)
/// Standard reference: 1 MHz
pub const REFERENCE_FREQUENCY_FOR_ABSORPTION_HZ: f64 = 1e6;

/// Tissue acoustic reference frequency for heterogeneous-medium simulations (Hz).
///
/// Value: 180 kHz — representative low-frequency therapeutic ultrasound
/// (low-intensity focused ultrasound, LIFU) used as the absorption-power-law
/// reference in tissue medium factory models.
///
/// Reference: Haar GT & Coussios C (2007). "High intensity focused ultrasound:
/// physical principles and devices." *Int. J. Hyperthermia* 23(2), 89–104.
pub const REFERENCE_FREQUENCY_TISSUE_HZ: f64 = 180_000.0;

/// Default sampling frequency (Hz)
/// Standard: 10 MHz for ultrasound simulations
pub const SAMPLING_FREQUENCY_DEFAULT: f64 = 10e6;

// ============================================================================
// Absorption and Attenuation Parameters
// ============================================================================

/// Tissue absorption coefficient at 1 MHz [dB/(MHz^y cm)]
pub const ABSORPTION_TISSUE: f64 = 0.75;

/// Absorption power law exponent
pub const ABSORPTION_POWER: f64 = 1.05;

/// Conversion factor from decibels to nepers (dimensionless)
///
/// # Definition
///
/// One neper (Np) is the natural-logarithm amplitude ratio; one decibel (dB) is
/// the base-10 logarithm amplitude ratio scaled by 20:
///
///   20 log₁₀(A₂/A₁) (dB) = 20 ln(A₂/A₁) / ln(10) (dB)
///   ln(A₂/A₁) (Np)
///
/// Therefore:  1 dB = ln(10) / 20 Np
///
/// # Exact value
///
/// `ln(10) / 20 = 2.302_585_092_994_046 / 20 = 0.115_129_254_649_702_28`
///
/// Prior approximations (`0.1151`, `1/8.686`) deviate by up to 0.033% from
/// the exact value. The exact constant eliminates inter-module inconsistency.
///
/// # References
/// - NIST SP 330 (2019), "The International System of Units (SI)", §4.1.
/// - ISO 80000-3:2019, Quantities and units — Part 3: Space and time.
pub const DB_TO_NP: f64 = 0.115_129_254_649_702_28; // ln(10) / 20

/// Conversion factor from nepers to decibels (dimensionless)
///
/// Exact reciprocal of `DB_TO_NP`:  20 / ln(10) = 8.685_889_638_065_037
///
/// # References
/// - NIST SP 330 (2019); ISO 80000-3:2019.
pub const NP_TO_DB: f64 = 8.685_889_638_065_037; // 20 / ln(10)

/// Coefficient of nonlinearity β for water at 20°C (dimensionless).
///
/// β = 1 + B/(2A) = 1 + 5.0/2 = 3.5.
///
/// B/A = 5.0 is the Beyer (1960) measurement at 20°C.
/// Delegates to [`crate::constants::tissue_acoustics::B_OVER_A_WATER_37C`]
/// for the 37°C body-temperature value (B/A = 5.0, β = 3.5, same numerical value).
///
/// # Reference
/// Beyer, R. T. (1960). "Parameter of nonlinearity in fluids."
/// *J. Acoust. Soc. Am.* 32(6), 719–721. DOI: 10.1121/1.1908195.
pub const NONLINEARITY_COEFFICIENT_WATER: f64 = 3.5; // β = 1 + B/(2A), B/A = 5.0

/// Coefficient of nonlinearity β for soft tissue at 37°C (dimensionless).
///
/// β = 1 + B/(2A) = 1 + 6.0/2 = 4.0.
///
/// B/A = 6.0 is the Hamilton & Blackstock (1998) midpoint for soft tissue
/// (reported range: B/A ≈ 5–9, Table 14.1; the soft-tissue mean near 6).
/// The full SSOT B/A constant is
/// [`crate::constants::tissue_acoustics::B_OVER_A_SOFT_TISSUE`] = 6.5,
/// which gives β = 4.25; the 4.0 here uses the conservative lower midpoint
/// B/A = 6 for models that do not differentiate tissue type.
///
/// # Reference
/// Hamilton, M. F., & Blackstock, D. T. (1998). *Nonlinear Acoustics*.
/// Academic Press. Table 14.1.
pub const NONLINEARITY_COEFFICIENT_TISSUE: f64 = 4.0; // β = 1 + B/(2A), B/A = 6.0

/// Acoustic nonlinearity parameter B/A for water at 20°C (dimensionless).
///
/// B/A = 5.0 is the Beyer (1960) polynomial evaluated at 20°C (≈ 4.965,
/// rounded to 5.0). Used when a scalar B/A ratio — rather than the derived
/// β = 1 + B/(2A) — is needed directly.
///
/// Note: `WATER_NONLINEARITY_B_A = 5.0` in this module is distinct from
/// `B_OVER_A_WATER = 5.2` in [`crate::constants::tissue_acoustics`],
/// which is the 20°C value including molecular relaxation corrections
/// (NIST SRD). Use `BA_RATIO_WATER` for Beyer 1960 comparisons;
/// use `B_OVER_A_WATER` for general acoustic simulations.
///
/// # Reference
/// Beyer, R. T. (1960). "Parameter of nonlinearity in fluids."
/// *J. Acoust. Soc. Am.* 32(6), 719–721. DOI: 10.1121/1.1908195.
pub const BA_RATIO_WATER: f64 = WATER_NONLINEARITY_B_A;

/// Acoustic nonlinearity parameter B/A for soft tissue at 37°C (dimensionless).
///
/// B/A = 6.0 — midpoint of the literature range (5–9) for mammalian soft tissue.
/// The full per-tissue SSOT constants are in
/// [`crate::constants::tissue_acoustics`] (e.g., `B_OVER_A_LIVER` = 6.75,
/// `B_OVER_A_MUSCLE` = 7.4).
///
/// # Reference
/// Hamilton, M. F., & Blackstock, D. T. (1998). *Nonlinear Acoustics*.
/// Academic Press. Table 14.1.
pub const BA_RATIO_TISSUE: f64 = 6.0;

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify DB_TO_NP × NP_TO_DB == 1 to machine precision (round-trip identity).
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_db_np_round_trip() {
        let product = DB_TO_NP * NP_TO_DB;
        assert!(
            (product - 1.0).abs() < 1e-14,
            "DB_TO_NP × NP_TO_DB = {product} (expected 1.0)"
        );
    }

    /// Confirm the exact relationship DB_TO_NP = ln(10)/20.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_db_to_np_exact_value() {
        let exact = std::f64::consts::LN_10 / 20.0;
        assert!(
            (DB_TO_NP - exact).abs() < f64::EPSILON * 10.0,
            "DB_TO_NP = {DB_TO_NP}, expected ln(10)/20 = {exact}"
        );
    }

    /// Confirm NP_TO_DB = 20/ln(10), the exact Np→dB scale factor.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_np_to_db_exact_value() {
        let exact = 20.0 / std::f64::consts::LN_10;
        assert!(
            (NP_TO_DB - exact).abs() < f64::EPSILON * 10.0,
            "NP_TO_DB = {NP_TO_DB}, expected 20/ln(10) = {exact}"
        );
    }

    /// 1 dB amplitude ratio ≈ 1.12202 linear; 1 Np ≈ 2.71828 linear.
    /// The conversion must satisfy: exp(DB_TO_NP) ≈ 10^(1/20).
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_db_to_np_amplitude_consistency() {
        let amp_from_db = 10_f64.powf(1.0 / 20.0); // 10^(1/20)
        let amp_from_np = DB_TO_NP.exp(); // e^(ln(10)/20) = 10^(1/20)
        assert!(
            (amp_from_db - amp_from_np).abs() < 1e-14,
            "Amplitude ratio mismatch: {amp_from_db} vs {amp_from_np}"
        );
    }
}
