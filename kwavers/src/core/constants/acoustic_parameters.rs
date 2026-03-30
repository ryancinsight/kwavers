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

/// Water surface tension at 20°C (N/m)
/// Value: 0.0728
/// Reference: NIST Chemistry `WebBook`
pub const WATER_SURFACE_TENSION_20C: f64 = 0.0728;

/// Water vapor pressure at 20°C (Pa)
/// Value: 2339
/// Reference: Wagner & Pruss (2002). "The IAPWS formulation for water"
pub const WATER_VAPOR_PRESSURE_20C: f64 = 2339.0;

/// Reference frequency for absorption calculations (Hz)
/// Value: 1e6 (1 `MHz`)
/// Standard reference in medical ultrasound
pub const REFERENCE_FREQUENCY_MHZ: f64 = 1e6;

/// Minimum time step for acoustic simulations (s)
/// Value: 1e-10
/// Based on CFL condition for typical ultrasound speeds
pub const MIN_ACOUSTIC_TIME_STEP: f64 = 1e-10;

// MAX_TIME_STEP moved to numerical.rs to avoid duplication

/// Rayleigh collapse time coefficient
/// Value: 0.915
/// Reference: Rayleigh (1917). "On the pressure developed in a liquid"
pub const RAYLEIGH_COLLAPSE_COEFFICIENT: f64 = 0.915;

/// Water specific heat capacity (J/(kg·K))
/// Value: 4182
/// Reference: NIST
pub const WATER_SPECIFIC_HEAT: f64 = 4182.0;

/// Water thermal conductivity at 20°C (W/(m·K))
/// Value: 0.598
/// Reference: NIST
pub const WATER_THERMAL_CONDUCTIVITY: f64 = 0.598;

/// Air specific heat at constant pressure (J/(kg·K))
/// Value: 1005
/// Reference: NIST
pub const AIR_SPECIFIC_HEAT_CP: f64 = 1005.0;

/// Air thermal conductivity at 20°C (W/(m·K))
/// Value: 0.0257
/// Reference: NIST
pub const AIR_THERMAL_CONDUCTIVITY: f64 = 0.0257;

/// Polytropic index for air
/// Value: 1.4 (ratio of specific heats)
/// Reference: Standard thermodynamics
pub const AIR_POLYTROPIC_INDEX: f64 = 1.4;

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

/// Bone sound speed (m/s)
/// Value: 3500 m/s
/// Reference: Hosokawa & Otani (1997). "Ultrasonic wave propagation in bovine cancellous bone"
pub const BONE_SOUND_SPEED: f64 = 3500.0;

/// Bone density (kg/m³)
/// Value: 1900 kg/m³
/// Reference: Duck, F. A. (1990). "Physical properties of tissue"
pub const BONE_DENSITY: f64 = 1900.0;

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
///   20 log₁₀(A₂/A₁) [dB] = 20 ln(A₂/A₁) / ln(10) [dB]
///   ln(A₂/A₁) [Np]
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

/// Water nonlinearity parameter (B/A)
pub const NONLINEARITY_WATER: f64 = 5.2;

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify DB_TO_NP × NP_TO_DB == 1 to machine precision (round-trip identity).
    #[test]
    fn test_db_np_round_trip() {
        let product = DB_TO_NP * NP_TO_DB;
        assert!(
            (product - 1.0).abs() < 1e-14,
            "DB_TO_NP × NP_TO_DB = {product} (expected 1.0)"
        );
    }

    /// Confirm the exact relationship DB_TO_NP = ln(10)/20.
    #[test]
    fn test_db_to_np_exact_value() {
        let exact = std::f64::consts::LN_10 / 20.0;
        assert!(
            (DB_TO_NP - exact).abs() < f64::EPSILON * 10.0,
            "DB_TO_NP = {DB_TO_NP}, expected ln(10)/20 = {exact}"
        );
    }

    /// Confirm NP_TO_DB = 20/ln(10), the exact Np→dB scale factor.
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
    #[test]
    fn test_db_to_np_amplitude_consistency() {
        let amp_from_db = 10_f64.powf(1.0 / 20.0); // 10^(1/20)
        let amp_from_np = DB_TO_NP.exp();            // e^(ln(10)/20) = 10^(1/20)
        assert!(
            (amp_from_db - amp_from_np).abs() < 1e-14,
            "Amplitude ratio mismatch: {amp_from_db} vs {amp_from_np}"
        );
    }
}
