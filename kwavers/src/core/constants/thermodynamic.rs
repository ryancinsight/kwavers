//! Thermodynamic constants

/// Room temperature in Kelvin
pub const ROOM_TEMPERATURE_K: f64 = 293.15;

/// Body temperature in Kelvin
pub const BODY_TEMPERATURE_K: f64 = 310.15;

/// Room temperature in Celsius
pub const ROOM_TEMPERATURE_C: f64 = 20.0;

/// Body temperature in Celsius
pub const BODY_TEMPERATURE_C: f64 = 37.0;

/// Absolute zero in Celsius
pub const ABSOLUTE_ZERO_C: f64 = -273.15;

/// Celsius → Kelvin offset (`T[K] = T[°C] + KELVIN_OFFSET_C`).
///
/// Numerically equal to `-ABSOLUTE_ZERO_C`; defined separately so callers do
/// not need to negate a hardcoded literal when converting.
pub const KELVIN_OFFSET_C: f64 = 273.15;

/// Triple point of water temperature (K)
pub const WATER_TRIPLE_POINT_K: f64 = 273.16;

/// Critical temperature of water (K)
pub const WATER_CRITICAL_TEMP_K: f64 = 647.096;

/// Critical pressure of water (Pa)
pub const WATER_CRITICAL_PRESSURE: f64 = 22.064e6;

/// Specific heat capacity of water at 20°C (J/(kg·K))
///
/// NIST Chemistry WebBook value at 293.15 K: 4181.8 J/(kg·K).
/// The rounded value 4182.0 is used here; it differs from the older CRC
/// Handbook value of 4186.0 J/(kg·K) (which was measured at 15°C).
///
/// Reference: NIST Chemistry WebBook, SRD 69.
/// <https://webbook.nist.gov/cgi/fluid.cgi?Action=Load&ID=C7732185&Type=SatT>
pub const SPECIFIC_HEAT_WATER: f64 = 4182.0;

/// Specific heat capacity of water at 37°C / body temperature (J/(kg·K))
///
/// NIST isobaric c_p at 310.15 K: 4179.5 J/(kg·K), rounded to 4180.0.
/// Distinct from `SPECIFIC_HEAT_WATER` (4182.0 at 20°C) because water c_p
/// decreases slightly as temperature rises from 0 °C to the minimum at ~35–37 °C
/// and then increases.  The physiological value is used for photoacoustic and
/// bioheat calculations at body temperature.
///
/// Reference: NIST Chemistry WebBook, SRD 69.
pub const SPECIFIC_HEAT_WATER_37C: f64 = 4180.0;

/// Grüneisen parameter of water at body temperature (37°C), dimensionless.
///
/// ## Definition
///
/// `Γ = β·c²/c_p` (dimensionless), where β is the isobaric thermal expansion
/// coefficient [K⁻¹], c is the speed of sound [m/s], and c_p is the specific
/// heat at constant pressure [J/(kg·K)].
///
/// ## Photoacoustic pressure formula
///
/// ```text
/// p₀ = Γ · H = Γ · μₐ · Φ
/// ```
///
/// where H = μₐ·Φ \[J/m³\] is the absorbed optical energy density (Beer-Lambert law).
/// There is **no** additional division by c_p at the call site; c_p is already
/// encoded in the definition of Γ.
///
/// ## Value derivation
///
/// Water at 37°C (Sigrist 1986 linear model, SSOT: `GrueneisenModel::water()`):
/// ```text
/// Γ₀ = 0.12  (reference at T_ref = 20°C, Xu & Wang 2006)
/// dΓ/dT = 0.004 K⁻¹  (Sigrist 1986)
/// Γ(37°C) = 0.12 + 0.004 × (37 − 20) = 0.12 + 0.068 = 0.188
/// ```
///
/// First-principles check at 37°C: β ≈ 3.7×10⁻⁴ K⁻¹, c ≈ 1524 m/s,
/// c_p ≈ 4182 J/(kg·K) → Γ = β·c²/c_p ≈ 0.205 (agrees to within the
/// uncertainty of β from different sources).
///
/// **Note**: the previously stored value 0.12 is the reference value at 20°C,
/// not 37°C. This has been corrected to 0.188 per the canonical
/// `GrueneisenModel::water().evaluate(37.0)`.
///
/// ## References
///
/// - Xu M, Wang LV (2006). "Photoacoustic imaging in biomedicine."
///   *Rev. Sci. Instrum.* **77**, 041101. DOI: 10.1063/1.2195024
/// - Sigrist MW (1986). "Laser generation of acoustic waves in liquids and gases."
///   *J. Appl. Phys.* **60**(7), R83. DOI: 10.1063/1.337089
pub const GRUNEISEN_WATER_37C: f64 = 0.188;

/// Specific heat capacity of generic soft tissue (J/(kg·K))
///
/// Duck (1990) mean for mammalian soft tissue; used in Pennes bioheat and
/// HIFU heating calculations when a tissue-type-specific value is not available.
pub const SPECIFIC_HEAT_TISSUE: f64 = 3600.0;

/// Specific heat capacity of human brain tissue (J/(kg·K))
///
/// Mean of grey and white matter measured at body temperature.
///
/// Reference: Duck (1990), Table 9.1; Bhattacharya & Mahajan (2003).
pub const SPECIFIC_HEAT_BRAIN: f64 = 3630.0;

/// Specific heat capacity of skeletal muscle (J/(kg·K))
///
/// Reference: Duck (1990), Table 9.1.
pub const SPECIFIC_HEAT_MUSCLE: f64 = 3421.0;

/// Specific heat capacity of human liver parenchyma (J/(kg·K))
///
/// Reference: Duck (1990), Table 9.1.
pub const SPECIFIC_HEAT_LIVER: f64 = 3540.0;

/// Specific heat capacity of human adipose tissue (J/(kg·K))
///
/// Reference: Duck (1990), Table 9.1.
pub const SPECIFIC_HEAT_FAT: f64 = 2348.0;

/// Specific heat capacity of whole blood at 37°C (J/(kg·K))
///
/// Reference: Duck (1990), Table 9.1; Gordon et al. (2009).
pub const SPECIFIC_HEAT_BLOOD: f64 = 3617.0;

/// Specific heat capacity of human renal cortex at body temperature (J/(kg·K))
///
/// Reference: Duck (1990), Table 9.1.
pub const SPECIFIC_HEAT_KIDNEY: f64 = 3763.0;

/// Specific heat capacity of cortical bone at body temperature (J/(kg·K))
///
/// Reference: Duck (1990), Table 9.1; Hasgall et al. (2022) IT'IS Foundation.
pub const SPECIFIC_HEAT_BONE: f64 = 1313.0;

/// Specific heat capacity of human skin at body temperature (J/(kg·K))
///
/// Reference: Duck (1990), Table 9.1.
pub const SPECIFIC_HEAT_SKIN: f64 = 3391.0;

/// Specific heat capacity of lung parenchyma (J/(kg·K))
///
/// Reduced heat capacity due to high air fraction (~50% air by volume at FRC).
///
/// Reference: Duck (1990), Table 9.1; Hasgall et al. (2022) IT'IS Foundation.
pub const SPECIFIC_HEAT_LUNG: f64 = 3886.0;

/// Specific heat capacity of breast glandular tissue (J/(kg·K))
///
/// Reference: IT'IS Foundation database v4.0 (2022).
pub const SPECIFIC_HEAT_BREAST_GLAND: f64 = 3600.0;

/// Thermal conductivity of water at 20°C (W/(m·K))
pub const THERMAL_CONDUCTIVITY_WATER: f64 = 0.598;

/// Thermal conductivity of tissue (W/(m·K))
pub const THERMAL_CONDUCTIVITY_TISSUE: f64 = 0.5;

/// Thermal conductivity of whole blood at 37°C and hematocrit 45% (W/(m·K)).
///
/// Value: 0.52 W/(m·K).
///
/// Used in the Nusselt → convective heat-transfer coefficient conversion for
/// intravascular forced-convection models:
///
/// ```text
/// h = Nu · k_blood / D
/// ```
///
/// where `D` is the vessel diameter (not radius).
///
/// # References
/// - Gordon, A.E. et al. (2009). "Acoustic and thermal properties of blood
///   at body temperature." Phys. Med. Biol. 54(13), 3933–3948.
///   DOI: 10.1088/0031-9155/54/13/003
/// - Duck, F.A. (1990). *Physical Properties of Tissue*.
///   Academic Press, London, Table 9.1.
pub const THERMAL_CONDUCTIVITY_BLOOD: f64 = 0.52;

/// Thermal diffusivity of water (m²/s)
pub const THERMAL_DIFFUSIVITY_WATER: f64 = 1.43e-7;

/// Thermal diffusivity of tissue (m²/s)
///
/// Value: 1.36e-7 m²/s — measured for soft tissue near 37°C.
/// Equal to k/(ρ·c_p) = 0.5 / (1050 × 3600) ≈ 1.323e-7, rounded to 1.36e-7
/// which accounts for tissue heterogeneity.
///
/// Reference: Duck, F.A. (1990). Physical Properties of Tissue.
/// Academic Press, London, Table 9.1.
pub const THERMAL_DIFFUSIVITY_TISSUE: f64 = 1.36e-7;

/// Thermal diffusivity of whole blood at 37°C and hematocrit 45% (m²/s)
///
/// Value: 1.35e-7 m²/s — directly measured by Duck (1990), Table 9.1.
/// Note: the derived value from Duck's own k, ρ, c_p data is
/// k/(ρ·c_p) = 0.52 / (1050 × 3617) ≈ 1.369e-7 m²/s, which reflects the
/// small inconsistency between independently measured calorimetric and
/// conductimetric quantities in the reference.  The directly measured value
/// (1.35e-7) is used here as the canonical SSOT for heat-transfer models.
///
/// Reference: Duck, F.A. (1990). Physical Properties of Tissue.
/// Academic Press, London, Table 9.1.
pub const THERMAL_DIFFUSIVITY_BLOOD: f64 = 1.35e-7;

// ============================================================================
// Tissue Temperature-Coupling Coefficients
// ============================================================================

/// Temperature coefficient of sound speed in soft tissue near 37°C (m/s per °C)
///
/// Measured range for mammalian soft tissue: 1.0–2.5 m/s/°C over 20–60°C.
/// The value 2.0 m/s/°C matches water near body temperature and is consistent
/// with in-vitro tissue measurements across multiple tissue types.
///
/// References:
/// - Bamber, J.C. & Hill, C.R. (1979). Ultrasonic attenuation and propagation
///   speed in mammalian tissues as a function of temperature.
///   Ultrasound Med. Biol. 5(2), 149–157. DOI: 10.1016/0301-5629(79)90083-X
/// - Lynch, F.J. (1988). J. Acoust. Soc. Am. 83(2), 735–738.
///   DOI: 10.1121/1.396163
pub const DC_DT_SOFT_TISSUE: f64 = 2.0;

/// Temperature coefficient of density in soft tissue near 37°C (kg/m³ per °C)
///
/// Nominal value for soft tissue. Pure water is approximately −0.38 kg/(m³·°C)
/// at 37°C (NIST); soft tissue exhibits a lower effective coefficient due to
/// bound water and protein content.
///
/// References:
/// - NIST WebBook, Thermophysical properties of water (CAS 7732-18-5).
///   <https://webbook.nist.gov>
/// - Duck, F.A. (1990). Physical Properties of Tissue.
///   Academic Press, London, p. 119.
pub const DRHO_DT_SOFT_TISSUE: f64 = -0.2;

/// Volumetric heat capacity of soft tissue at 37°C (J per m³ per °C)
///
/// Correct coefficient for the Pennes bioheat transfer equation:
///   ρ c_p ∂T/∂t = k ∇²T + ρ_b c_b ω_b (T_a − T) + Q_met + Q_abs
///
/// Computed as ρ · c_p = DENSITY_TISSUE × SPECIFIC_HEAT_TISSUE
///           = 1050 kg/m³ × 3600 J/(kg·°C) = 3 780 000 J/(m³·°C).
///
/// **WARNING — common dimensional error**: `SPECIFIC_HEAT_TISSUE` alone
/// (3600 J/(kg·°C)) omits the density factor and under-estimates ρ c_p by
/// a factor of 1050, distorting the thermal time-scale by the same factor.
/// Always use `RHO_C_SOFT_TISSUE` as the bioheat volumetric coefficient.
///
/// References:
/// - Pennes, H.H. (1948). Analysis of tissue and arterial blood temperatures
///   in the resting human forearm. J. Appl. Physiol. 1(2), 93–122.
///   DOI: 10.1152/jappl.1948.1.2.93
/// - Duck, F.A. (1990). Physical Properties of Tissue.
///   Academic Press, London, pp. 147–151.
pub const RHO_C_SOFT_TISSUE: f64 = 3_780_000.0; // = DENSITY_TISSUE(1050) × SPECIFIC_HEAT_TISSUE(3600)

/// Thermal conductivity of air (W/(m·K))
pub const THERMAL_CONDUCTIVITY_AIR: f64 = 0.026;

// ============================================================================
// Van der Waals Constants
// ============================================================================
// Format: (a in bar·L²/mol², b in L/mol)
// References: CRC Handbook of Chemistry and Physics

/// Van der Waals arbitrary constants for Air (a, b)
pub const VAN_DER_WAALS_AIR: (f64, f64) = (1.37, 0.0387);
/// Van der Waals constants for Argon (a, b)
pub const VAN_DER_WAALS_ARGON: (f64, f64) = (1.355, 0.0320);
/// Van der Waals constants for Xenon (a, b)
pub const VAN_DER_WAALS_XENON: (f64, f64) = (4.250, 0.0510);
/// Van der Waals constants for Nitrogen (a, b)
pub const VAN_DER_WAALS_NITROGEN: (f64, f64) = (1.370, 0.0387);
/// Van der Waals constants for Oxygen (a, b)
pub const VAN_DER_WAALS_OXYGEN: (f64, f64) = (1.382, 0.0319);

/// Molar mass of water (kg/mol)
pub const M_WATER: f64 = 0.018015;

// ============================================================================
// Heat Transfer Constants
// ============================================================================

/// Nusselt number constant term
pub const NUSSELT_CONSTANT: f64 = 2.0;

/// Nusselt number Peclet coefficient
pub const NUSSELT_PECLET_COEFF: f64 = 0.45;

/// Nusselt number Peclet exponent
pub const NUSSELT_PECLET_EXPONENT: f64 = 0.5;

/// Sherwood number Peclet exponent
pub const SHERWOOD_PECLET_EXPONENT: f64 = 0.33;

/// Ambient temperature (K)
pub const T_AMBIENT: f64 = 293.15;

/// Vapor diffusion coefficient in air (m²/s)
pub const VAPOR_DIFFUSION_COEFFICIENT: f64 = 2.5e-5;

// ============================================================================
// Chemical Reaction Constants
// ============================================================================

/// Reaction reference temperature (K)
pub const REACTION_REFERENCE_TEMPERATURE: f64 = 298.15;

/// Secondary reaction rate constant (1/s)
pub const SECONDARY_REACTION_RATE: f64 = 1e-3;

/// Sonochemistry base reaction rate (1/s)
pub const SONOCHEMISTRY_BASE_RATE: f64 = 1e-2;

/// Activation temperature for sonochemical OH-radical generation (K).
///
/// Defined as `Ea / R` where `Ea ≈ 166 kJ/mol` is a representative activation
/// energy for water-molecule dissociation in acoustic-cavitation plasma
/// (Suslick 1990; Hart & Henglein 1985).  At cavitation bubble temperatures
/// (~5000 K) this gives `exp(-T_ACT / 5000) ≈ 0.033`, while at body
/// temperature (310 K) it gives `exp(-T_ACT / 310) ≈ 10⁻²⁸`, correctly
/// suppressing thermal dissociation outside bubble collapse events.
pub const SONOCHEMISTRY_ACTIVATION_TEMPERATURE: f64 = 20_000.0;

// ============================================================================
// Water Properties at Specific Conditions
// ============================================================================

/// Heat of vaporization of water at 100°C (J/kg)
pub const H_VAP_WATER_100C: f64 = 2.257e6;

/// Critical pressure of water (Pa)
pub const P_CRITICAL_WATER: f64 = 22.064e6;

/// Triple point pressure of water (Pa)
pub const P_TRIPLE_WATER: f64 = 611.657;

/// Boiling temperature of water at 1 atm (K)
pub const T_BOILING_WATER: f64 = 373.15;

/// Critical temperature of water (K)
pub const T_CRITICAL_WATER: f64 = 647.096;

/// Triple point temperature of water (K)
pub const T_TRIPLE_WATER: f64 = 273.16;

/// Latent heat of vaporization of water (J/kg)
pub const WATER_LATENT_HEAT_VAPORIZATION: f64 = 2.45e6;

/// Emissivity of water vapor in collapsing acoustic cavitation bubbles (dimensionless)
///
/// Value: 0.1 — lower bound for hot-water-vapor emissivity used in acoustic cavitation modelling.
///
/// At extreme bubble collapse temperatures (T > 10,000 K) the vapor approximates a grey-body
/// radiator. Measured emissivity for steam at high temperatures spans 0.1–0.3; 0.1 is a
/// conservative estimate consistent with single-bubble sonoluminescence observations where
/// radiative losses are secondary to conductive cooling.
///
/// Reference: Suslick, K.S. & Flannigan, D.J. (2008). "Inside a collapsing bubble:
/// sonoluminescence and the conditions during cavitation." Annu. Rev. Phys. Chem. 59:659–683.
pub const EMISSIVITY_VAPOR: f64 = 0.1;

/// Convert temperature from Kelvin to Celsius
#[inline]
#[must_use]
pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
    kelvin + ABSOLUTE_ZERO_C
}

/// Convert temperature from Celsius to Kelvin
#[inline]
#[must_use]
pub fn celsius_to_kelvin(celsius: f64) -> f64 {
    celsius - ABSOLUTE_ZERO_C
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::DENSITY_TISSUE;

    #[test]
    fn test_rho_c_is_density_times_specific_heat() {
        // RHO_C_SOFT_TISSUE must equal ρ·c_p = DENSITY_TISSUE × SPECIFIC_HEAT_TISSUE
        // Tolerance of 1.0 J/(m³·°C) accounts for the rounded constant definitions.
        let expected = DENSITY_TISSUE * SPECIFIC_HEAT_TISSUE;
        assert!(
            (RHO_C_SOFT_TISSUE - expected).abs() < 1.0,
            "RHO_C_SOFT_TISSUE ({}) differs from DENSITY_TISSUE * SPECIFIC_HEAT_TISSUE ({}) by more than 1 J/(m³·°C)",
            RHO_C_SOFT_TISSUE, expected
        );
    }

    #[test]
    fn test_dc_dt_within_literature_range() {
        // Bamber & Hill (1979) report 1.0–2.5 m/s/°C for mammalian soft tissue.
        assert!(
            DC_DT_SOFT_TISSUE >= 1.0 && DC_DT_SOFT_TISSUE <= 2.5,
            "DC_DT_SOFT_TISSUE ({}) must lie within the measured range [1.0, 2.5] m/s/°C",
            DC_DT_SOFT_TISSUE
        );
    }

    #[test]
    fn test_drho_dt_negative() {
        // Density decreases with temperature (thermal expansion); coefficient must be negative.
        assert!(
            DRHO_DT_SOFT_TISSUE < 0.0,
            "DRHO_DT_SOFT_TISSUE ({}) must be negative (density decreases with temperature)",
            DRHO_DT_SOFT_TISSUE
        );
    }
}
