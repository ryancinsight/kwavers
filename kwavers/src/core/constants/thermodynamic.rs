//! Fundamental thermodynamic constants: temperature references, Grüneisen
//! parameters, water heat capacities, phase transition data, and van der Waals
//! coefficients.
//!
//! Tissue- and fluid-specific thermal properties live in [`super::tissue_thermal`].

// ── Thermodynamic invariants ──────────────────────────────────────────────────

/// Adiabatic index (heat capacity ratio) of diatomic ideal gases (Air, N₂, O₂) [-].
///
/// γ = Cp/Cv = 7/5 = 1.4. Reference: NIST; CRC Handbook, 104th ed., §2.
pub const HEAT_CAPACITY_RATIO_DIATOMIC: f64 = 1.4;

/// Adiabatic index (heat capacity ratio) of monatomic ideal gases (Ar, Xe, He, Ne) [-].
///
/// γ = Cp/Cv = 5/3 ≈ 1.6667. Reference: NIST; CRC Handbook, 104th ed., §2.
pub const HEAT_CAPACITY_RATIO_MONATOMIC: f64 = 5.0 / 3.0;

// ── Temperature references ────────────────────────────────────────────────────

/// Room temperature in Kelvin.
pub const ROOM_TEMPERATURE_K: f64 = 293.15;

/// Bubble-dynamics reference temperature [K].
///
/// Sonoluminescence and Cherenkov emission models initialize the bubble interior
/// at 300 K (≈ 27°C) per Brenner et al. (2002) Rev. Mod. Phys. 74(2):425–484.
/// Distinct from `ROOM_TEMPERATURE_K` (293.15 K = 20°C).
pub const BUBBLE_REFERENCE_TEMPERATURE_K: f64 = 300.0;

/// Body temperature in Kelvin.
pub const BODY_TEMPERATURE_K: f64 = 310.15;

/// Room temperature in Celsius.
pub const ROOM_TEMPERATURE_C: f64 = 20.0;

/// Body temperature in Celsius.
pub const BODY_TEMPERATURE_C: f64 = 37.0;

/// Arterial blood temperature in Celsius (dimensionless-equivalent to body temperature).
///
/// Physiological convention: arterial blood equilibrates with core body temperature
/// at 37.0°C. Numerically identical to `BODY_TEMPERATURE_C`; provided as a distinct
/// named constant to make the physical quantity explicit at Pennes bioheat call sites
/// (`T_b` in `ω_b · ρ_b · c_b · (T_b − T)`).
///
/// Reference: Pennes, H. H. (1948). "Analysis of tissue and arterial blood temperatures
/// in the resting human forearm." *J. Appl. Physiol.* 1(2), 93–122.
pub const BLOOD_TEMPERATURE_C: f64 = BODY_TEMPERATURE_C;

/// Absolute zero in Celsius.
pub const ABSOLUTE_ZERO_C: f64 = -273.15;

/// Celsius → Kelvin offset (`T[K] = T[°C] + KELVIN_OFFSET_C`).
pub const KELVIN_OFFSET_C: f64 = 273.15;

/// Triple point of water temperature (K).
pub const WATER_TRIPLE_POINT_K: f64 = 273.16;

/// Critical temperature of water (K).
pub const WATER_CRITICAL_TEMP_K: f64 = 647.096;

/// Critical pressure of water (Pa).
pub const WATER_CRITICAL_PRESSURE: f64 = 22.064e6;

// ── Water heat capacities ─────────────────────────────────────────────────────

/// Specific heat capacity of water at 20°C (J/(kg·K)).
///
/// NIST Chemistry WebBook at 293.15 K: 4181.8 J/(kg·K), rounded to 4182.0.
/// Reference: NIST Chemistry WebBook, SRD 69.
pub const SPECIFIC_HEAT_WATER: f64 = 4182.0;

/// Specific heat capacity of water at 37°C / body temperature (J/(kg·K)).
///
/// NIST isobaric c_p at 310.15 K: 4179.5 J/(kg·K), rounded to 4180.0.
/// Reference: NIST Chemistry WebBook, SRD 69.
pub const SPECIFIC_HEAT_WATER_37C: f64 = 4180.0;

// ── Water thermal conductivities ──────────────────────────────────────────────

/// Thermal conductivity of water at 20°C (W/(m·K)).
pub const THERMAL_CONDUCTIVITY_WATER: f64 = 0.598;

/// Thermal conductivity of water at 37°C / body temperature (W/(m·K)).
///
/// NIST value at 310.15 K: 0.6233 W/(m·K), rounded to 0.623.
/// Reference: NIST Chemistry WebBook, SRD 69.
pub const THERMAL_CONDUCTIVITY_WATER_37C: f64 = 0.623;

// ── Grüneisen parameters ──────────────────────────────────────────────────────

/// Grüneisen parameter of water at body temperature (37°C), dimensionless.
///
/// Γ(37°C) = 0.12 + 0.004 × (37 − 20) = 0.188 (Sigrist 1986 linear model).
///
/// Reference: Xu & Wang (2006) Rev. Sci. Instrum. 77, 041101;
/// Sigrist MW (1986) J. Appl. Phys. 60(7), R83.
pub const GRUNEISEN_WATER_37C: f64 = 0.188;

/// Grüneisen parameter of liquid water at 20°C (dimensionless).
///
/// Reference value at 20°C from Xu & Wang (2006).
/// Use `GRUNEISEN_WATER_37C` (0.188) for 37°C simulations.
pub const GRUNEISEN_WATER_20C: f64 = 0.12;

/// Grüneisen parameter of generic soft tissue at 37°C (dimensionless).
///
/// Reference: Wang & Wu (2007) Biomedical Optics, Wiley-Interscience, p. 287;
/// Duck FA (1990) Physical Properties of Tissue.
pub const GRUNEISEN_SOFT_TISSUE: f64 = 0.15;

/// Linear temperature coefficient dΓ/dT for liquid water [K⁻¹].
///
/// Γ_water(T) = GRUNEISEN_WATER_20C + GRUNEISEN_WATER_TEMP_COEFF·(T − GRUNEISEN_WATER_T_REF_C)
///
/// Reference: Sigrist MW (1986). J. Appl. Phys. 60(7), R83. DOI: 10.1063/1.337089
pub const GRUNEISEN_WATER_TEMP_COEFF: f64 = 0.004; // K⁻¹

/// Reference temperature for the linear water Grüneisen model [°C].
///
/// GRUNEISEN_WATER_20C applies at this temperature. At 37°C the model yields
/// GRUNEISEN_WATER_37C = 0.188.
///
/// Reference: Sigrist MW (1986). J. Appl. Phys. 60(7), R83.
pub const GRUNEISEN_WATER_T_REF_C: f64 = 20.0; // °C

/// Linear temperature coefficient dΓ/dT for generic soft tissue [K⁻¹].
///
/// Γ_tissue(T) = GRUNEISEN_SOFT_TISSUE + GRUNEISEN_SOFT_TISSUE_TEMP_COEFF·(T − BODY_TEMPERATURE_C)
///
/// Reference: Xu M, Wang LV (2006). Rev. Sci. Instrum. 77, 041101. DOI: 10.1063/1.2195024
pub const GRUNEISEN_SOFT_TISSUE_TEMP_COEFF: f64 = 0.003; // K⁻¹

// ── Thermal expansion coefficients ────────────────────────────────────────────

/// Isobaric thermal expansion coefficient of liquid water at 20°C (K⁻¹).
///
/// Reference: NIST Chemistry WebBook, SRD 69.
pub const THERMAL_EXPANSION_WATER_20C: f64 = 2.07e-4;

/// Isobaric thermal expansion coefficient of dry air at 20°C (K⁻¹).
///
/// β = 1/T = 1/293.15 ≈ 3.41×10⁻³ K⁻¹. Reference: NIST, ideal gas approximation.
pub const THERMAL_EXPANSION_AIR_20C: f64 = 3.43e-3;

/// Thermal conductivity of dry air at 20°C, 1 atm (W/(m·K)).
///
/// Reference: NIST Chemistry WebBook, SRD 69.
pub const THERMAL_CONDUCTIVITY_AIR: f64 = 0.0257;

// ── Van der Waals constants ───────────────────────────────────────────────────
// Format: (a in bar·L²/mol², b in L/mol)
// References: CRC Handbook of Chemistry and Physics.

/// Van der Waals constants for Air (a, b).
pub const VAN_DER_WAALS_AIR: (f64, f64) = (1.37, 0.0387);
/// Van der Waals constants for Argon (a, b).
pub const VAN_DER_WAALS_ARGON: (f64, f64) = (1.355, 0.0320);
/// Van der Waals constants for Xenon (a, b).
pub const VAN_DER_WAALS_XENON: (f64, f64) = (4.250, 0.0510);
/// Van der Waals constants for Nitrogen (a, b).
pub const VAN_DER_WAALS_NITROGEN: (f64, f64) = (1.370, 0.0387);
/// Van der Waals constants for Oxygen (a, b).
pub const VAN_DER_WAALS_OXYGEN: (f64, f64) = (1.382, 0.0319);

/// Molar mass of water (kg/mol).
pub const M_WATER: f64 = 0.018015;

// ── Heat transfer / mass transport ────────────────────────────────────────────

/// Dittus-Boelter turbulent-pipe-flow Nusselt correlation coefficient [-].
///
/// `Nu = DITTUS_BOELTER_COEFFICIENT · Re^0.8 · Pr^0.4`
///
/// Valid for Re > 10 000, 0.7 < Pr < 160 (conservative extension to Re ≥ 2300).
/// Reference: Dittus FW, Boelter LMK (1930). Univ. Calif. Publ. Eng. 2(13):443–461.
/// Also: Incropera FP, DeWitt DP (2007). Fundamentals of Heat and Mass Transfer, 6th ed.,
/// §8.5 (Eq. 8.60).
pub const DITTUS_BOELTER_COEFFICIENT: f64 = 0.023;

/// Dittus-Boelter velocity exponent (n in Re^n) [-].
///
/// Standard value n = 0.8 for both heating and cooling (approximate).
/// Reference: Dittus & Boelter 1930; Incropera & DeWitt 2007, §8.5.
pub const DITTUS_BOELTER_VELOCITY_EXPONENT: f64 = 0.8;

/// Dittus-Boelter Prandtl exponent for heating (n in Pr^n) [-].
///
/// n = 0.4 when the fluid is being heated (T_wall > T_bulk).
/// Reference: Dittus & Boelter 1930; Incropera & DeWitt 2007, §8.5.
pub const DITTUS_BOELTER_PRANDTL_EXPONENT_HEATING: f64 = 0.4;

/// Laminar fully-developed-pipe-flow Nusselt number (constant wall temperature) [-].
///
/// Nu₀ = 3.66 (Graetz solution for thermally developing flow at uniform wall temperature,
/// Pr → 0 limit gives Nu₀ = 3.657 ≈ 3.66).
/// Reference: Sieder EN, Tate GE (1936). Ind. Eng. Chem. 28(12):1429–1435;
/// Incropera & DeWitt 2007, §8.4.1.
pub const NUSSELT_LAMINAR_PIPE_CONST_TEMP: f64 = 3.66;

/// Reynolds number transition from laminar to turbulent flow in a smooth pipe [-].
///
/// Re_crit ≈ 2300 (Osborne Reynolds 1883 critical threshold).
/// Reference: Reynolds O (1883). Phil. Trans. R. Soc. Lond. 174:935–982.
pub const REYNOLDS_LAMINAR_TURBULENT_THRESHOLD: f64 = 2300.0;

/// Nusselt number constant term.
pub const NUSSELT_CONSTANT: f64 = 2.0;

/// Nusselt number Peclet coefficient.
pub const NUSSELT_PECLET_COEFF: f64 = 0.45;

/// Nusselt number Peclet exponent.
pub const NUSSELT_PECLET_EXPONENT: f64 = 0.5;

/// Sherwood number Peclet exponent.
pub const SHERWOOD_PECLET_EXPONENT: f64 = 0.33;

/// Ambient temperature (K).
pub const T_AMBIENT: f64 = 293.15;

/// Vapor diffusion coefficient in air (m²/s).
pub const VAPOR_DIFFUSION_COEFFICIENT: f64 = 2.5e-5;

// ── Chemical reaction constants ───────────────────────────────────────────────

/// Reaction reference temperature (K).
pub const REACTION_REFERENCE_TEMPERATURE: f64 = 298.15;

/// Secondary reaction rate constant (1/s).
pub const SECONDARY_REACTION_RATE: f64 = 1e-3;

/// Sonochemistry base reaction rate (1/s).
pub const SONOCHEMISTRY_BASE_RATE: f64 = 1e-2;

/// Activation temperature for sonochemical OH-radical generation (K).
///
/// Ea / R where Ea ≈ 166 kJ/mol (water-molecule dissociation in cavitation plasma).
/// Reference: Suslick (1990); Hart & Henglein (1985).
pub const SONOCHEMISTRY_ACTIVATION_TEMPERATURE: f64 = 20_000.0;

// ── Water properties at specific conditions ───────────────────────────────────

/// Heat of vaporization of water at 100°C (J/kg).
pub const H_VAP_WATER_100C: f64 = 2.257e6;

/// Critical pressure of water (Pa).
pub const P_CRITICAL_WATER: f64 = 22.064e6;

/// Triple point pressure of water (Pa).
pub const P_TRIPLE_WATER: f64 = 611.657;

/// Boiling temperature of water at 1 atm (K).
pub const T_BOILING_WATER: f64 = 373.15;

/// Critical temperature of water (K).
pub const T_CRITICAL_WATER: f64 = 647.096;

/// Triple point temperature of water (K).
pub const T_TRIPLE_WATER: f64 = 273.16;

/// Latent heat of vaporization of water (J/kg).
pub const WATER_LATENT_HEAT_VAPORIZATION: f64 = 2.45e6;

// ── Antoine equation coefficients for water (log₁₀-mmHg form) ────────────────
// Reference: Stull (1947), Ind. Eng. Chem. 39(4):517–540.
// Valid range: 1–100 °C.
// log₁₀(P_mmHg) = A − B / (C + T_celsius)

/// Antoine A coefficient for water vapor pressure (dimensionless).
pub const WATER_ANTOINE_A: f64 = 8.07131;

/// Antoine B coefficient for water vapor pressure (°C).
pub const WATER_ANTOINE_B: f64 = 1730.63;

/// Antoine C coefficient for water vapor pressure (°C).
pub const WATER_ANTOINE_C: f64 = 233.426;

/// Emissivity of water vapor in collapsing acoustic cavitation bubbles (dimensionless).
///
/// Value: 0.1 — lower bound per single-bubble sonoluminescence observations.
/// Reference: Suslick & Flannigan (2008) Annu. Rev. Phys. Chem. 59:659–683.
pub const EMISSIVITY_VAPOR: f64 = 0.1;

// ── Temperature conversion functions ─────────────────────────────────────────

/// Convert temperature from Kelvin to Celsius.
#[inline]
#[must_use]
pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
    kelvin + ABSOLUTE_ZERO_C
}

/// Convert temperature from Celsius to Kelvin.
#[inline]
#[must_use]
pub fn celsius_to_kelvin(celsius: f64) -> f64 {
    celsius - ABSOLUTE_ZERO_C
}

#[cfg(test)]
mod tests {
    use crate::core::constants::fundamental::DENSITY_TISSUE;
    use crate::core::constants::tissue_thermal::{
        DC_DT_SOFT_TISSUE, DRHO_DT_SOFT_TISSUE, RHO_C_SOFT_TISSUE, SPECIFIC_HEAT_TISSUE,
    };

    #[test]
    fn test_rho_c_is_density_times_specific_heat() {
        let expected = DENSITY_TISSUE * SPECIFIC_HEAT_TISSUE;
        assert!(
            (RHO_C_SOFT_TISSUE - expected).abs() < 1.0,
            "RHO_C_SOFT_TISSUE ({}) differs from DENSITY_TISSUE * SPECIFIC_HEAT_TISSUE ({}) by more than 1 J/(m³·°C)",
            RHO_C_SOFT_TISSUE, expected
        );
    }

    #[test]
    fn test_dc_dt_within_literature_range() {
        assert!(
            DC_DT_SOFT_TISSUE >= 1.0 && DC_DT_SOFT_TISSUE <= 2.5,
            "DC_DT_SOFT_TISSUE ({}) must lie within the measured range [1.0, 2.5] m/s/°C",
            DC_DT_SOFT_TISSUE
        );
    }

    #[test]
    fn test_drho_dt_negative() {
        assert!(
            DRHO_DT_SOFT_TISSUE < 0.0,
            "DRHO_DT_SOFT_TISSUE ({}) must be negative (density decreases with temperature)",
            DRHO_DT_SOFT_TISSUE
        );
    }
}
