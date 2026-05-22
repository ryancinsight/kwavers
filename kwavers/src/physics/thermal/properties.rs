//! Temperature-dependent thermal properties.
//!
//! # Scope
//!
//! This module provides local scalar update laws for thermal therapy and
//! thermometry workflows that need stable property updates from a reference
//! state. It does not replace tissue-specific property tables; base values
//! remain supplied by [`ThermalPropertyData`] or acoustic media records.
//!
//! # Theorem: reference-state invariance
//!
//! Each update law has the form `x(T) = x0 * f(T - T0)` or `x(T) = x0 + g(T - T0)`,
//! where `T0 = 37 °C`, `f(0) = 1`, and `g(0) = 0`. Therefore the update is
//! identity-preserving at reference temperature: `x(T0) = x0`. This prevents
//! drift when a solver rebuilds coefficients at body temperature.
//!
//! # Theorem: absorption positivity
//!
//! The acoustic absorption law is `alpha(T) = alpha0 * exp(gamma * max(T - T0, 0))`
//! with `gamma >= 0`. If `alpha0 >= 0`, then `exp(_) > 0`, hence
//! `alpha(T) >= 0` for all finite `T`. This removes the prior linear law's
//! negative-absorption failure mode during ablation-range heating.
//!
//! # References
//!
//! - Duck, F.A. (1990). *Physical Properties of Tissue*. Academic Press.
//! - Bamber, J.C. & Hill, C.R. (1979). Acoustic properties of normal and cancerous
//!   human liver. *Ultrasound in Medicine & Biology*, 5(2), 149-157.
//! - IT'IS Foundation Tissue Properties Database, version 4.2.
//! - Gachouch, O. et al. (2025). A novel ultrasound thermometry method based on
//!   thermal strain and short and constant acoustic bursts. *Sensors*, 25(2), 385.

use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
use crate::domain::medium::properties::ThermalPropertyData;

/// Soft-tissue thermal-conductivity coefficient [1/°C].
///
/// A small positive coefficient preserves the IT'IS/Duck reference value while
/// allowing moderate hyperthermia updates without changing tissue identity.
const THERMAL_CONDUCTIVITY_COEFF_PER_C: f64 = 0.002;

/// Soft-tissue specific-heat coefficient [1/°C].
const SPECIFIC_HEAT_COEFF_PER_C: f64 = 0.001;

/// Soft-tissue sound-speed coefficient [1/°C].
///
/// Duck/Szabo tissue coefficients are commonly represented near body
/// temperature by `dc/(c dT) ≈ 1.6e-3`. Recent ultrasound thermometry papers
/// still treat the hyperthermia range as locally linear, while warning that
/// ablation temperatures require tissue-specific validation.
const SOFT_TISSUE_SOUND_SPEED_COEFF_PER_C: f64 = 0.0016;

/// Pre-coagulation soft-tissue absorption coefficient [1/°C].
const SOFT_TISSUE_ABSORPTION_COEFF_PER_C: f64 = 0.015;

/// Temperature-dependent thermal conductivity
///
/// # Formula
///
/// `k(T) = k0 * (1 + beta_k * (T - 37 °C))`.
///
/// # Proof of reference invariance
///
/// At `T = 37 °C`, the multiplier is `1 + beta_k * 0 = 1`, so `k(37) = k0`.
///
/// # Arguments
///
/// * `k0` - Base thermal conductivity (W/m/K)
/// * `temperature` - Current temperature (°C)
#[must_use]
pub fn conductivity_vs_temperature(k0: f64, temperature: f64) -> f64 {
    k0 * THERMAL_CONDUCTIVITY_COEFF_PER_C.mul_add(temperature - BODY_TEMPERATURE_C, 1.0)
}

/// Temperature-dependent specific heat
///
/// # Formula
///
/// `c_p(T) = c_p0 * (1 + beta_cp * (T - 37 °C))`.
///
/// # Proof of reference invariance
///
/// At `T = 37 °C`, the multiplier is one, so `c_p(37) = c_p0`.
///
/// # Arguments
///
/// * `c0` - Base specific heat capacity (J/kg/K)
/// * `temperature` - Current temperature (°C)
#[must_use]
pub fn specific_heat_vs_temperature(c0: f64, temperature: f64) -> f64 {
    c0 * SPECIFIC_HEAT_COEFF_PER_C.mul_add(temperature - BODY_TEMPERATURE_C, 1.0)
}

/// Temperature-dependent blood perfusion
///
/// Models physiological response to temperature:
/// - Cold: Reduced perfusion (vasoconstriction)
/// - Mild heating (37-42°C): Increased perfusion (vasodilation)
/// - High heating (42-50°C): Decreasing perfusion (approaching shutdown)
/// - Above 50°C: Vascular shutdown
///
/// # Arguments
///
/// * `w_b0` - Base blood perfusion rate (kg/m³/s)
/// * `temperature` - Current temperature (°C)
#[must_use]
pub fn perfusion_vs_temperature(w_b0: f64, temperature: f64) -> f64 {
    if temperature < BODY_TEMPERATURE_C {
        // Reduced perfusion when cold
        w_b0 * (0.5 + 0.5 * temperature / BODY_TEMPERATURE_C)
    } else if temperature < 42.0 {
        // Increased perfusion with mild heating (vasodilation)
        w_b0 * (1.0 + 0.3 * (temperature - BODY_TEMPERATURE_C) / 5.0)
    } else if temperature < 50.0 {
        // Decreasing perfusion approaching shutdown
        w_b0 * (1.3 - 1.3 * (temperature - 42.0) / 8.0)
    } else {
        // Vascular shutdown
        0.0
    }
}

/// Update thermal properties based on temperature
///
/// Returns a new `ThermalPropertyData` with temperature-corrected values.
/// Preserves blood-specific heat as it is less temperature-sensitive.
///
/// # Arguments
///
/// * `base_properties` - Base thermal properties at reference temperature
/// * `temperature` - Current temperature (°C)
/// # Panics
/// - Panics if `Temperature-updated properties should be valid if base properties are valid`.
///
#[must_use]
pub fn update_properties(
    base_properties: &ThermalPropertyData,
    temperature: f64,
) -> ThermalPropertyData {
    let new_perfusion = base_properties
        .blood_perfusion
        .map(|w_b| perfusion_vs_temperature(w_b, temperature));

    ThermalPropertyData::new(
        conductivity_vs_temperature(base_properties.conductivity, temperature),
        specific_heat_vs_temperature(base_properties.specific_heat, temperature),
        base_properties.density, // Density assumed constant over typical temperature ranges
        new_perfusion,
        base_properties.blood_specific_heat, // Blood specific heat relatively constant
    )
    .expect("Temperature-updated properties should be valid if base properties are valid")
}

/// Acoustic absorption coefficient temperature dependence
///
/// # Formula
///
/// `alpha(T) = alpha0 * exp(gamma * max(T - 37 °C, 0))`.
///
/// `gamma = 0.015 1/°C` matches the soft-tissue coefficient already used by the
/// bioheat absorption model and is consistent with pre-coagulation liver
/// ultrasound measurements. The `max` term leaves sub-body cooling neutral
/// rather than inventing a tissue-specific low-temperature law.
///
/// # Proof of positivity
///
/// For `alpha0 >= 0`, `exp(gamma * max(T - 37, 0)) > 0`; therefore
/// `alpha(T) >= 0` for every finite temperature. At `T = 37 °C`, the exponent is
/// zero and `alpha(37) = alpha0`.
///
/// # Arguments
///
/// * `alpha0` - Base absorption coefficient (Np/m)
/// * `temperature` - Current temperature (°C)
#[must_use]
pub fn absorption_vs_temperature(alpha0: f64, temperature: f64) -> f64 {
    alpha0
        * (SOFT_TISSUE_ABSORPTION_COEFF_PER_C * (temperature - BODY_TEMPERATURE_C).max(0.0))
            .exp()
}

/// Sound speed temperature dependence
///
/// # Formula
///
/// `c(T) = c0 * (1 + beta_c * (T - 37 °C))`, with
/// `beta_c = 1.6e-3 1/°C` for generic soft tissue near body temperature.
///
/// # Proof of reference invariance and monotonicity
///
/// At `T = 37 °C`, `c(T) = c0`. For `c0 > 0` and `beta_c > 0`,
/// `dc/dT = c0 * beta_c > 0`, so sound speed increases monotonically within the
/// local hyperthermia range where the linear thermal-strain model is valid.
///
/// # Arguments
///
/// * `c0` - Base sound speed (m/s)
/// * `temperature` - Current temperature (°C)
#[must_use]
pub fn sound_speed_vs_temperature(c0: f64, temperature: f64) -> f64 {
    c0 * SOFT_TISSUE_SOUND_SPEED_COEFF_PER_C.mul_add(temperature - BODY_TEMPERATURE_C, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_dependent_properties() {
        let base = ThermalPropertyData::soft_tissue();

        // Test at elevated temperature
        let props_45 = update_properties(&base, 45.0);

        // Conductivity should increase
        assert!(props_45.conductivity > base.conductivity);

        // Specific heat should increase slightly
        assert!(props_45.specific_heat > base.specific_heat);

        // Perfusion should decrease (approaching shutdown)
        let base_perfusion = base.blood_perfusion.unwrap();
        let new_perfusion = props_45.blood_perfusion.unwrap();
        assert!(new_perfusion < base_perfusion);
    }

    #[test]
    fn test_absorption_temperature() {
        let alpha0 = 0.5;

        // At body temperature
        assert_eq!(
            absorption_vs_temperature(alpha0, BODY_TEMPERATURE_C),
            alpha0
        );

        let alpha_50 = absorption_vs_temperature(alpha0, 50.0);
        let expected = alpha0 * (SOFT_TISSUE_ABSORPTION_COEFF_PER_C * 13.0).exp();
        assert!(
            (alpha_50 - expected).abs() < 1e-12,
            "alpha_50={alpha_50:.12e}, expected={expected:.12e}"
        );
        assert!(alpha_50 > alpha0);
    }

    #[test]
    fn test_absorption_never_negative_in_ablation_range() {
        let alpha0 = 0.5;
        for temperature in [BODY_TEMPERATURE_C, 50.0, 70.0, 90.0, 100.0] {
            let alpha = absorption_vs_temperature(alpha0, temperature);
            assert!(
                alpha >= 0.0,
                "absorption must remain non-negative at {temperature} °C, got {alpha}"
            );
        }
    }

    #[test]
    fn test_conductivity_increases_with_temperature() {
        let k0 = 0.5;

        let k_37 = conductivity_vs_temperature(k0, BODY_TEMPERATURE_C);
        let k_45 = conductivity_vs_temperature(k0, 45.0);
        let k_30 = conductivity_vs_temperature(k0, 30.0);

        assert_eq!(k_37, k0);
        assert_eq!(k_45, k0 * (1.0 + THERMAL_CONDUCTIVITY_COEFF_PER_C * 8.0));
        assert!(k_45 > k_37); // Increases with heating
        assert!(k_30 < k_37); // Decreases with cooling
    }

    #[test]
    fn test_perfusion_shutdown() {
        let w_b0 = 1.0;

        // Normal temperature
        let w_37 = perfusion_vs_temperature(w_b0, BODY_TEMPERATURE_C);
        assert_eq!(w_37, w_b0);

        // Mild hyperthermia - increased perfusion
        let w_40 = perfusion_vs_temperature(w_b0, 40.0);
        assert!(w_40 > w_b0);

        // High temperature - shutdown
        let w_55 = perfusion_vs_temperature(w_b0, 55.0);
        assert_eq!(w_55, 0.0);
    }

    #[test]
    fn test_sound_speed_temperature() {
        use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
        let c0 = SOUND_SPEED_TISSUE;

        let c_37 = sound_speed_vs_temperature(c0, BODY_TEMPERATURE_C);
        let c_45 = sound_speed_vs_temperature(c0, 45.0);

        assert_eq!(c_37, c0);
        let expected = c0 * (1.0 + SOFT_TISSUE_SOUND_SPEED_COEFF_PER_C * 8.0);
        assert!(
            (c_45 - expected).abs() < 1e-10,
            "c_45={c_45:.12e}, expected={expected:.12e}"
        );
        assert!(c_45 > c_37);
    }

    #[test]
    fn test_property_update_preserves_density() {
        let base = ThermalPropertyData::soft_tissue();
        let updated = update_properties(&base, 40.0);

        // Density should not change with moderate temperature variations
        assert_eq!(updated.density, base.density);
    }

    #[test]
    fn test_property_update_preserves_blood_specific_heat() {
        let base = ThermalPropertyData::soft_tissue();
        let updated = update_properties(&base, 40.0);

        // Blood specific heat relatively constant
        assert_eq!(updated.blood_specific_heat, base.blood_specific_heat);
    }

    #[test]
    fn test_round_trip_property_update() {
        let base = ThermalPropertyData::soft_tissue();

        // Update to elevated temperature
        let elevated = update_properties(&base, 45.0);

        // Verify changes
        assert!(elevated.conductivity > base.conductivity);
        assert!(elevated.specific_heat > base.specific_heat);

        // Update back to reference temperature
        let back_to_ref = update_properties(&base, BODY_TEMPERATURE_C);

        // The formulas are applied independently each time, so this just verifies
        // that applying the formula at reference temperature preserves values
        // (within numerical precision)
        let ref_again = update_properties(&base, BODY_TEMPERATURE_C);

        assert!((back_to_ref.conductivity - ref_again.conductivity).abs() < 1e-10);
        assert!((back_to_ref.specific_heat - ref_again.specific_heat).abs() < 1e-10);
        assert!((ref_again.conductivity - base.conductivity).abs() < 1e-10);
        assert!((ref_again.specific_heat - base.specific_heat).abs() < 1e-10);

        // Verify that elevated temperature actually changed the properties
        let conductivity_change =
            (elevated.conductivity - base.conductivity).abs() / base.conductivity;
        let specific_heat_change =
            (elevated.specific_heat - base.specific_heat).abs() / base.specific_heat;

        assert!(
            conductivity_change > 0.01,
            "Conductivity should change by >1% at 45°C"
        );
        assert!(
            specific_heat_change > 0.001,
            "Specific heat should change by >0.1% at 45°C"
        );
    }
}
