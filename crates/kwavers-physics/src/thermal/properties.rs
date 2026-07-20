//! Temperature-dependent thermal properties.
//!
//! # Scope
//!
//! This module composes the Proteus thermophysical response law with local
//! perfusion, absorption, and acoustic updates used by thermal therapy and
//! thermometry workflows. Tissue-specific base values remain supplied by
//! [`ThermalPropertyData`] or acoustic media records.
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

use aequitas::systems::si::quantities::{ReciprocalTemperature, ThermodynamicTemperature};
use kwavers_core::constants::thermodynamic::{
    BODY_TEMPERATURE_C, BODY_TEMPERATURE_K, KELVIN_OFFSET_C,
};
use kwavers_core::constants::tissue_thermal::{
    SOFT_TISSUE_ABSORPTION_COEFF_PER_C, SOFT_TISSUE_SOUND_SPEED_COEFF_PER_C,
    SPECIFIC_HEAT_COEFF_PER_C, THERMAL_CONDUCTIVITY_COEFF_PER_C,
};
use kwavers_medium::properties::ThermalPropertyData;
use proteus::{ConstantResponse, ConstitutiveLaw, LinearResponse, ResponseSet, TemperatureLaw};

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
/// # Errors
///
/// Returns an error when the temperature is non-physical or an evaluated
/// thermophysical property violates the Proteus contract.
///
pub fn update_properties(
    base_properties: &ThermalPropertyData,
    temperature: f64,
) -> Result<ThermalPropertyData, String> {
    let new_perfusion = base_properties
        .blood_perfusion
        .map(|w_b| perfusion_vs_temperature(w_b, temperature));
    let responses = ResponseSet::new(
        ConstantResponse,
        LinearResponse::new(ReciprocalTemperature::from_base(SPECIFIC_HEAT_COEFF_PER_C))
            .map_err(|error| error.to_string())?,
        LinearResponse::new(ReciprocalTemperature::from_base(
            THERMAL_CONDUCTIVITY_COEFF_PER_C,
        ))
        .map_err(|error| error.to_string())?,
    );
    let law = TemperatureLaw::new(
        *base_properties.thermophysical(),
        ThermodynamicTemperature::from_base(BODY_TEMPERATURE_K),
        responses,
    )
    .map_err(|error| error.to_string())?;
    let temperature = ThermodynamicTemperature::from_base(temperature + KELVIN_OFFSET_C);
    let thermophysical = law
        .properties(&temperature)
        .map_err(|error| error.to_string())?;

    ThermalPropertyData::from_thermophysical(
        thermophysical,
        new_perfusion,
        base_properties.blood_specific_heat,
    )
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
        * (SOFT_TISSUE_ABSORPTION_COEFF_PER_C * (temperature - BODY_TEMPERATURE_C).max(0.0)).exp()
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
        let props_45 = update_properties(&base, 45.0).expect("45 °C produces physical properties");

        // Conductivity should increase
        assert!(props_45.conductivity() > base.conductivity());

        // Specific heat should increase slightly
        assert!(props_45.specific_heat() > base.specific_heat());

        // Perfusion should decrease (approaching shutdown)
        let base_perfusion = base.blood_perfusion.expect("soft tissue defines perfusion");
        let new_perfusion = props_45
            .blood_perfusion
            .expect("updated soft tissue retains perfusion");
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
        let base = ThermalPropertyData::soft_tissue();
        let k0 = base.conductivity();
        let k_37 = update_properties(&base, BODY_TEMPERATURE_C)
            .expect("body temperature is physical")
            .conductivity();
        let k_45 = update_properties(&base, 45.0)
            .expect("45 °C is physical")
            .conductivity();
        let k_30 = update_properties(&base, 30.0)
            .expect("30 °C is physical")
            .conductivity();

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
        use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
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
        let updated = update_properties(&base, 40.0).expect("40 °C produces physical properties");

        // Density should not change with moderate temperature variations
        assert_eq!(updated.density(), base.density());
    }

    #[test]
    fn property_update_rejects_non_physical_temperature() {
        let base = ThermalPropertyData::soft_tissue();
        let error = update_properties(&base, f64::NAN)
            .expect_err("non-finite thermodynamic temperature must be rejected");
        assert!(error.contains("Evaluation"));
    }

    #[test]
    fn test_property_update_preserves_blood_specific_heat() {
        let base = ThermalPropertyData::soft_tissue();
        let updated = update_properties(&base, 40.0).expect("40 °C produces physical properties");

        // Blood specific heat relatively constant
        assert_eq!(updated.blood_specific_heat, base.blood_specific_heat);
    }

    #[test]
    fn test_round_trip_property_update() {
        let base = ThermalPropertyData::soft_tissue();

        // Update to elevated temperature
        let elevated = update_properties(&base, 45.0).expect("45 °C produces physical properties");

        // Verify changes
        assert!(elevated.conductivity() > base.conductivity());
        assert!(elevated.specific_heat() > base.specific_heat());

        // Update back to reference temperature
        let back_to_ref = update_properties(&base, BODY_TEMPERATURE_C)
            .expect("body temperature produces reference properties");

        // The formulas are applied independently each time, so this just verifies
        // that applying the formula at reference temperature preserves values
        // (within numerical precision)
        let ref_again = update_properties(&base, BODY_TEMPERATURE_C)
            .expect("body temperature produces reference properties");

        assert!((back_to_ref.conductivity() - ref_again.conductivity()).abs() < 1e-10);
        assert!((back_to_ref.specific_heat() - ref_again.specific_heat()).abs() < 1e-10);
        assert!((ref_again.conductivity() - base.conductivity()).abs() < 1e-10);
        assert!((ref_again.specific_heat() - base.specific_heat()).abs() < 1e-10);

        // Verify that elevated temperature actually changed the properties
        let conductivity_change =
            (elevated.conductivity() - base.conductivity()).abs() / base.conductivity();
        let specific_heat_change =
            (elevated.specific_heat() - base.specific_heat()).abs() / base.specific_heat();

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
