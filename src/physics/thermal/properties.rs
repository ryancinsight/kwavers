//! Temperature-dependent thermal properties
//!
//! References:
//! - Duck (1990) "Physical Properties of Tissue"
//! - IT'IS Foundation tissue property database

use crate::domain::medium::properties::ThermalPropertyData;

/// Temperature-dependent thermal conductivity
///
/// Linear increase with temperature (approximately 0.2%/°C)
///
/// # Arguments
///
/// * `k0` - Base thermal conductivity (W/m/K)
/// * `temperature` - Current temperature (°C)
#[must_use]
pub fn conductivity_vs_temperature(k0: f64, temperature: f64) -> f64 {
    // Linear increase with temperature (approximately 0.2%/°C)
    k0 * (1.0 + 0.002 * (temperature - 37.0))
}

/// Temperature-dependent specific heat
///
/// Slight increase with temperature
///
/// # Arguments
///
/// * `c0` - Base specific heat capacity (J/kg/K)
/// * `temperature` - Current temperature (°C)
#[must_use]
pub fn specific_heat_vs_temperature(c0: f64, temperature: f64) -> f64 {
    // Slight increase with temperature
    c0 * (1.0 + 0.001 * (temperature - 37.0))
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
    if temperature < 37.0 {
        // Reduced perfusion when cold
        w_b0 * (0.5 + 0.5 * temperature / 37.0)
    } else if temperature < 42.0 {
        // Increased perfusion with mild heating (vasodilation)
        w_b0 * (1.0 + 0.3 * (temperature - 37.0) / 5.0)
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
/// Decreases with temperature (approximately -2%/°C)
///
/// # Arguments
///
/// * `alpha0` - Base absorption coefficient (Np/m)
/// * `temperature` - Current temperature (°C)
#[must_use]
pub fn absorption_vs_temperature(alpha0: f64, temperature: f64) -> f64 {
    // Decreases with temperature (approximately -2%/°C)
    alpha0 * (1.0 - 0.02 * (temperature - 37.0).max(0.0))
}

/// Sound speed temperature dependence
///
/// Increases with temperature (approximately 1.5 m/s/°C)
///
/// # Arguments
///
/// * `c0` - Base sound speed (m/s)
/// * `temperature` - Current temperature (°C)
#[must_use]
pub fn sound_speed_vs_temperature(c0: f64, temperature: f64) -> f64 {
    // Increases with temperature (approximately 1.5 m/s/°C)
    c0 + 1.5 * (temperature - 37.0)
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
        assert_eq!(absorption_vs_temperature(alpha0, 37.0), alpha0);

        // At elevated temperature - should decrease
        assert!(absorption_vs_temperature(alpha0, 50.0) < alpha0);
    }

    #[test]
    fn test_conductivity_increases_with_temperature() {
        let k0 = 0.5;

        let k_37 = conductivity_vs_temperature(k0, 37.0);
        let k_45 = conductivity_vs_temperature(k0, 45.0);
        let k_30 = conductivity_vs_temperature(k0, 30.0);

        assert_eq!(k_37, k0); // Reference temperature
        assert!(k_45 > k_37); // Increases with heating
        assert!(k_30 < k_37); // Decreases with cooling
    }

    #[test]
    fn test_perfusion_shutdown() {
        let w_b0 = 1.0;

        // Normal temperature
        let w_37 = perfusion_vs_temperature(w_b0, 37.0);
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
        let c0 = 1540.0;

        let c_37 = sound_speed_vs_temperature(c0, 37.0);
        let c_45 = sound_speed_vs_temperature(c0, 45.0);

        assert_eq!(c_37, c0);
        assert!((c_45 - c_37 - 1.5 * 8.0).abs() < 1e-6); // 1.5 m/s/°C
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
        let back_to_ref = update_properties(&base, 37.0);

        // The formulas are applied independently each time, so this just verifies
        // that applying the formula at reference temperature preserves values
        // (within numerical precision)
        let ref_again = update_properties(&base, 37.0);

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
