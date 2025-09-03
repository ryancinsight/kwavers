//! Temperature-dependent thermal properties
//!
//! References:
//! - Duck (1990) "Physical Properties of Tissue"
//! - IT'IS Foundation tissue property database

use super::ThermalProperties;

/// Temperature-dependent thermal conductivity
#[must_use]
pub fn conductivity_vs_temperature(k0: f64, temperature: f64) -> f64 {
    // Linear increase with temperature (approximately 0.2%/°C)
    k0 * (1.0 + 0.002 * (temperature - 37.0))
}

/// Temperature-dependent specific heat
#[must_use]
pub fn specific_heat_vs_temperature(c0: f64, temperature: f64) -> f64 {
    // Slight increase with temperature
    c0 * (1.0 + 0.001 * (temperature - 37.0))
}

/// Temperature-dependent blood perfusion
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
#[must_use]
pub fn update_properties(
    base_properties: &ThermalProperties,
    temperature: f64,
) -> ThermalProperties {
    ThermalProperties {
        k: conductivity_vs_temperature(base_properties.k, temperature),
        c: specific_heat_vs_temperature(base_properties.c, temperature),
        w_b: perfusion_vs_temperature(base_properties.w_b, temperature),
        ..base_properties.clone()
    }
}

/// Acoustic absorption coefficient temperature dependence
#[must_use]
pub fn absorption_vs_temperature(alpha0: f64, temperature: f64) -> f64 {
    // Decreases with temperature (approximately -2%/°C)
    alpha0 * (1.0 - 0.02 * (temperature - 37.0).max(0.0))
}

/// Sound speed temperature dependence
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
        let base = ThermalProperties::default();

        // Test at elevated temperature
        let props_45 = update_properties(&base, 45.0);

        // Conductivity should increase
        assert!(props_45.k > base.k);

        // Perfusion should decrease (approaching shutdown)
        assert!(props_45.w_b < base.w_b);
    }

    #[test]
    fn test_absorption_temperature() {
        let alpha0 = 0.5;

        // At body temperature
        assert_eq!(absorption_vs_temperature(alpha0, 37.0), alpha0);

        // At elevated temperature - should decrease
        assert!(absorption_vs_temperature(alpha0, 50.0) < alpha0);
    }
}
