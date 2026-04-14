use crate::domain::imaging::photoacoustic::ThermoelasticProperties;

#[derive(Debug, Clone, Copy)]
pub struct ConfinementAssessment {
    pub optical_penetration_depth_m: f64,
    pub stress_confinement_time_s: f64,
    pub thermal_confinement_time_s: f64,
    pub stress_confined: bool,
    pub thermal_confined: bool,
}

impl ConfinementAssessment {
    #[must_use]
    pub fn evaluate(
        mu_a_m_inv: f64,
        pulse_duration_s: f64,
        thermoelastic: ThermoelasticProperties,
    ) -> Self {
        let optical_penetration_depth_m = if mu_a_m_inv > 0.0 {
            1.0 / mu_a_m_inv
        } else {
            f64::INFINITY
        };
        let stress_confinement_time_s =
            optical_penetration_depth_m / thermoelastic.sound_speed_m_s.max(f64::MIN_POSITIVE);
        let thermal_diffusivity = thermoelastic.thermal_diffusivity_m2_s();
        let thermal_confinement_time_s = optical_penetration_depth_m.powi(2)
            / (4.0 * thermal_diffusivity.max(f64::MIN_POSITIVE));

        Self {
            optical_penetration_depth_m,
            stress_confinement_time_s,
            thermal_confinement_time_s,
            stress_confined: pulse_duration_s <= stress_confinement_time_s,
            thermal_confined: pulse_duration_s <= thermal_confinement_time_s,
        }
    }
}
