use super::StateDependentConstants;

impl StateDependentConstants {
    /// Calculate surface tension of water with temperature dependence
    ///
    /// Uses International Association for the Properties of Water and Steam (IAPWS)
    /// correlation with critical point scaling:
    /// σ(T) = B·τ^μ·(1 + b·τ)
    ///
    /// Where τ = 1 - T/T_c (reduced temperature)
    /// Constants: B = 0.2358 N/m, μ = 1.256, b = -0.625
    ///
    /// Valid range: 0-370°C
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Surface tension [N/m]
    pub fn surface_tension_water(&self, temperature: f64) -> f64 {
        const T_CRITICAL: f64 = 647.096; // K (IAPWS critical temperature)
        const B: f64 = 0.2358; // N/m (amplitude)
        const MU: f64 = 1.256; // Exponent
        const BETA: f64 = -0.625; // Correction coefficient

        let t_kelvin = temperature + 273.15;

        if t_kelvin >= T_CRITICAL {
            return 0.0;
        }

        let tau = 1.0 - t_kelvin / T_CRITICAL;
        let sigma = B * tau.powf(MU) * (1.0 + BETA * tau);

        sigma.max(0.0)
    }

    /// Calculate vapor pressure of water with temperature dependence
    ///
    /// Uses Antoine equation:
    /// log₁₀(P_v) = A - B/(C + T)
    ///
    /// Where P_v is in mmHg and T in °C.
    ///
    /// Constants for water (0-100°C):
    /// - A = 8.07131
    /// - B = 1730.63
    /// - C = 233.426
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    ///
    /// # Returns
    /// Vapor pressure [Pa]
    pub fn vapor_pressure_water(&self, temperature: f64) -> f64 {
        const A: f64 = 8.07131;
        const B: f64 = 1730.63;
        const C: f64 = 233.426;

        let log_p_mmhg = A - B / (C + temperature);
        let p_mmhg = 10.0_f64.powf(log_p_mmhg);

        p_mmhg * 133.322
    }

    /// Calculate cavitation threshold pressure.
    ///
    /// The threshold is the minimum negative pressure (tension) required to initiate cavitation.
    /// P_threshold (tension) = P_v + sqrt(2·σ/(3·R_nuclei)) - P_ambient
    ///
    /// All parameters are temperature-dependent.
    ///
    /// # Arguments
    /// * `temperature` - Temperature [°C]
    /// * `nuclei_radius` - Bubble nuclei radius [m]
    /// * `ambient_pressure` - Ambient pressure [Pa]
    ///
    /// # Returns
    /// Cavitation threshold pressure amplitude (negative = tension) [Pa]
    pub fn cavitation_threshold(
        &self,
        temperature: f64,
        nuclei_radius: f64,
        ambient_pressure: f64,
    ) -> f64 {
        let p_v = self.vapor_pressure_water(temperature);
        let sigma = self.surface_tension_water(temperature);
        let p_blake = (2.0 * sigma / (3.0 * nuclei_radius)).sqrt();

        -(ambient_pressure - p_v - p_blake)
    }
}
