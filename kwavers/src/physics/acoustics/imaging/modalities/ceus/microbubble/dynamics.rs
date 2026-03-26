use super::response::BubbleResponse;
use crate::core::error::KwaversResult;
use crate::domain::imaging::ultrasound::ceus::Microbubble;

/// Linearised total damping constant for micron-scale bubbles at resonance.
///
/// # Physical Basis
///
/// For a gas bubble oscillating in a viscous liquid, the total damping
/// δ_total = δ_viscous + δ_thermal + δ_radiation. Prosperetti (1977)
/// showed that for air bubbles in water at 20°C with radii in the
/// 1–10 μm CEUS-relevant range, the combined dimensionless damping
/// coefficient is approximately 0.1 at frequencies near resonance.
///
/// # Reference
///
/// Prosperetti, A. (1977). "Thermal effects and damping mechanisms in
/// the forced radial oscillations of gas bubbles in liquids." JASA,
/// 61(1), 17–27.
pub(crate) const PROSPERETTI_TOTAL_DAMPING_COEFFICIENT: f64 = 0.1;

/// Microbubble dynamics simulation
#[derive(Debug)]
pub struct BubbleDynamics {
    /// Time step for integration (s)
    dt: f64,
    /// Ambient pressure (Pa)
    ambient_pressure: f64,
    /// Liquid density (kg/m³)
    liquid_density: f64,
    /// Acoustic damping coefficient
    damping_coefficient: f64,
}

impl BubbleDynamics {
    /// Create new bubble dynamics simulator
    pub fn new() -> Self {
        Self {
            dt: 1e-9,                   // 1 ns time step
            ambient_pressure: 101325.0, // Atmospheric pressure
            liquid_density: 1000.0,     // Water density
            damping_coefficient: PROSPERETTI_TOTAL_DAMPING_COEFFICIENT,
        }
    }

    /// Simulate radial oscillation response to acoustic pressure
    ///
    /// # Arguments
    ///
    /// * `bubble` - Microbubble properties
    /// * `acoustic_pressure` - Incident acoustic pressure (Pa)
    /// * `frequency` - Acoustic frequency (Hz)
    /// * `duration` - Simulation duration (s)
    ///
    /// # Returns
    ///
    /// Time series of bubble radius and scattered pressure
    pub fn simulate_oscillation(
        &self,
        bubble: &Microbubble,
        acoustic_pressure: f64,
        frequency: f64,
        duration: f64,
    ) -> KwaversResult<BubbleResponse> {
        let n_steps = (duration / self.dt) as usize;
        let mut radius = vec![0.0; n_steps];
        let mut scattered_pressure = vec![0.0; n_steps];

        // Initial conditions
        radius[0] = bubble.radius_eq;
        let mut radius_dot = 0.0; // Initial velocity

        // Time integration using Verlet method
        for i in 0..n_steps.saturating_sub(1) {
            let time = i as f64 * self.dt;

            // Incident acoustic pressure
            let p_acoustic =
                acoustic_pressure * (2.0 * std::f64::consts::PI * frequency * time).sin();

            // Gas pressure (polytropic)
            let p_gas = self.ambient_pressure
                * (bubble.radius_eq / radius[i]).powf(bubble.polytropic_index);

            // Surface tension and shell pressure
            let p_surface = 2.0 * bubble.surface_tension / radius[i];
            let p_shell = 4.0
                * bubble.shell_elasticity
                * bubble.shell_thickness
                * (radius[i] - bubble.radius_eq)
                / (bubble.radius_eq * bubble.radius_eq);

            // Viscous damping term
            let damping_force = -self.damping_coefficient * radius_dot / radius[i];

            // Rayleigh-Plesset equation
            let total_pressure = p_gas + p_surface + p_shell - self.ambient_pressure - p_acoustic;
            let acceleration = total_pressure / (self.liquid_density * radius[i]) + damping_force
                - 1.5 * radius_dot * radius_dot / radius[i];

            // Verlet integration
            let radius_new =
                radius[i] + radius_dot * self.dt + 0.5 * acceleration * self.dt * self.dt;
            let acceleration_new = acceleration; // Simplified

            radius_dot += 0.5 * (acceleration + acceleration_new) * self.dt;

            if i + 1 < n_steps {
                radius[i + 1] = radius_new;

                // Scattered pressure using linear scattering approximation
                let volume_change =
                    4.0 / 3.0 * std::f64::consts::PI * (radius_new.powi(3) - radius[i].powi(3));
                scattered_pressure[i + 1] = self.liquid_density * volume_change / self.dt;
            }
        }

        Ok(BubbleResponse {
            time: (0..n_steps).map(|i| i as f64 * self.dt).collect(),
            radius,
            scattered_pressure,
        })
    }

    /// Compute nonlinear scattering efficiency
    ///
    /// # Arguments
    ///
    /// * `bubble` - Microbubble properties
    /// * `pressure_amplitude` - Acoustic pressure amplitude (Pa)
    /// * `frequency` - Acoustic frequency (Hz)
    ///
    /// # Returns
    ///
    /// Nonlinear scattering coefficient (dimensionless)
    #[must_use]
    pub fn nonlinear_scattering_efficiency(
        &self,
        bubble: &Microbubble,
        pressure_amplitude: f64,
        frequency: f64,
    ) -> f64 {
        // Simplified model based on compression-only behavior
        let resonance_freq = bubble.resonance_frequency(self.ambient_pressure, self.liquid_density);
        let freq_ratio = frequency / resonance_freq;

        // Nonlinearity parameter β
        let beta = if freq_ratio < 1.0 {
            // Subharmonic regime
            1.0 - freq_ratio.powi(2)
        } else {
            // Ultraharmonic regime
            freq_ratio.powi(2) - 1.0
        };

        // Pressure-dependent nonlinearity
        let acoustic_parameter = pressure_amplitude * bubble.radius_eq
            / (self.ambient_pressure * bubble.shell_elasticity);

        // Empirical nonlinear scattering efficiency
        beta * acoustic_parameter.min(1.0)
    }
}

impl Default for BubbleDynamics {
    fn default() -> Self {
        Self::new()
    }
}
