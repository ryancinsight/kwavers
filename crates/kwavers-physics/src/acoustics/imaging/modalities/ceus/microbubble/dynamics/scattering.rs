//! Nonlinear scattering efficiency for CEUS microbubbles.

use super::BubbleDynamics;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_imaging::ultrasound::ceus::Microbubble;

impl BubbleDynamics {
    /// Compute nonlinear scattering efficiency via Lorentzian resonance response.
    #[must_use]
    pub fn nonlinear_scattering_efficiency(
        &self,
        bubble: &Microbubble,
        pressure_amplitude: f64,
        frequency: f64,
    ) -> f64 {
        let r0 = bubble.radius_eq.max(1e-12);
        let resonance_freq = bubble
            .resonance_frequency(self.ambient_pressure, self.liquid_density)
            .max(1.0);
        let omega_ratio = frequency / resonance_freq;
        let omega_res = TWO_PI * resonance_freq;
        let epsilon = pressure_amplitude / (self.liquid_density * r0 * r0 * omega_res * omega_res);
        let delta = self.damping_coefficient;
        let omega_sq = omega_ratio * omega_ratio;
        let denom_sq = (delta * omega_ratio).mul_add(delta * omega_ratio, (1.0 - omega_sq).powi(2));
        let chi = 1.0 / denom_sq.sqrt().max(f64::EPSILON);
        let kappa = bubble.polytropic_index;
        let nonlinearity_prefactor = 1.5 * kappa * 3.0f64.mul_add(kappa, 1.0);

        (nonlinearity_prefactor * epsilon * chi).max(0.0)
    }
}
