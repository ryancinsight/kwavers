//! Electromagnetic-Acoustic Coupling (Photoacoustic Effects)
//!
//! ## Mathematical Foundation
//!
//! Initial photoacoustic pressure:
//! ```text
//! p₀ = Γ μ_a Φ
//! ```
//!
//! Grüneisen parameter:
//! ```text
//! Γ = β c² / C_p
//! ```
//!
//! Diffuse fluence (Green's function):
//! ```text
//! Φ(r) ∝ exp(-μ_eff r) / r,   μ_eff = √(3 μ_a (μ_a + μ_s'))
//! ```
//!
//! ## References
//!
//! - Wang & Wu (2007) "Biomedical Optics: Principles and Imaging"
//! - Cox et al. (2012) "Quantitative spectroscopic photoacoustic imaging"

use ndarray::ArrayD;
use super::MultiPhysicsCoupling;

/// Electromagnetic-acoustic coupling for photoacoustic effects
pub trait ElectromagneticAcousticCoupling: MultiPhysicsCoupling {
    /// Optical absorption coefficient μ_a (m⁻¹)
    fn optical_absorption_coefficient(&self, position: &[f64], wavelength: f64) -> f64;

    /// Reduced scattering coefficient μ_s' (m⁻¹)
    fn reduced_scattering_coefficient(&self, position: &[f64], wavelength: f64) -> f64;

    /// Grüneisen parameter Γ = β c² / C_p (dimensionless)
    fn gruneisen_parameter(&self, position: &[f64]) -> f64;

    /// Anisotropy factor g (dimensionless, -1 to 1)
    fn anisotropy_factor(&self, _position: &[f64]) -> f64 {
        0.9 // Typical for tissue (forward scattering)
    }

    /// Compute initial acoustic pressure from optical fluence: p₀ = Γ μ_a Φ
    fn fluence_to_pressure(
        &self,
        fluence: &ArrayD<f64>,
        position: &[f64],
        wavelength: f64,
    ) -> ArrayD<f64> {
        let gamma = self.gruneisen_parameter(position);
        let mu_a = self.optical_absorption_coefficient(position, wavelength);
        fluence.mapv(|phi| gamma * mu_a * phi)
    }

    /// Compute optical fluence from electromagnetic energy density
    fn em_energy_to_fluence(
        &self,
        energy_density: &ArrayD<f64>,
        pulse_duration: f64,
    ) -> ArrayD<f64> {
        energy_density.mapv(|u| u * pulse_duration)
    }

    /// Optical diffusion approximation for fluence
    fn diffuse_fluence(
        &self,
        source_position: &[f64],
        evaluation_position: &[f64],
        wavelength: f64,
    ) -> f64 {
        let r = ((evaluation_position[0] - source_position[0]).powi(2)
            + (evaluation_position[1] - source_position[1]).powi(2)
            + (evaluation_position[2] - source_position[2]).powi(2))
        .sqrt();

        if r == 0.0 {
            return 0.0;
        }

        let mu_a = self.optical_absorption_coefficient(evaluation_position, wavelength);
        let mu_s_prime = self.reduced_scattering_coefficient(evaluation_position, wavelength);
        let mu_eff = (3.0 * mu_a * (mu_a + mu_s_prime)).sqrt();
        (-mu_eff * r).exp() / r
    }
}
