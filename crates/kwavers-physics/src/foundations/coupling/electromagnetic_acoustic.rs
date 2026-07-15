//! Electromagnetic–Acoustic Coupling (Photoacoustic Effects)
//!
//! ## Mathematical Foundation
//!
//! A short laser pulse deposits energy in an optically absorbing medium.
//! Under stress-confinement conditions (pulse duration τ_p << τ_s = d/c_s where
//! d is the optical penetration depth and c_s is the speed of sound), the
//! deposited optical energy converts directly to an initial pressure distribution:
//!
//! ```text
//! p₀(r) = Γ(r) · μ_a(r) · Φ(r)
//! ```
//!
//! where:
//! - Γ = βc²/C_p is the Grüneisen parameter (dimensionless), with β the
//!   isobaric volume expansion coefficient [K⁻¹], c the sound speed [m/s],
//!   and C_p the specific heat capacity [J/(kg·K)].
//! - μ_a is the optical absorption coefficient [m⁻¹].
//! - Φ is the optical fluence distribution [J/m²].
//!
//! ## Theorem (Diffusion Approximation Validity)
//!
//! **Statement.** The photon diffusion equation approximation is valid when:
//!
//! ```text
//! μ_a << μ_s'    (scattering dominates absorption)
//! r >> 1/μ_tr    (far from source, μ_tr = μ_a + μ_s')
//! ```
//!
//! Under these conditions the Green's function solution for the fluence rate
//! from a point source at the origin is:
//!
//! ```text
//! Φ(r) = S · exp(−μ_eff · r) / (4π D r)
//! ```
//!
//! where D = 1/(3μ_tr) is the diffusion coefficient and
//! μ_eff = √(3μ_a·μ_tr) = √(3μ_a(μ_a + μ_s')) is the effective attenuation
//! coefficient.
//!
//! **Proof sketch.** The steady-state photon diffusion equation is:
//! `D ∇²Φ − μ_a Φ = −S δ(r)`.  Applying the Helmholtz Green's function in 3D
//! with wavenumber k = √(μ_a/D) = μ_eff yields the expression above.
//! (Wang & Wu 2007, §3.3.)
//!
//! **Breakdown conditions.** For near-source regions (r < 1/μ_tr), ballistic
//! (single-scatter) or radiative-transfer models are required.  For tissue with
//! μ_s' ≈ 1 mm⁻¹, this means the diffusion approximation fails within ≈1 mm.
//!
//! ## Theorem (Stress-Confinement Condition)
//!
//! Photoacoustic signal generation is maximally efficient when the laser pulse
//! duration satisfies τ_p < τ_s = d/c_s (stress confinement), where d = 1/μ_eff
//! is the optical penetration depth.  For tissue (c_s≈1500 m/s, μ_eff≈1 cm⁻¹),
//! this requires τ_p < 1/(0.01·1500) ≈ 67 μs.  For pulsed Nd:YAG (τ_p≈10 ns),
//! the condition is trivially satisfied.
//!
//! ## References
//!
//! - Wang L.V., Wu H.I. (2007). *Biomedical Optics: Principles and Imaging*.
//!   Wiley-Interscience, §3.3.
//! - Cox B.T. et al. (2012). "Quantitative spectroscopic photoacoustic imaging:
//!   a review." J. Biomed. Opt. 17(6), 061202. DOI: 10.1117/1.JBO.17.6.061202
//! - Xu M., Wang L.V. (2006). "Photoacoustic imaging in biomedicine."
//!   Rev. Sci. Instrum. 77(4), 041101. DOI: 10.1063/1.2195024

use super::MultiPhysicsCoupling;
use leto::Array3;

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
        fluence: &Array3<f64>,
        position: &[f64],
        wavelength: f64,
    ) -> Array3<f64> {
        let gamma = self.gruneisen_parameter(position);
        let mu_a = self.optical_absorption_coefficient(position, wavelength);
        fluence.mapv(|phi| gamma * mu_a * phi)
    }

    /// Compute optical fluence from electromagnetic energy density
    fn em_energy_to_fluence(
        &self,
        energy_density: &Array3<f64>,
        pulse_duration: f64,
    ) -> Array3<f64> {
        energy_density.mapv(|u| u * pulse_duration)
    }

    /// Optical diffusion approximation for fluence
    fn diffuse_fluence(
        &self,
        source_position: &[f64],
        evaluation_position: &[f64],
        wavelength: f64,
    ) -> f64 {
        let r = (evaluation_position[2] - source_position[2])
            .mul_add(
                evaluation_position[2] - source_position[2],
                (evaluation_position[1] - source_position[1]).mul_add(
                    evaluation_position[1] - source_position[1],
                    (evaluation_position[0] - source_position[0]).powi(2),
                ),
            )
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
