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

use super::{CouplingPosition, MultiPhysicsCoupling};
use aequitas::systems::si::quantities::{
    Dimensionless, EnergyPerArea, EnergyPerVolume, Length, Pressure, ReciprocalLength, Time,
    Velocity,
};
use hyperion::{
    coefficient::{Absorption, InteractionCoefficient, ReducedScattering},
    quantity::PathLength,
    transport::DiffusionCoefficients,
    TransportError,
};
use leto::Array3;

/// Electromagnetic-acoustic coupling for photoacoustic effects.
pub trait ElectromagneticAcousticCoupling: MultiPhysicsCoupling {
    /// Optical absorption coefficient at the given wavelength.
    fn optical_absorption_coefficient(
        &self,
        position: &CouplingPosition,
        wavelength: Length,
    ) -> InteractionCoefficient<f64, Absorption>;

    /// Reduced scattering coefficient at the given wavelength.
    fn reduced_scattering_coefficient(
        &self,
        position: &CouplingPosition,
        wavelength: Length,
    ) -> InteractionCoefficient<f64, ReducedScattering>;

    /// Grüneisen parameter.
    fn gruneisen_parameter(&self, position: &CouplingPosition) -> Dimensionless;

    /// Mean cosine of the photon scattering angle.
    fn anisotropy_factor(&self, position: &CouplingPosition) -> Dimensionless;

    /// Initial pressure under stress confinement, p₀ = Γ μₐ Φ.
    fn fluence_to_pressure(
        &self,
        fluence: EnergyPerArea<&Array3<f64>>,
        position: &CouplingPosition,
        wavelength: Length,
    ) -> Pressure<Array3<f64>> {
        let gain = self.gruneisen_parameter(position)
            * self
                .optical_absorption_coefficient(position, wavelength)
                .into_quantity();
        Pressure::from_base(
            fluence
                .into_base()
                .mapv(|value| (gain * EnergyPerArea::from_base(value)).into_base()),
        )
    }

    /// Fluence for energy density held constant over the pulse duration.
    /// Photon travel distance is the propagation speed times that duration.
    fn em_energy_to_fluence(
        &self,
        energy_density: EnergyPerVolume<&Array3<f64>>,
        propagation_speed: Velocity,
        pulse_duration: Time,
    ) -> EnergyPerArea<Array3<f64>> {
        let travel = propagation_speed * pulse_duration;
        EnergyPerArea::from_base(
            energy_density
                .into_base()
                .mapv(|value| (EnergyPerVolume::from_base(value) * travel).into_base()),
        )
    }

    /// Evaluate the radial diffusion kernel exp(−μ_eff r)/r away from the source.
    ///
    /// # Errors
    ///
    /// Returns a Hyperion transport error for a coincident point or an invalid
    /// path or coefficient.
    fn diffuse_fluence_kernel(
        &self,
        source_position: &CouplingPosition,
        evaluation_position: &CouplingPosition,
        wavelength: Length,
    ) -> Result<ReciprocalLength, TransportError<f64>> {
        let [dx, dy, dz] = std::array::from_fn(|axis| {
            (evaluation_position[axis] - source_position[axis]).into_base()
        });
        let distance = Length::from_base(dx.hypot(dy).hypot(dz));
        if distance.into_base() == 0.0 {
            return Err(TransportError::InvalidValue {
                field: hyperion::ValueKind::PathLength,
                value: 0.0,
                constraint: hyperion::ValueConstraint::FinitePositive,
            });
        }
        let absorption = self.optical_absorption_coefficient(evaluation_position, wavelength);
        let reduced = self.reduced_scattering_coefficient(evaluation_position, wavelength);
        let attenuation =
            DiffusionCoefficients::new(absorption, reduced)?.effective_attenuation()?;
        let transmission = attenuation
            .optical_depth(PathLength::new(distance)?)?
            .transmission()
            .into_quantity();
        Ok(transmission / distance)
    }
}
