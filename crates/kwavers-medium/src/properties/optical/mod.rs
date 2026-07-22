//! Optical material property data structures for light propagation and scattering
//!
//! # Mathematical Foundation
//!
//! ## Radiative Transfer Equation (RTE)
//!
//! Light propagation in scattering media:
//! ```text
//! dI/ds = -μ_t I + μ_s ∫ p(θ) I(s') dΩ'
//! ```
//!
//! Where:
//! - `I`: Radiance (W/m²/sr)
//! - `s`: Path length (m)
//! - `μ_t = μ_a + μ_s`: Total attenuation coefficient (m⁻¹)
//! - `μ_a`: Absorption coefficient (m⁻¹)
//! - `μ_s`: Scattering coefficient (m⁻¹)
//! - `p(θ)`: Phase function (angular scattering probability)
//!
//! ## Henyey-Greenstein Phase Function
//!
//! Anisotropic scattering model:
//! ```text
//! p(θ) = (1 - g²) / (4π (1 + g² - 2g cos θ)^(3/2))
//! ```
//! - `g`: Anisotropy factor (⟨cos θ⟩)
//! - `g = 0`: Isotropic scattering
//! - `g > 0`: Forward scattering (typical for biological tissue)
//! - `g < 0`: Backward scattering
//!
//! ## Invariants
//!
//! - `absorption_coefficient ≥ 0` (m⁻¹)
//! - `scattering_coefficient ≥ 0` (m⁻¹)
//! - `-1 ≤ anisotropy ≤ 1` (dimensionless)
//! - `refractive_index ≥ 1.0` (vacuum is lower bound)

mod coefficients;
mod presets;
#[cfg(test)]
mod tests;

use aequitas::systems::si::{quantities::ReciprocalLength, units::PerMeter};
use hyperion::{
    coefficient::{Absorption, InteractionCoefficient, OpticalCoefficients, Scattering},
    quantity::Anisotropy,
    transport::{reduced_scattering, DiffusionCoefficients},
    TransportError,
};
use std::fmt;

/// Canonical optical material properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpticalPropertyData {
    coefficients: OpticalCoefficients<f64>,
    anisotropy: Anisotropy<f64>,
    refractive_index: f64,
}

impl fmt::Display for OpticalPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Optical(μ_a={:.2} m⁻¹, μ_s={:.1} m⁻¹, μ_s'={:.1} m⁻¹, g={:.3}, n={:.3})",
            self.absorption_coefficient(),
            self.scattering_coefficient(),
            self.reduced_scattering_coefficient(),
            self.anisotropy(),
            self.refractive_index(),
        )
    }
}

impl OpticalPropertyData {
    fn from_si(
        absorption_coefficient: f64,
        scattering_coefficient: f64,
        anisotropy: f64,
        refractive_index: f64,
    ) -> Result<Self, String> {
        if !refractive_index.is_finite() || refractive_index < 1.0 {
            return Err(format!(
                "Refractive index must be finite and at least 1.0, got {refractive_index}"
            ));
        }

        let absorption = InteractionCoefficient::<_, Absorption>::new(
            ReciprocalLength::from_unit::<PerMeter>(absorption_coefficient),
        )
        .map_err(|error| error.to_string())?;
        let scattering = InteractionCoefficient::<_, Scattering>::new(
            ReciprocalLength::from_unit::<PerMeter>(scattering_coefficient),
        )
        .map_err(|error| error.to_string())?;
        let coefficients =
            OpticalCoefficients::new(absorption, scattering).map_err(|error| error.to_string())?;
        let anisotropy = Anisotropy::new(
            aequitas::systems::si::quantities::Dimensionless::from_base(anisotropy),
        )
        .map_err(|error| error.to_string())?;

        reduced_scattering(scattering, anisotropy).map_err(|error| error.to_string())?;

        Ok(Self {
            coefficients,
            anisotropy,
            refractive_index,
        })
    }

    /// Return the validated unreduced optical coefficient pair.
    #[must_use]
    pub const fn optical_coefficients(&self) -> OpticalCoefficients<f64> {
        self.coefficients
    }

    /// Return the validated diffusion coefficient pair.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError::DegenerateTransport`] for a vacuum aggregate.
    pub fn diffusion_coefficients(
        &self,
    ) -> Result<DiffusionCoefficients<f64>, TransportError<f64>> {
        DiffusionCoefficients::new(
            *self.coefficients.absorption(),
            reduced_scattering(*self.coefficients.scattering(), self.anisotropy)
                .expect("invariant: construction validates reduced scattering"),
        )
    }
}
