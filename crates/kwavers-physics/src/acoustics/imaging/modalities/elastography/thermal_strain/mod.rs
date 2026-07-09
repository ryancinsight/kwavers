//! # Thermal Strain Imaging (Ultrasound Thermometry)
//!
//! Non-invasive temperature estimation from the apparent tissue strain induced
//! by heating. Mild heating perturbs both the local sound speed and the physical
//! dimensions of tissue; a pulse-echo system tracks the resulting apparent
//! displacement of scatterers, whose axial gradient — the *thermal strain* — is
//! linearly proportional to the temperature change.
//!
//! ## Theorem (thermal strain ↔ temperature)
//!
//! Consider a 1-D medium probed from `z = 0` with a transducer that estimates
//! scatterer positions using a fixed reference sound speed `c₀`. Let the
//! temperature rise by `ΔT(z)`. To first order in `ΔT`, the apparent axial
//! displacement of an echo originally at depth `z` is
//!
//! ```text
//! u(z) = ∫₀ᶻ (α_th − β_c) · ΔT(z') dz',     β_c = (1/c₀)·(dc/dT),
//! ```
//!
//! and therefore the *thermal strain* obeys
//!
//! ```text
//! ε_T(z) = ∂u/∂z = (α_th − β_c) · ΔT(z) = k_T · ΔT(z).
//! ```
//!
//! ### Proof
//! Two first-order effects displace the echo:
//!
//! 1. **Thermal expansion.** A material element between `0` and `z` lengthens by
//!    its integrated linear expansion, moving the scatterer physically to
//!    `z_phys = z + ∫₀ᶻ α_th ΔT dz'`.
//! 2. **Sound-speed change.** Heating changes the speed to
//!    `c(z') = c₀[1 + β_c ΔT(z')]` with `β_c = (1/c₀)(dc/dT)`.
//!
//! The instrument estimates position by integrating the round-trip time at the
//! fixed reference speed, `ẑ = ∫₀^{z_phys} (c₀/c(s)) ds`. Expanding the integrand
//! to first order, `c₀/c(s) = 1 − β_c ΔT(s) + O(ΔT²)`, and splitting the
//! integral at `z`:
//!
//! ```text
//! ẑ = ∫₀ᶻ [1 − β_c ΔT(s)] ds + (c₀/c(z))·∫₀ᶻ α_th ΔT ds
//!    = z + ∫₀ᶻ (α_th − β_c) ΔT(s) ds + O(ΔT²),
//! ```
//!
//! where the second term used `c₀/c(z) = 1 + O(ΔT)` multiplying an `O(ΔT)`
//! expansion contribution. The apparent displacement is `u(z) = ẑ − z`, giving
//! the integral relation; differentiating in `z` (Leibniz) yields
//! `ε_T(z) = (α_th − β_c) ΔT(z)`. ∎
//!
//! Inverting, `ΔT(z) = ε_T(z) / k_T`. For water-based soft tissue `dc/dT > 0`,
//! so `β_c` dominates `α_th` and `k_T < 0`; for lipid-based tissue `dc/dT < 0`,
//! giving `k_T > 0`. The sign reversal between water- and lipid-based tissue is
//! the physical basis of thermal-strain tissue discrimination.
//!
//! ## Pipeline
//! 1. [`tracking::track_axial_displacement`] — NCC speckle tracking of the
//!    apparent displacement `u(z)` between a pre- and post-heating RF volume.
//! 2. [`strain::least_squares_strain`] — Kallel–Ophir moving least-squares
//!    estimate of `ε_T = ∂u/∂z`.
//! 3. [`ThermalStrainConfig::temperature_from_strain`] — pointwise inversion to
//!    `ΔT`.
//!
//! ## References
//! - Maass-Moreno, R., & Damianou, C. A. (1996). "Noninvasive temperature
//!   estimation in tissue via ultrasound echo-shifts. Part I." *JASA*, 100(4),
//!   2514–2521.
//! - Simon, C., VanBaren, P., & Ebbini, E. S. (1998). "Two-dimensional
//!   temperature estimation using diagnostic ultrasound." *IEEE TUFFC*, 45(4),
//!   1088–1099.
//! - Seo, C. H., Shi, Y., Huang, S.-W., Kim, K., & O'Donnell, M. (2011).
//!   "Thermal strain imaging: A review." *Interface Focus*, 1(4), 649–664.

pub mod config;
pub mod strain;
pub mod tracking;

#[cfg(test)]
mod tests;

pub use config::ThermalStrainConfig;
pub use tracking::TrackingParams;

use kwavers_core::error::{KwaversResult, ValidationError};
use leto::Array3;

/// Result of a thermal strain reconstruction.
#[derive(Debug, Clone)]
pub struct ThermalStrainResult {
    /// Apparent axial displacement field `u(z)` (m).
    pub displacement: Array3<f64>,
    /// Thermal strain field `ε_T(z) = ∂u/∂z` (dimensionless).
    pub strain: Array3<f64>,
    /// Reconstructed temperature change `ΔT(z)` [°C].
    pub temperature_change: Array3<f64>,
}

/// Thermal strain imager: pre/post RF volumes → temperature-change map.
#[derive(Debug, Clone)]
pub struct ThermalStrainImager {
    config: ThermalStrainConfig,
    tracking: TrackingParams,
    /// RF sampling rate (Hz) used to convert sample lags to physical distance.
    sampling_rate: f64,
}

impl ThermalStrainImager {
    /// Construct an imager.
    ///
    /// # Errors
    /// Returns [`ValidationError`] when the configuration is invalid (see
    /// [`ThermalStrainConfig::validate`]) or `sampling_rate` is non-positive.
    pub fn new(
        config: ThermalStrainConfig,
        tracking: TrackingParams,
        sampling_rate: f64,
    ) -> KwaversResult<Self> {
        config.validate()?;
        if sampling_rate <= 0.0 {
            return Err(ValidationError::InvalidValue {
                parameter: "sampling_rate".to_owned(),
                value: sampling_rate,
                reason: "must be positive".to_owned(),
            }
            .into());
        }
        Ok(Self {
            config,
            tracking,
            sampling_rate,
        })
    }

    /// The configured thermoacoustic parameters.
    #[must_use]
    pub fn config(&self) -> &ThermalStrainConfig {
        &self.config
    }

    /// Reconstruct the temperature-change field from a pre- and post-heating RF
    /// volume.
    ///
    /// Both volumes are `[nx, ny, nz]` with the axial direction along the last
    /// axis.
    ///
    /// # Errors
    /// Returns [`ValidationError::DimensionMismatch`] when the two volumes have
    /// different shapes.
    pub fn reconstruct_temperature(
        &self,
        reference: &Array3<f64>,
        tracked: &Array3<f64>,
    ) -> KwaversResult<ThermalStrainResult> {
        if reference.dim() != tracked.dim() {
            return Err(ValidationError::DimensionMismatch {
                expected: format!("{:?}", reference.dim()),
                actual: format!("{:?}", tracked.dim()),
            }
            .into());
        }
        let dz = self.config.axial_sample_spacing(self.sampling_rate);
        let displacement =
            tracking::track_axial_displacement(reference, tracked, self.tracking, dz);
        let strain = strain::least_squares_strain(&displacement, dz, self.config.strain_window);
        let inv_k = 1.0 / self.config.combined_coefficient();
        let temperature_change = strain.mapv(|e| e * inv_k);
        Ok(ThermalStrainResult {
            displacement,
            strain,
            temperature_change,
        })
    }
}
