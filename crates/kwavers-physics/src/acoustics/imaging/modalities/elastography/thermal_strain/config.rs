//! Thermoacoustic configuration for thermal strain imaging.
//!
//! Holds the two material coefficients that couple a local temperature change
//! `ΔT` to the apparent axial strain measured by speckle tracking, together with
//! the acquisition geometry needed to convert echo sample shifts into physical
//! displacement.

use kwavers_core::constants::{
    DC_DT_SOFT_TISSUE, SOUND_SPEED_TISSUE, THERMAL_EXPANSION_SOFT_TISSUE,
};
use kwavers_core::error::{KwaversResult, ValidationError};

/// Material and acquisition parameters for thermal strain thermometry.
///
/// The governing relation (derived in the [module documentation](super)) is
///
/// ```text
/// ε_T(z) = (α_th − β_c) · ΔT(z) = k_T · ΔT(z),   β_c = (1/c₀)·(dc/dT)
/// ```
///
/// so the temperature change is recovered from the measured thermal strain by
/// `ΔT = ε_T / k_T`. The combined coefficient `k_T` is negative for water-based
/// soft tissue (sound speed rises with temperature, `dc/dT > 0`) and positive
/// for lipid-based tissue (`dc/dT < 0`); this sign inversion is the basis of
/// water/lipid thermal-strain contrast.
///
/// # References
/// - Maass-Moreno, R., & Damianou, C. A. (1996). "Noninvasive temperature
///   estimation in tissue via ultrasound echo-shifts. Part I." *JASA*, 100(4),
///   2514–2521.
/// - Simon, C., VanBaren, P., & Ebbini, E. S. (1998). "Two-dimensional temperature
///   estimation using diagnostic ultrasound." *IEEE TUFFC*, 45(4), 1088–1099.
/// - Pernot, M., et al. (2004). "Temperature estimation using ultrasonic spatial
///   compound imaging." *IEEE TUFFC*, 51(5), 606–615.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalStrainConfig {
    /// Reference (baseline) sound speed `c₀` [m/s].
    pub sound_speed: f64,
    /// Temperature coefficient of sound speed `dc/dT` [m/s per °C].
    /// Positive for water-based tissue, negative for lipid/fat.
    pub dc_dt: f64,
    /// Linear coefficient of thermal expansion `α_th` [1/°C].
    pub thermal_expansion: f64,
    /// Axial window length (echo samples) for the least-squares strain estimator.
    /// Must be odd and ≥ 3 so the window is centered.
    pub strain_window: usize,
}

impl ThermalStrainConfig {
    /// Generic water-based soft tissue defaults, sourced from the workspace
    /// constant SSOT (`DC_DT_SOFT_TISSUE`, `THERMAL_EXPANSION_SOFT_TISSUE`,
    /// `SOUND_SPEED_TISSUE`).
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            sound_speed: SOUND_SPEED_TISSUE,
            dc_dt: DC_DT_SOFT_TISSUE,
            thermal_expansion: THERMAL_EXPANSION_SOFT_TISSUE,
            strain_window: 11,
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`ValidationError`] when the sound speed is non-positive, the
    /// strain window is even or smaller than 3, or the combined coefficient is
    /// (near) zero — in which case the strain↔temperature map is singular.
    pub fn validate(&self) -> KwaversResult<()> {
        if self.sound_speed <= 0.0 {
            return Err(ValidationError::InvalidValue {
                parameter: "sound_speed".to_owned(),
                value: self.sound_speed,
                reason: "must be positive".to_owned(),
            }
            .into());
        }
        if self.strain_window < 3 || self.strain_window.is_multiple_of(2) {
            return Err(ValidationError::InvalidValue {
                parameter: "strain_window".to_owned(),
                value: self.strain_window as f64,
                reason: "must be odd and >= 3".to_owned(),
            }
            .into());
        }
        let k_t = self.combined_coefficient();
        if k_t.abs() < f64::EPSILON {
            return Err(ValidationError::InvalidValue {
                parameter: "combined_coefficient".to_owned(),
                value: k_t,
                reason: "thermoacoustic coefficient (α_th − β_c) is singular".to_owned(),
            }
            .into());
        }
        Ok(())
    }

    /// Fractional sound-speed temperature coefficient `β_c = (1/c₀)·(dc/dT)` [1/°C].
    #[must_use]
    pub fn sound_speed_coefficient(&self) -> f64 {
        self.dc_dt / self.sound_speed
    }

    /// Combined thermoacoustic strain coefficient `k_T = α_th − β_c` [1/°C].
    ///
    /// Apparent thermal strain per unit temperature change.
    #[must_use]
    pub fn combined_coefficient(&self) -> f64 {
        self.thermal_expansion - self.sound_speed_coefficient()
    }

    /// Convert a measured thermal strain to a temperature change [°C].
    ///
    /// `ΔT = ε_T / k_T`.
    #[must_use]
    pub fn temperature_from_strain(&self, strain: f64) -> f64 {
        strain / self.combined_coefficient()
    }

    /// Axial sample spacing (m) for an RF line sampled at `sampling_rate` (Hz).
    ///
    /// Each fast-time sample corresponds to a round-trip increment of
    /// `Δz = c₀ / (2·f_s)`.
    #[must_use]
    pub fn axial_sample_spacing(&self, sampling_rate: f64) -> f64 {
        self.sound_speed / (2.0 * sampling_rate)
    }
}
