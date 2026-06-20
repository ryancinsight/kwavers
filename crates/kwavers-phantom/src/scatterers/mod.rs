//! Discrete point-scatterer clouds and synthetic-aperture RF synthesis.
//!
//! This is the kwavers analogue of the Field II core abstraction: a tissue model
//! as a cloud of discrete point scatterers, each with a position and a scattering
//! amplitude, from which per-element radio-frequency (RF) echo traces are
//! synthesized.
//!
//! # Echo model
//!
//! [`ScattererCloud::synthesize_rf`] uses the **monostatic synthetic-aperture**
//! point-element model: each transducer element independently transmits and
//! receives. For element at position `e` and scatterer `s`,
//!
//! ```text
//! RF_e(t) = Σ_s ( a_s / r² ) · pulse( t − 2r/c ),     r = ‖e − s‖
//! ```
//!
//! where `a_s` is the scatterer amplitude, `2r/c` the pulse-echo time of flight,
//! and `1/r²` the round-trip spherical spreading (`1/r` on transmit and `1/r` on
//! receive). This is the correct far-field point-element limit and the basis of
//! synthetic-aperture imaging.
//!
//! Optional power-law tissue attenuation scales each echo by the round-trip
//! amplitude factor `exp(−α(f₀)·2r)` (see [`RfSynthesisConfig`]); with it
//! disabled the model reduces to pure spreading.
//!
//! It does **not** model finite-aperture diffraction — the Tupholme–Stepanishen
//! spatial impulse response that Field II convolves in for extended elements.
//! That refines the near-field response and is a tracked follow-up; the
//! point-element model is exact for point elements and far-field scatterers.
//!
//! # References
//! - Jensen, J. A. (1991). "A model for the propagation and scattering of
//!   ultrasound in tissue." *J. Acoust. Soc. Am.* 89(1), 182–190 (Field II model).
//! - Jensen, J. A., & Svendsen, N. B. (1992). "Calculation of pressure fields from
//!   arbitrarily shaped, apodized, and excited ultrasound transducers."
//!   *IEEE Trans. UFFC* 39(2), 262–267 (spatial impulse response).

use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array2;

/// A single discrete point scatterer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointScatterer {
    /// Position [m].
    pub position: [f64; 3],
    /// Scattering amplitude (dimensionless reflectivity; sign carries phase).
    pub amplitude: f64,
}

/// A cloud of discrete point scatterers (the Field II tissue model).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScattererCloud {
    scatterers: Vec<PointScatterer>,
}

/// Configuration for [`ScattererCloud::synthesize_rf`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RfSynthesisConfig {
    /// Sound speed `c` [m/s].
    pub sound_speed: f64,
    /// Sampling frequency `fs` [Hz].
    pub sampling_frequency: f64,
    /// Number of output time samples per element.
    pub num_samples: usize,
    /// Minimum element–scatterer distance [m]; closer scatterers are skipped to
    /// avoid the `1/r²` singularity (a scatterer inside the element footprint is
    /// outside the far-field point model).
    pub min_distance: f64,
    /// Power-law tissue attenuation `α₀` in dB/(cm·MHz). `0` ⇒ lossless (the
    /// pure spreading model). When positive, each echo is scaled by the
    /// round-trip amplitude factor `exp(−α(f₀)·2r)` evaluated at
    /// `center_frequency_hz`.
    pub attenuation_db_cm_mhz: f64,
    /// Pulse centre frequency `f₀` [Hz] for the power-law attenuation. Only used
    /// when `attenuation_db_cm_mhz > 0`.
    pub center_frequency_hz: f64,
}

impl ScattererCloud {
    /// Empty cloud.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from explicit positions and matching amplitudes.
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if the lengths differ.
    pub fn from_points(positions: &[[f64; 3]], amplitudes: &[f64]) -> KwaversResult<Self> {
        if positions.len() != amplitudes.len() {
            return Err(KwaversError::InvalidInput(format!(
                "ScattererCloud::from_points: {} positions vs {} amplitudes",
                positions.len(),
                amplitudes.len()
            )));
        }
        Ok(Self {
            scatterers: positions
                .iter()
                .zip(amplitudes)
                .map(|(&position, &amplitude)| PointScatterer {
                    position,
                    amplitude,
                })
                .collect(),
        })
    }

    /// Append a scatterer.
    pub fn push(&mut self, position: [f64; 3], amplitude: f64) {
        self.scatterers.push(PointScatterer {
            position,
            amplitude,
        });
    }

    /// The scatterers.
    #[must_use]
    pub fn scatterers(&self) -> &[PointScatterer] {
        &self.scatterers
    }

    /// Number of scatterers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scatterers.len()
    }

    /// Whether the cloud is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scatterers.is_empty()
    }

    /// Synthesize per-element monostatic pulse-echo RF, shape `(n_elements, num_samples)`.
    ///
    /// See the [module docs](self) for the echo model. Each element transmits and
    /// receives the `pulse` (a sampled excitation at `config.sampling_frequency`).
    ///
    /// # Errors
    /// - [`KwaversError::InvalidInput`] if `sound_speed` or `sampling_frequency`
    ///   is non-finite/`≤ 0`, `num_samples == 0`, `pulse` is empty, or
    ///   `min_distance < 0`.
    pub fn synthesize_rf(
        &self,
        element_positions: &[[f64; 3]],
        pulse: &[f64],
        config: &RfSynthesisConfig,
    ) -> KwaversResult<Array2<f64>> {
        if !config.sound_speed.is_finite() || config.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "synthesize_rf requires sound_speed > 0, got {}",
                config.sound_speed
            )));
        }
        if !config.sampling_frequency.is_finite() || config.sampling_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "synthesize_rf requires sampling_frequency > 0, got {}",
                config.sampling_frequency
            )));
        }
        if config.num_samples == 0 {
            return Err(KwaversError::InvalidInput(
                "synthesize_rf requires num_samples > 0".to_owned(),
            ));
        }
        if pulse.is_empty() {
            return Err(KwaversError::InvalidInput(
                "synthesize_rf requires a non-empty pulse".to_owned(),
            ));
        }
        if !config.min_distance.is_finite() || config.min_distance < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "synthesize_rf requires min_distance >= 0, got {}",
                config.min_distance
            )));
        }
        if !config.attenuation_db_cm_mhz.is_finite() || config.attenuation_db_cm_mhz < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "synthesize_rf requires attenuation_db_cm_mhz >= 0, got {}",
                config.attenuation_db_cm_mhz
            )));
        }
        if config.attenuation_db_cm_mhz > 0.0
            && (!config.center_frequency_hz.is_finite() || config.center_frequency_hz <= 0.0)
        {
            return Err(KwaversError::InvalidInput(format!(
                "synthesize_rf requires center_frequency_hz > 0 when attenuation is enabled, got {}",
                config.center_frequency_hz
            )));
        }

        let n_elements = element_positions.len();
        let mut rf = Array2::<f64>::zeros((n_elements, config.num_samples));
        let two_over_c = 2.0 / config.sound_speed;
        // Power-law attenuation coefficient at the centre frequency, in Np/m:
        // α₀[dB/(cm·MHz)] · f₀[MHz] · 100[cm/m] / NP_TO_DB[dB/Np].  Zero ⇒ lossless.
        let alpha_np_m = if config.attenuation_db_cm_mhz > 0.0 {
            config.attenuation_db_cm_mhz * (config.center_frequency_hz / MHZ_TO_HZ) * 100.0
                / NP_TO_DB
        } else {
            0.0
        };

        for (e_idx, &e) in element_positions.iter().enumerate() {
            for s in &self.scatterers {
                let dx = e[0] - s.position[0];
                let dy = e[1] - s.position[1];
                let dz = e[2] - s.position[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r < config.min_distance || r <= 0.0 {
                    continue;
                }
                // Round-trip spherical spreading (1/r transmit × 1/r receive),
                // times the round-trip power-law attenuation exp(−α·2r) (1 when
                // lossless).
                let mut amp = s.amplitude / (r * r);
                if alpha_np_m > 0.0 {
                    amp *= (-alpha_np_m * 2.0 * r).exp();
                }
                let tof = r * two_over_c; // 2r/c
                let n_delay = (tof * config.sampling_frequency).round();
                if !n_delay.is_finite() || n_delay < 0.0 {
                    continue;
                }
                let n_delay = n_delay as usize;
                for (k, &p) in pulse.iter().enumerate() {
                    let n = n_delay + k;
                    if n < config.num_samples {
                        rf[[e_idx, n]] += amp * p;
                    }
                }
            }
        }
        Ok(rf)
    }
}

#[cfg(test)]
mod tests;
