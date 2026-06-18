//! Narrowband subspace spatial spectra: Eigenspace-MV and MUSIC.
//!
//! These are the passive-acoustic-mapping localizers of **Theorem 22.2**
//! (`docs/book/passive_acoustic_mapping.md`). Both operate on the Hermitian
//! cross-spectral matrix `R = (1/K) Σ x_k x_kᴴ` formed from narrowband complex
//! snapshots, eigendecompose it to partition the rank-`K` signal subspace `U_s`
//! from the noise subspace `U_n`, and evaluate a localization map at a candidate
//! focus `r_f` via the phase-only steering vector `a(r_f) = exp(-j 2π f τ(r_f))`:
//!
//! - **Eigenspace MV (signal-subspace projector):** `b_ES(r_f) = |aᴴ P_s a|²`,
//!   `P_s = U_s U_sᴴ` — large where `a` lies in the source subspace.
//! - **MUSIC (noise-subspace pseudospectrum):** `P_MUSIC(r_f) = 1 / ‖U_nᴴ a‖²`
//!   — sharp peaks at the true source locations.
//!
//! SSOT reuse: snapshot formation ([`extract_narrowband_snapshots`]), covariance
//! ([`estimate_sample_covariance`]), steering ([`NarrowbandSteering`]), and the
//! eigendecomposition itself ([`EigenspaceMV`] / [`MUSIC`] in
//! `beamforming::adaptive::subspace`). This module only wires the PAM data flow;
//! it does not re-derive any linear algebra.

use crate::signal_processing::beamforming::adaptive::subspace::{EigenspaceMV, MUSIC};
use crate::signal_processing::beamforming::covariance::estimate_sample_covariance;
use crate::signal_processing::beamforming::narrowband::snapshots::{
    extract_narrowband_snapshots, SnapshotScenario, SnapshotSelection,
};
use crate::signal_processing::beamforming::narrowband::steering::NarrowbandSteering;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::{Array2, Array3};
use num_complex::Complex64;

/// Method selector for the subspace localization map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubspaceMethod {
    /// Eigenspace minimum-variance signal-subspace projector `|aᴴ P_s a|²`.
    EigenspaceMv,
    /// MUSIC noise-subspace pseudospectrum `1 / ‖U_nᴴ a‖²`.
    Music,
}

/// Configuration for a narrowband subspace spatial spectrum.
#[derive(Debug, Clone)]
pub struct SubspaceSpectrumConfig {
    /// Localization-map evaluation frequency `f` \[Hz] (cavitation-emission band centre).
    pub frequency_hz: f64,
    /// Sampling frequency of the time-domain sensor data \[Hz].
    pub sampling_frequency_hz: f64,
    /// Speed of sound used for time-of-flight steering \[m/s].
    pub sound_speed: f64,
    /// Number of sources `K` = signal-subspace dimension (must satisfy `0 < K < N`).
    pub num_sources: usize,
    /// Diagonal loading `ε ≥ 0` added to `R` before eigendecomposition.
    pub diagonal_loading: f64,
}

impl SubspaceSpectrumConfig {
    fn validate(&self, n_sensors: usize) -> KwaversResult<()> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "SubspaceSpectrumConfig: frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if !self.sampling_frequency_hz.is_finite() || self.sampling_frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "SubspaceSpectrumConfig: sampling_frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if !self.sound_speed.is_finite() || self.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "SubspaceSpectrumConfig: sound_speed must be finite and > 0".to_owned(),
            ));
        }
        if !self.diagonal_loading.is_finite() || self.diagonal_loading < 0.0 {
            return Err(KwaversError::InvalidInput(
                "SubspaceSpectrumConfig: diagonal_loading must be finite and >= 0".to_owned(),
            ));
        }
        if self.num_sources == 0 || self.num_sources >= n_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "SubspaceSpectrumConfig: num_sources {} must satisfy 0 < K < N (={n_sensors})",
                self.num_sources
            )));
        }
        Ok(())
    }
}

/// Form the Hermitian cross-spectral matrix `R` from time-domain sensor data.
///
/// Reuses the SSOT narrowband snapshot extractor and covariance estimator; the
/// only PAM-specific choice here is the auto snapshot scenario (robust, narrow
/// fractional bandwidth) at the cavitation-emission frequency.
fn cross_spectral_matrix(
    sensor_data: &Array3<f64>,
    cfg: &SubspaceSpectrumConfig,
) -> KwaversResult<Array2<Complex64>> {
    let selection = SnapshotSelection::Auto(SnapshotScenario {
        frequency_hz: cfg.frequency_hz,
        sampling_frequency_hz: cfg.sampling_frequency_hz,
        fractional_bandwidth: Some(0.05),
        prefer_robustness: true,
        prefer_time_resolution: false,
    });
    let snapshots = extract_narrowband_snapshots(sensor_data, &selection)?;
    estimate_sample_covariance(&snapshots, cfg.diagonal_loading)
}

/// Validate shapes shared by both subspace spectra and return `n_sensors`.
fn validate_inputs(
    sensor_data: &Array3<f64>,
    sensor_positions: &[[f64; 3]],
    candidate: [f64; 3],
    cfg: &SubspaceSpectrumConfig,
) -> KwaversResult<usize> {
    let (n_sensors, channels, n_samples) = sensor_data.dim();
    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "subspace spectrum expects sensor_data shape (n_sensors, 1, n_samples); got channels={channels}"
        )));
    }
    if n_sensors == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "subspace spectrum requires n_sensors > 0 and n_samples > 0".to_owned(),
        ));
    }
    if sensor_positions.len() != n_sensors {
        return Err(KwaversError::InvalidInput(format!(
            "subspace spectrum sensor_positions len ({}) != n_sensors ({n_sensors})",
            sensor_positions.len()
        )));
    }
    if candidate.iter().any(|v| !v.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "subspace spectrum: candidate must be finite".to_owned(),
        ));
    }
    cfg.validate(n_sensors)?;
    Ok(n_sensors)
}

/// Eigenspace-MV signal-subspace localization map at a single candidate point.
///
/// `b_ES(r_f) = |aᴴ(r_f) P_s a(r_f)|²` (Theorem 22.2).
///
/// # Errors
/// - [`KwaversError::InvalidInput`] for shape/parameter violations.
/// - Propagates snapshot-extraction, covariance, steering, and eigendecomposition errors.
pub fn eigenspace_mv_spatial_spectrum_point(
    sensor_data: &Array3<f64>,
    sensor_positions: &[[f64; 3]],
    candidate: [f64; 3],
    cfg: &SubspaceSpectrumConfig,
) -> KwaversResult<f64> {
    validate_inputs(sensor_data, sensor_positions, candidate, cfg)?;
    let r = cross_spectral_matrix(sensor_data, cfg)?;
    let steering = NarrowbandSteering::new(sensor_positions.to_vec(), cfg.sound_speed)?;
    let a = steering
        .steering_vector_point(candidate, cfg.frequency_hz)?
        .into_array();
    EigenspaceMV::new(cfg.num_sources).signal_subspace_response(&r, &a)
}

/// MUSIC noise-subspace pseudospectrum at a single candidate point.
///
/// `P_MUSIC(r_f) = 1 / ‖U_nᴴ a(r_f)‖²` (Theorem 22.2).
///
/// # Errors
/// - [`KwaversError::InvalidInput`] for shape/parameter violations.
/// - Propagates snapshot-extraction, covariance, steering, and eigendecomposition errors.
pub fn music_spatial_spectrum_point(
    sensor_data: &Array3<f64>,
    sensor_positions: &[[f64; 3]],
    candidate: [f64; 3],
    cfg: &SubspaceSpectrumConfig,
) -> KwaversResult<f64> {
    validate_inputs(sensor_data, sensor_positions, candidate, cfg)?;
    let r = cross_spectral_matrix(sensor_data, cfg)?;
    let steering = NarrowbandSteering::new(sensor_positions.to_vec(), cfg.sound_speed)?;
    let a = steering
        .steering_vector_point(candidate, cfg.frequency_hz)?
        .into_array();
    MUSIC::new(cfg.num_sources).pseudospectrum(&r, &a)
}

/// Dispatch helper used by the PAM mapper: evaluate the selected subspace map.
///
/// # Errors
/// Propagates the errors of the selected spatial-spectrum function.
pub fn subspace_spatial_spectrum_point(
    method: SubspaceMethod,
    sensor_data: &Array3<f64>,
    sensor_positions: &[[f64; 3]],
    candidate: [f64; 3],
    cfg: &SubspaceSpectrumConfig,
) -> KwaversResult<f64> {
    match method {
        SubspaceMethod::EigenspaceMv => {
            eigenspace_mv_spatial_spectrum_point(sensor_data, sensor_positions, candidate, cfg)
        }
        SubspaceMethod::Music => {
            music_spatial_spectrum_point(sensor_data, sensor_positions, candidate, cfg)
        }
    }
}

#[cfg(test)]
mod tests;
