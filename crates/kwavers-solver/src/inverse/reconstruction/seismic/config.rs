//! Configuration structures for seismic imaging algorithms

use crate::reconstruction::{
    ReconstructionConfig, ReconstructionFilterType, ReconstructionInterpolationMethod,
};
use serde::{Deserialize, Serialize};

// Import constants from the constants module
use super::constants::{
    DEFAULT_FWI_ITERATIONS, DEFAULT_FWI_TOLERANCE, DEFAULT_REGULARIZATION_LAMBDA,
    DEFAULT_RICKER_FREQUENCY, DEFAULT_TIME_STEP,
};

/// Seismic imaging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeismicImagingConfig {
    /// Base reconstruction configuration
    pub base_config: ReconstructionConfig,
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Time step (seconds) — used by RTM forward/backward propagation and
    /// by the imaging-condition temporal derivatives.  The wavefield-update
    /// CFL condition is `dt ≤ dx · CFL / (c_max · √D)` for D spatial
    /// dimensions; callers are responsible for satisfying CFL.
    pub dt: f64,
    /// Dominant source frequency (Hz) for the RTM forward-propagation
    /// Ricker wavelet.  Must be band-limited below the grid Nyquist
    /// `c_min / (2·dx)` and resolved by at least 2–3 voxels per wavelength.
    pub source_frequency_hz: f64,
    /// Number of FWI iterations
    pub fwi_iterations: usize,
    /// Convergence tolerance for FWI
    pub fwi_tolerance: f64,
    /// Regularization parameter for smoothness
    pub regularization_lambda: f64,
    /// Enable multi-scale approach
    pub enable_multiscale: bool,
    /// Frequency bands for multi-scale FWI
    pub frequency_bands: Vec<(f64, f64)>,
    /// RTM imaging condition
    pub rtm_imaging_condition: RtmImagingCondition,
    /// Enable attenuation modeling
    pub enable_attenuation: bool,
    /// Anisotropy parameters (if applicable)
    pub anisotropy_params: Option<AnisotropyParameters>,
}

/// RTM imaging conditions based on established literature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RtmImagingCondition {
    /// Zero-lag cross-correlation (Claerbout, 1985)
    /// I(x) = ∫ S(x,t) * R(x,t) dt
    ZeroLag,

    /// Normalized cross-correlation (Valenciano et al., 2006)
    /// I(x) = ∫ S(x,t) * R(x,t) dt / (|S| * |R|)
    Normalized,

    /// Laplacian imaging condition (Zhang & Sun, 2009)
    /// I(x) = ∫ ∇²S(x,t) * R(x,t) dt
    Laplacian,

    /// Energy-normalized imaging condition (Schleicher et al., 2008)
    /// I(x) = ∫ S(x,t) * R(x,t) dt / ∫ S(x,t)² dt
    EnergyNormalized,

    /// Source-normalized imaging condition (Guitton et al., 2007)
    /// I(x) = ∫ ∂S/∂t * R(x,t) dt
    SourceNormalized,

    /// Poynting vector imaging condition (Yoon et al., 2004)
    /// I(x) = ∫ ∇S(x,t) · ∇R(x,t) dt
    Poynting,
}

/// Anisotropy parameters for VTI media
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnisotropyParameters {
    /// Thomsen parameter epsilon
    pub epsilon: f64,
    /// Thomsen parameter delta
    pub delta: f64,
    /// Thomsen parameter gamma
    pub gamma: f64,
}

impl Default for SeismicImagingConfig {
    fn default() -> Self {
        Self {
            base_config: ReconstructionConfig {
                sound_speed: 3000.0,                         // Typical crustal velocity
                sampling_frequency: 1.0 / DEFAULT_TIME_STEP, // 2000 Hz
                algorithm: crate::reconstruction::ReconstructionAlgorithm::FullWaveformInversion,
                filter: ReconstructionFilterType::None,
                interpolation: ReconstructionInterpolationMethod::Linear,
            },
            nx: 100,
            ny: 100,
            nz: 100,
            dt: DEFAULT_TIME_STEP,
            source_frequency_hz: DEFAULT_RICKER_FREQUENCY,
            fwi_iterations: DEFAULT_FWI_ITERATIONS,
            fwi_tolerance: DEFAULT_FWI_TOLERANCE,
            regularization_lambda: DEFAULT_REGULARIZATION_LAMBDA,
            enable_multiscale: true,
            frequency_bands: vec![(2.0, 8.0), (8.0, 15.0), (15.0, 30.0)],
            rtm_imaging_condition: RtmImagingCondition::ZeroLag,
            enable_attenuation: false,
            anisotropy_params: None,
        }
    }
}
