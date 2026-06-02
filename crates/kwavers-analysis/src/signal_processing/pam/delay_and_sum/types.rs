use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use serde::{Deserialize, Serialize};

pub use kwavers_math::signal::ApodizationType;

/// Spatial-map combination strategy for passive acoustic mapping.
///
/// Both modes share the time-aligned (delay-and-interpolate) front end; they
/// differ only in how the apodized, delayed per-sensor samples are combined
/// into the per-pixel value.
///
/// # References
/// - Gyöngy & Coussios (2010), IEEE TBME 57(1): delay-and-sum PAM.
/// - Matrone et al. (2015), IEEE TMI 34(4): the sign-preserving
///   delay-multiply-and-sum (DMAS) beamformer.
/// - Recent PAM work (2024–2025) applies DMAS to passive cavitation mapping for
///   sharper mainlobes and lower sidelobes than DAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PamImagingMode {
    /// Delay-and-sum: pixel value `I = ⟨(Σᵢ wᵢ sᵢ′)²⟩`.
    DelayAndSum,
    /// Sign-preserving delay-multiply-and-sum: with `ŝᵢ = sign(wᵢsᵢ′)·√|wᵢsᵢ′|`,
    /// the DMAS sample is `y = Σ_{i<j} ŝᵢŝⱼ = ½[(Σᵢ ŝᵢ)² − Σᵢ ŝᵢ²]` and the
    /// pixel value is `I = ⟨y²⟩`. The pairwise correlation rejects incoherent
    /// (off-focus) energy more strongly than DAS.
    DelayMultiplyAndSum,
}

/// Configuration for delay-and-sum PAM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayAndSumConfig {
    /// Sound speed in medium (m/s).
    pub sound_speed: f64,
    /// Sampling frequency (Hz).
    pub sampling_frequency: f64,
    /// Detection threshold (multiple of noise floor).
    pub detection_threshold: f64,
    /// Temporal window size (samples).
    pub window_size: usize,
    /// Apodization window type.
    pub apodization: ApodizationType,
    /// Enable coherence factor weighting.
    pub coherence_weighting: bool,
}

impl Default for DelayAndSumConfig {
    fn default() -> Self {
        Self {
            sound_speed: SOUND_SPEED_TISSUE,
            sampling_frequency: 5.0 * MHZ_TO_HZ,
            detection_threshold: 3.0,
            window_size: 512,
            apodization: ApodizationType::Hamming,
            coherence_weighting: true,
        }
    }
}

/// Detected cavitation event.
#[derive(Debug, Clone)]
pub struct PamCavitationEvent {
    /// 3D position (m).
    pub position: [f64; 3],
    /// Intensity (arbitrary units).
    pub intensity: f64,
    /// Time of occurrence (s).
    pub time: f64,
    /// Coherence factor (0–1).
    pub coherence: f64,
    /// Peak frequency content (Hz); `None` if not estimated.
    pub peak_frequency: Option<f64>,
}
