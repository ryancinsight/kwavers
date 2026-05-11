//! Data types for the ULM microbubble detection pipeline.

/// One detected and localized microbubble in a single frame.
#[derive(Debug, Clone, PartialEq)]
pub struct BubbleDetection {
    /// Sub-pixel lateral position [pixels or m, same units as input grid]
    pub x: f64,
    /// Sub-pixel axial position [pixels or m]
    pub z: f64,
    /// Fitted Gaussian amplitude [a.u.]
    pub amplitude: f64,
    /// Fitted Gaussian width σ [same units as x/z]
    pub sigma: f64,
    /// Background level [a.u.]
    pub background: f64,
    /// Frame index
    pub frame: usize,
}

/// Configuration for SVD clutter filtering.
#[derive(Debug, Clone, Default)]
pub struct SvdClutterConfig {
    /// Override automatic SVHT threshold with fixed k (0 = automatic).
    pub fixed_clutter_rank: usize,
    /// Safety margin added to SVHT k (default 0).
    pub rank_margin: usize,
}

/// Configuration for Gaussian localization.
#[derive(Debug, Clone)]
pub struct LocalizationConfig {
    /// Detection threshold: candidate_amplitude > threshold_sigma_multiplier × noise_std
    pub threshold_sigma_multiplier: f64,
    /// Minimum PSF width accepted (pixels)
    pub min_sigma_px: f64,
    /// Maximum PSF width accepted (pixels)
    pub max_sigma_px: f64,
    /// Minimum amplitude-to-background ratio for acceptance
    pub min_snr_ratio: f64,
    /// Half-side of the local neighbourhood used for Gaussian fit (default 2 → 5×5)
    pub fit_half_width: usize,
    /// Maximum Gauss-Newton iterations
    pub max_gauss_newton_iter: usize,
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self {
            threshold_sigma_multiplier: 3.0,
            min_sigma_px: 0.3,
            max_sigma_px: 3.0,
            min_snr_ratio: 2.0,
            fit_half_width: 2,
            max_gauss_newton_iter: 20,
        }
    }
}
