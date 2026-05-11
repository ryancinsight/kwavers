//! Configuration types for ULM super-resolution reconstruction.

/// Configuration for super-resolution reconstruction.
#[derive(Debug, Clone)]
pub struct SuperResConfig {
    /// Physical extent of the lateral (x) dimension (m).
    pub x_extent: f64,
    /// Physical extent of the axial (z) dimension (m).
    pub z_extent: f64,
    /// SR pixel size (m). Default: 5 μm (Nouhoum et al. 2021).
    pub pixel_size: f64,
    /// Gaussian rendering width σ_loc (m). Default: 5 μm.
    pub gauss_sigma: f64,
    /// Sliding average half-width for trajectory smoothing (0 = no smoothing).
    pub smooth_halfwidth: usize,
    /// Rendering mode.
    pub mode: RenderMode,
    /// Total acquisition duration (s) for density normalization.
    /// `None` disables density normalization.
    pub total_time_s: Option<f64>,
}

impl Default for SuperResConfig {
    fn default() -> Self {
        Self {
            x_extent: 0.01,      // 10 mm
            z_extent: 0.012,     // 12 mm
            pixel_size: 5e-6,    // 5 μm
            gauss_sigma: 5e-6,   // 5 μm
            smooth_halfwidth: 2, // ±2 frame sliding average
            mode: RenderMode::GaussianSplat,
            total_time_s: None,
        }
    }
}

/// Super-resolution rendering mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderMode {
    /// Integer histogram accumulation — fastest; counts per bin.
    Histogram,
    /// Gaussian kernel density splatting — smoother; approximates the SR PSF.
    GaussianSplat,
}
