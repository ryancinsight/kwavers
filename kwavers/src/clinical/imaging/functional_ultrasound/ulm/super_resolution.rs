//! Super-Resolution Reconstruction for ULM
//!
//! Accumulates microbubble localizations into a high-resolution density image,
//! achieving resolution ~5 μm — ~20× finer than the diffraction-limited acquisition.
//!
//! ## Algorithm
//!
//! ### Histogram Accumulation (Errico et al. 2015)
//!
//! Each localized bubble position (x_k, z_k) votes into one bin of the SR grid:
//! ```text
//! I_SR[m, n] = Σ_k δ(⌊x_k/d_SR⌋ = m, ⌊z_k/d_SR⌋ = n)
//! ```
//! where d_SR is the SR pixel size (5 μm).
//!
//! ### Gaussian Rendering (Betzig et al. 2006; Christensen-Jeffries et al. 2020)
//!
//! Each localization is rendered as a 2D isotropic Gaussian with σ = σ_loc:
//! ```text
//! I_SR[x, z] += exp(−[(x − x_k)² + (z − z_k)²] / (2σ_loc²))
//! ```
//! This is equivalent to kernel density estimation with a Gaussian kernel, and
//! approximates the theoretical super-resolution PSF.
//!
//! Support radius = 3σ (captures 99.7% of Gaussian mass per dimension).
//!
//! ### Sliding Average Trajectory Smoothing (Nouhoum et al. 2021)
//!
//! Before accumulation, each track can be smoothed to reduce per-frame localization
//! noise using a causal-symmetric box filter with half-width w:
//! ```text
//! x̄_k = (1 / |W_k|) Σ_{j ∈ W_k} x_j,   W_k = [max(0, k−w), min(n−1, k+w)]
//! ```
//!
//! ### Density Normalization
//!
//! For quantitative comparison across experiments with different acquisition durations:
//! ```text
//! I_density[m, n] = I_SR[m, n] / T_total    [events / second]
//! ```
//!
//! ## References
//!
//! - Errico, C., et al. (2015). Ultrafast ultrasound localization microscopy for deep
//!   super-resolution vascular imaging. *Nature* 527:499–502. DOI: 10.1038/nature16066
//! - Betzig, E., et al. (2006). Imaging intracellular fluorescent proteins at nanometer
//!   resolution. *Science* 313(5793):1642–1645. DOI: 10.1126/science.1127344
//! - Christensen-Jeffries, K., et al. (2020). Super-resolution ultrasound imaging.
//!   *Ultrasound Med. Biol.* 46(4):865–891. DOI: 10.1016/j.ultrasmedbio.2019.11.013
//! - Nouhoum, M., et al. (2021). A functional ultrasound brain GPS for automatic
//!   vascular-based neuronavigation. *Sci. Rep.* 11:15197. DOI: 10.1038/s41598-021-94764-7

use crate::clinical::imaging::functional_ultrasound::ulm::tracking::BubbleTrack;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for super-resolution reconstruction.
#[derive(Debug, Clone)]
pub struct SuperResConfig {
    /// Physical extent of the lateral (x) dimension [m].
    pub x_extent: f64,
    /// Physical extent of the axial (z) dimension [m].
    pub z_extent: f64,
    /// SR pixel size [m]. Default: 5 μm (Nouhoum et al. 2021).
    pub pixel_size: f64,
    /// Gaussian rendering width σ_loc [m]. Default: 5 μm.
    pub gauss_sigma: f64,
    /// Sliding average half-width for trajectory smoothing (0 = no smoothing).
    pub smooth_halfwidth: usize,
    /// Rendering mode.
    pub mode: RenderMode,
    /// Total acquisition duration [s] for density normalization.
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

// ─── Reconstructor ────────────────────────────────────────────────────────────

/// Accumulates bubble localizations from completed tracks into a super-resolved image.
///
/// ## Coordinate Convention
///
/// x ∈ [0, x_extent), z ∈ [0, z_extent). Positions outside this range are silently
/// discarded.
#[derive(Debug)]
pub struct SuperResReconstructor {
    config: SuperResConfig,
    /// SR image grid: `image[[ix, iz]]` = accumulated intensity.
    image: Array2<f64>,
    nx: usize,
    nz: usize,
}

impl SuperResReconstructor {
    /// Create a new reconstructor.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] if `pixel_size ≤ 0`, extents ≤ 0,
    /// or if the resulting grid would have zero size.
    pub fn new(config: SuperResConfig) -> KwaversResult<Self> {
        if config.pixel_size <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "pixel_size must be > 0".to_string(),
            ));
        }
        if config.x_extent <= 0.0 || config.z_extent <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "image extents must be > 0".to_string(),
            ));
        }
        let nx = (config.x_extent / config.pixel_size).ceil() as usize;
        let nz = (config.z_extent / config.pixel_size).ceil() as usize;
        if nx == 0 || nz == 0 {
            return Err(KwaversError::InvalidInput(
                "resulting grid has zero size".to_string(),
            ));
        }
        Ok(Self {
            config,
            image: Array2::zeros((nx, nz)),
            nx,
            nz,
        })
    }

    /// Accumulate all localizations from `tracks` into the SR image.
    ///
    /// For each track:
    /// 1. Apply sliding-average smoothing over the trajectory.
    /// 2. Render each smoothed position into the grid using the configured mode.
    pub fn accumulate(&mut self, tracks: &[BubbleTrack]) {
        for track in tracks {
            let positions = self.smooth_track(track);
            for (x, z) in positions {
                match self.config.mode {
                    RenderMode::Histogram => self.accumulate_histogram(x, z),
                    RenderMode::GaussianSplat => self.accumulate_gaussian(x, z),
                }
            }
        }
    }

    /// Return the accumulated SR image (raw counts or kernel-density values).
    #[must_use]
    pub fn image(&self) -> &Array2<f64> {
        &self.image
    }

    /// Return a density-normalized SR image [events / second].
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] if `total_time_s` was not set or is ≤ 0.
    pub fn density_image(&self) -> KwaversResult<Array2<f64>> {
        let t = self.config.total_time_s.ok_or_else(|| {
            KwaversError::InvalidInput(
                "total_time_s must be set in SuperResConfig for density normalization".to_string(),
            )
        })?;
        if t <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "total_time_s must be > 0".to_string(),
            ));
        }
        Ok(self.image.mapv(|v| v / t))
    }

    /// Return the grid dimensions (nx, nz).
    #[must_use]
    pub fn grid_size(&self) -> (usize, usize) {
        (self.nx, self.nz)
    }

    // ─── Private helpers ──────────────────────────────────────────────────────

    /// Apply a sliding-average box filter of half-width `w` to the track positions.
    ///
    /// ## Algorithm
    ///
    /// For position k: x̄_k = mean(x[max(0,k−w) .. min(n, k+w+1)]).
    /// If `smooth_halfwidth == 0`, returns positions unchanged.
    #[must_use]
    pub(super) fn smooth_track(&self, track: &BubbleTrack) -> Vec<(f64, f64)> {
        let dets = &track.detections;
        let n = dets.len();
        if n == 0 {
            return Vec::new();
        }
        let w = self.config.smooth_halfwidth;
        if w == 0 {
            return dets.iter().map(|d| (d.x, d.z)).collect();
        }
        (0..n)
            .map(|k| {
                let lo = k.saturating_sub(w);
                let hi = (k + w + 1).min(n);
                let count = (hi - lo) as f64;
                let xm = dets[lo..hi].iter().map(|d| d.x).sum::<f64>() / count;
                let zm = dets[lo..hi].iter().map(|d| d.z).sum::<f64>() / count;
                (xm, zm)
            })
            .collect()
    }

    /// Increment the histogram bin that contains (x, z). Out-of-bounds silently ignored.
    fn accumulate_histogram(&mut self, x: f64, z: f64) {
        if let Some((ix, iz)) = self.to_grid_index(x, z) {
            self.image[[ix, iz]] += 1.0;
        }
    }

    /// Add a 2D isotropic Gaussian centred at (x, z) to the SR image.
    ///
    /// ## Kernel
    ///
    /// ```text
    /// K(Δx, Δz) = exp(−(Δx² + Δz²) / (2σ²))
    /// ```
    ///
    /// Support: ±3σ (99.7% of Gaussian mass). Not pre-normalized so relative
    /// intensities are preserved across the image.
    fn accumulate_gaussian(&mut self, x: f64, z: f64) {
        let sigma = self.config.gauss_sigma;
        let d = self.config.pixel_size;
        let radius = ((3.0 * sigma / d).ceil() as usize).max(1);

        if x < 0.0 || z < 0.0 {
            return;
        }
        let cx = x / d;
        let cz = z / d;
        let ix_c = cx as usize;
        let iz_c = cz as usize;

        let ix_lo = ix_c.saturating_sub(radius);
        let ix_hi = (ix_c + radius + 1).min(self.nx);
        let iz_lo = iz_c.saturating_sub(radius);
        let iz_hi = (iz_c + radius + 1).min(self.nz);

        let two_sigma_sq = 2.0 * sigma * sigma;

        for ix in ix_lo..ix_hi {
            let dx = (ix as f64 + 0.5) * d - x;
            for iz in iz_lo..iz_hi {
                let dz = (iz as f64 + 0.5) * d - z;
                let w = (-(dx * dx + dz * dz) / two_sigma_sq).exp();
                self.image[[ix, iz]] += w;
            }
        }
    }

    /// Convert physical position (x, z) to grid index (ix, iz), or None if out of bounds.
    #[inline]
    fn to_grid_index(&self, x: f64, z: f64) -> Option<(usize, usize)> {
        if x < 0.0 || z < 0.0 {
            return None;
        }
        let ix = (x / self.config.pixel_size) as usize;
        let iz = (z / self.config.pixel_size) as usize;
        if ix < self.nx && iz < self.nz {
            Some((ix, iz))
        } else {
            None
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clinical::imaging::functional_ultrasound::ulm::microbubble_detection::BubbleDetection;
    use crate::clinical::imaging::functional_ultrasound::ulm::tracking::BubbleTrack;

    fn make_track(positions: &[(f64, f64)]) -> BubbleTrack {
        let dets: Vec<BubbleDetection> = positions
            .iter()
            .enumerate()
            .map(|(i, &(x, z))| BubbleDetection {
                x,
                z,
                amplitude: 1.0,
                sigma: 1.0,
                background: 0.0,
                frame: i,
            })
            .collect();
        BubbleTrack {
            id: 0,
            detections: dets,
            last_frame: positions.len().saturating_sub(1),
            gap: 0,
            active: false,
        }
    }

    #[test]
    fn test_histogram_single_localization() {
        let config = SuperResConfig {
            x_extent: 1e-3,
            z_extent: 1e-3,
            pixel_size: 5e-6,
            mode: RenderMode::Histogram,
            smooth_halfwidth: 0,
            ..Default::default()
        };
        let mut recon = SuperResReconstructor::new(config).unwrap();
        let track = make_track(&[(100e-6, 200e-6)]);
        recon.accumulate(&[track]);

        let ix = (100e-6_f64 / 5e-6) as usize;
        let iz = (200e-6_f64 / 5e-6) as usize;
        assert_eq!(
            recon.image()[[ix, iz]],
            1.0,
            "Single localization should give count=1"
        );

        // Total over entire image must equal number of localizations.
        let total: f64 = recon.image().sum();
        assert!((total - 1.0).abs() < 1e-10, "total={total}");
    }

    #[test]
    fn test_histogram_multiple_localizations() {
        let config = SuperResConfig {
            x_extent: 1e-3,
            z_extent: 1e-3,
            pixel_size: 5e-6,
            mode: RenderMode::Histogram,
            smooth_halfwidth: 0,
            ..Default::default()
        };
        let mut recon = SuperResReconstructor::new(config).unwrap();
        // 3 detections at the same grid cell → count = 3
        let track = make_track(&[(100e-6, 200e-6), (101e-6, 200e-6), (102e-6, 200e-6)]);
        recon.accumulate(&[track]);
        let total: f64 = recon.image().sum();
        assert!((total - 3.0).abs() < 1e-10, "total count must be 3");
    }

    #[test]
    fn test_gaussian_splat_peak_below_one() {
        // A single point localization splatted as Gaussian must spread the mass.
        let config = SuperResConfig {
            x_extent: 500e-6,
            z_extent: 500e-6,
            pixel_size: 5e-6,
            gauss_sigma: 20e-6,
            mode: RenderMode::GaussianSplat,
            smooth_halfwidth: 0,
            ..Default::default()
        };
        let mut recon = SuperResReconstructor::new(config).unwrap();
        let track = make_track(&[(250e-6, 250e-6)]);
        recon.accumulate(&[track]);

        let peak = recon
            .image()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(peak < 1.0, "Gaussian splatted peak {peak:.4} must be < 1");
        assert!(peak > 0.0, "Peak must be positive");
    }

    #[test]
    fn test_density_normalization() {
        let config = SuperResConfig {
            x_extent: 1e-3,
            z_extent: 1e-3,
            pixel_size: 5e-6,
            mode: RenderMode::Histogram,
            smooth_halfwidth: 0,
            total_time_s: Some(2.0),
            ..Default::default()
        };
        let mut recon = SuperResReconstructor::new(config).unwrap();
        let track = make_track(&[(100e-6, 200e-6)]);
        recon.accumulate(&[track]);
        let density = recon.density_image().unwrap();

        let ix = (100e-6_f64 / 5e-6) as usize;
        let iz = (200e-6_f64 / 5e-6) as usize;
        assert!(
            (density[[ix, iz]] - 0.5).abs() < 1e-10,
            "1 count / 2 s = 0.5 Hz, got {}",
            density[[ix, iz]]
        );
    }

    #[test]
    fn test_density_image_error_without_time() {
        let config = SuperResConfig {
            total_time_s: None,
            ..Default::default()
        };
        let recon = SuperResReconstructor::new(config).unwrap();
        assert!(recon.density_image().is_err());
    }

    #[test]
    fn test_sliding_average_smoothing_constant_track() {
        // A constant-position track must be unchanged by any smoothing window.
        let config = SuperResConfig {
            x_extent: 1e-3,
            z_extent: 1e-3,
            pixel_size: 5e-6,
            smooth_halfwidth: 2,
            ..Default::default()
        };
        let recon = SuperResReconstructor::new(config).unwrap();
        let track = make_track(&[(200e-6, 300e-6); 5]);
        let positions = recon.smooth_track(&track);
        for (x, z) in positions {
            assert!((x - 200e-6).abs() < 1e-12, "x should be constant");
            assert!((z - 300e-6).abs() < 1e-12, "z should be constant");
        }
    }

    #[test]
    fn test_sliding_average_reduces_noise() {
        // Alternating positions ±ε around a mean should be smoothed toward the mean.
        let config = SuperResConfig {
            x_extent: 2e-3,
            z_extent: 2e-3,
            pixel_size: 5e-6,
            smooth_halfwidth: 2,
            ..Default::default()
        };
        let recon = SuperResReconstructor::new(config).unwrap();
        let eps = 100e-6;
        let center_x = 1e-3;
        let center_z = 1e-3;
        let raw: Vec<(f64, f64)> = (0..10)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                (center_x + sign * eps, center_z)
            })
            .collect();
        let track = make_track(&raw);
        let smoothed = recon.smooth_track(&track);

        // Interior smoothed values must be strictly closer to center_x than ε.
        for (x, _) in &smoothed[2..8] {
            let dev = (x - center_x).abs();
            assert!(
                dev < eps,
                "Smoothed deviation {dev:.4e} >= raw deviation {eps:.4e}"
            );
        }
    }

    #[test]
    fn test_grid_size_matches_config() {
        let config = SuperResConfig {
            x_extent: 0.01,
            z_extent: 0.012,
            pixel_size: 5e-6,
            ..Default::default()
        };
        let recon = SuperResReconstructor::new(config).unwrap();
        let (nx, nz) = recon.grid_size();
        assert_eq!(nx, 2000, "10 mm / 5 μm = 2000");
        assert_eq!(nz, 2400, "12 mm / 5 μm = 2400");
    }

    #[test]
    fn test_out_of_bounds_localizations_ignored() {
        let config = SuperResConfig {
            x_extent: 1e-3,
            z_extent: 1e-3,
            pixel_size: 5e-6,
            mode: RenderMode::Histogram,
            smooth_halfwidth: 0,
            ..Default::default()
        };
        let mut recon = SuperResReconstructor::new(config).unwrap();
        // Position at x=2 mm > x_extent=1 mm — must be silently discarded.
        let track = make_track(&[(2e-3, 500e-6)]);
        recon.accumulate(&[track]);
        let total: f64 = recon.image().sum();
        assert!(
            (total - 0.0).abs() < 1e-10,
            "Out-of-bounds localization must be ignored"
        );
    }
}
