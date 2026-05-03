//! `SuperResReconstructor` — accumulates bubble localizations into SR images.

use super::types::{RenderMode, SuperResConfig};
use crate::clinical::imaging::functional_ultrasound::ulm::tracking::BubbleTrack;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;

/// Accumulates bubble localizations from completed tracks into a super-resolved image.
///
/// ## Coordinate Convention
///
/// x ∈ [0, x_extent), z ∈ [0, z_extent). Positions outside this range are silently
/// discarded.
#[derive(Debug)]
pub struct SuperResReconstructor {
    pub(super) config: SuperResConfig,
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
    pub(crate) fn smooth_track(&self, track: &BubbleTrack) -> Vec<(f64, f64)> {
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
