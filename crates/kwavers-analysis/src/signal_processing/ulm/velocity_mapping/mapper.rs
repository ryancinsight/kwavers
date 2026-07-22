//! `VelocityMapper` — 2D velocity field reconstruction from bubble tracks.

use super::config::VelocityMapConfig;
use super::output::VelocityMap;
use crate::signal_processing::ulm::tracking::BubbleTrack;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array2;

/// Reconstructs a 2D velocity field from accumulated bubble tracks.
#[derive(Debug)]
pub struct VelocityMapper {
    config: VelocityMapConfig,
    nx: usize,
    nz: usize,
    /// Accumulated sum of v_x estimates per cell.
    sum_vx: Array2<f64>,
    /// Accumulated sum of v_z estimates per cell.
    sum_vz: Array2<f64>,
    /// Vote count per cell.
    pub(super) count: Array2<u32>,
}

impl VelocityMapper {
    /// Create a new velocity mapper.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::InvalidInput` if `pixel_size ≤ 0`, extents ≤ 0,
    /// or `frame_dt ≤ 0`.
    pub fn new(config: VelocityMapConfig) -> KwaversResult<Self> {
        if config.pixel_size <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "pixel_size must be > 0".to_owned(),
            ));
        }
        if config.x_extent <= 0.0 || config.z_extent <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "image extents must be > 0".to_owned(),
            ));
        }
        if config.frame_dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "frame_dt must be > 0".to_owned(),
            ));
        }
        let nx = (config.x_extent / config.pixel_size).ceil() as usize;
        let nz = (config.z_extent / config.pixel_size).ceil() as usize;
        Ok(Self {
            config,
            nx,
            nz,
            sum_vx: Array2::zeros((nx, nz)),
            sum_vz: Array2::zeros((nx, nz)),
            count: Array2::zeros((nx, nz)),
        })
    }

    /// Accumulate velocity estimates from `tracks`.
    ///
    /// For each consecutive detection pair (k, k+1) in each track, compute the
    /// instantaneous velocity and assign it to the segment midpoint cell.
    pub fn accumulate(&mut self, tracks: &[BubbleTrack]) {
        for track in tracks {
            let dets = &track.detections;
            let n = dets.len();
            if n < 2 {
                continue;
            }
            for k in 0..n - 1 {
                let d0 = &dets[k];
                let d1 = &dets[k + 1];
                // Time elapsed between frames (in seconds).
                let dt = (d1.frame as f64 - d0.frame as f64).max(1.0) * self.config.frame_dt;
                let vx = (d1.x - d0.x) / dt;
                let vz = (d1.z - d0.z) / dt;
                // Assign to midpoint cell.
                let mx = (d0.x + d1.x) * 0.5;
                let mz = (d0.z + d1.z) * 0.5;
                if let Some((ix, iz)) = self.to_grid_index(mx, mz) {
                    self.sum_vx[[ix, iz]] += vx;
                    self.sum_vz[[ix, iz]] += vz;
                    self.count[[ix, iz]] += 1;
                }
            }
        }
    }

    /// Compute and return the velocity map.
    ///
    /// Cells with `count < min_count` are set to NaN in all output fields.
    #[must_use]
    pub fn compute(&self) -> VelocityMap {
        let min_count = self.config.min_count as u32;
        let nan = f64::NAN;
        let mut vx = Array2::<f64>::from_elem((self.nx, self.nz), nan);
        let mut vz = Array2::<f64>::from_elem((self.nx, self.nz), nan);
        let mut speed = Array2::<f64>::from_elem((self.nx, self.nz), nan);
        let mut direction = Array2::<f64>::from_elem((self.nx, self.nz), nan);

        for ix in 0..self.nx {
            for iz in 0..self.nz {
                let c = self.count[[ix, iz]];
                if c >= min_count {
                    let cf = c as f64;
                    let mvx = self.sum_vx[[ix, iz]] / cf;
                    let mvz = self.sum_vz[[ix, iz]] / cf;
                    vx[[ix, iz]] = mvx;
                    vz[[ix, iz]] = mvz;
                    speed[[ix, iz]] = mvx.hypot(mvz);
                    direction[[ix, iz]] = mvz.atan2(mvx);
                }
            }
        }

        let wall_shear_stress = self.estimate_wall_shear_stress(&speed);

        VelocityMap {
            vx,
            vz,
            speed,
            direction,
            wall_shear_stress,
            count: self.count.clone(),
        }
    }

    /// Estimate wall shear stress proxy τ = μ · ‖∇speed‖ via central differences.
    ///
    /// ## Algorithm
    ///
    /// For interior cell (ix, iz) with all four cardinal neighbors populated:
    /// ```text
    /// ∂speed/∂x ≈ (speed[ix+1,iz] − speed[ix−1,iz]) / (2d)
    /// ∂speed/∂z ≈ (speed[ix,iz+1] − speed[ix,iz−1]) / (2d)
    /// τ[ix,iz] = μ · √((∂speed/∂x)² + (∂speed/∂z)²)
    /// ```
    fn estimate_wall_shear_stress(&self, speed: &Array2<f64>) -> Array2<f64> {
        let mu = self.config.viscosity;
        let d = self.config.pixel_size;
        let mut wss = Array2::<f64>::from_elem((self.nx, self.nz), f64::NAN);

        for ix in 1..self.nx.saturating_sub(1) {
            for iz in 1..self.nz.saturating_sub(1) {
                let sx_plus = speed[[ix + 1, iz]];
                let sx_minus = speed[[ix - 1, iz]];
                let sz_plus = speed[[ix, iz + 1]];
                let sz_minus = speed[[ix, iz - 1]];
                if sx_plus.is_nan() || sx_minus.is_nan() || sz_plus.is_nan() || sz_minus.is_nan() {
                    continue;
                }
                let gx = (sx_plus - sx_minus) / (2.0 * d);
                let gz = (sz_plus - sz_minus) / (2.0 * d);
                wss[[ix, iz]] = mu * gx.hypot(gz);
            }
        }
        wss
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
