//! Velocity Mapping from ULM Microbubble Tracks
//!
//! Reconstructs 2D velocity vector fields from accumulated bubble trajectories,
//! enabling quantitative hemodynamic measurements including flow speed, direction,
//! and wall shear stress estimation.
//!
//! ## Algorithm
//!
//! ### Instantaneous Velocity Estimation (Heiles et al. 2022)
//!
//! For each bubble track with detections at positions (x_k, z_k) at frame indices f_k:
//! ```text
//! v_x[k] = (x_{k+1} − x_k) / ((f_{k+1} − f_k) · Δt)   [m/s]
//! v_z[k] = (z_{k+1} − z_k) / ((f_{k+1} − f_k) · Δt)
//! ```
//! The estimate is assigned to the segment midpoint (x̄_k, z̄_k) = ((x_k + x_{k+1})/2, ...).
//!
//! ### Grid Accumulation (bin-and-average)
//!
//! Each velocity estimate votes into the nearest grid cell:
//! ```text
//! V_x[m,n] += v_x[k]   (for all k whose midpoint falls in cell (m,n))
//! count[m,n] += 1
//! ⟨v_x⟩[m,n] = V_x[m,n] / count[m,n]
//! ```
//! Cells with `count < min_count` are set to NaN (insufficient statistics).
//!
//! ### Velocity Magnitude and Direction
//!
//! ```text
//! speed[m,n]     = √(⟨v_x⟩² + ⟨v_z⟩²)    [m/s]
//! direction[m,n] = atan2(⟨v_z⟩, ⟨v_x⟩)   ∈ (−π, π]  [rad]
//! ```
//!
//! ### Wall Shear Stress Estimation (Womersley 1955; Reneman et al. 2006)
//!
//! Wall shear stress τ_w = μ · ∂u/∂n (velocity gradient perpendicular to wall).
//! In the discrete approximation without explicit wall segmentation, the
//! central-difference gradient magnitude of the speed field is used:
//! ```text
//! τ_proxy[m,n] = μ · ‖∇speed[m,n]‖
//!             = μ · √(((speed[m+1,n] − speed[m−1,n])/(2d))²
//!                   + ((speed[m,n+1] − speed[m,n−1])/(2d))²)
//! ```
//! where μ is dynamic blood viscosity [Pa·s] and d is the pixel size [m].
//! Only interior cells with all four neighbors populated (non-NaN) are computed.
//!
//! ## References
//!
//! - Heiles, B., et al. (2022). Performance benchmarking of microbubble-localization
//!   algorithms for ultrasound localization microscopy.
//!   *Nat. Biomed. Eng.* 6(5):605–616. DOI: 10.1038/s41551-021-00824-8
//! - Reneman, R. S., Arts, T., & Hoeks, A. P. G. (2006). Wall shear stress — an important
//!   determinant of endothelial cell function and structure in the arterial system in vivo.
//!   *J. Vasc. Res.* 43(3):251–269. DOI: 10.1159/000091648
//! - Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow and
//!   viscous drag in arteries when the pressure gradient is known.
//!   *J. Physiol.* 127(3):553–563. DOI: 10.1113/jphysiol.1955.sp005276

use crate::clinical::imaging::functional_ultrasound::ulm::tracking::BubbleTrack;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for velocity mapping.
#[derive(Debug, Clone)]
pub struct VelocityMapConfig {
    /// Physical extent of the lateral (x) dimension [m].
    pub x_extent: f64,
    /// Physical extent of the axial (z) dimension [m].
    pub z_extent: f64,
    /// Grid pixel size [m]. Default: 10 μm.
    pub pixel_size: f64,
    /// Frame duration Δt [s] (= 1 / frame_rate). Default: 1e-3 s (1 kHz).
    pub frame_dt: f64,
    /// Dynamic blood viscosity μ [Pa·s] for wall shear stress estimation.
    /// Default: 3e-3 Pa·s (whole blood at 37 °C, Merrill et al. 1965).
    pub viscosity: f64,
    /// Minimum number of velocity estimates per cell required to produce a valid
    /// output. Cells with fewer estimates are set to NaN.
    pub min_count: usize,
}

impl Default for VelocityMapConfig {
    fn default() -> Self {
        Self {
            x_extent: 0.01,    // 10 mm
            z_extent: 0.012,   // 12 mm
            pixel_size: 10e-6, // 10 μm
            frame_dt: 1e-3,    // 1 kHz acquisition
            viscosity: 3e-3,   // 3 mPa·s (whole blood)
            min_count: 3,
        }
    }
}

// ─── Output ───────────────────────────────────────────────────────────────────

/// Output of the velocity mapping algorithm.
#[derive(Debug)]
pub struct VelocityMap {
    /// Mean lateral velocity component ⟨v_x⟩ [m/s].  NaN where count < min_count.
    pub vx: Array2<f64>,
    /// Mean axial velocity component ⟨v_z⟩ [m/s].  NaN where count < min_count.
    pub vz: Array2<f64>,
    /// Velocity magnitude (speed) [m/s].  NaN where count < min_count.
    pub speed: Array2<f64>,
    /// Flow direction [rad] ∈ (−π, π].  NaN where count < min_count.
    pub direction: Array2<f64>,
    /// Wall shear stress proxy μ · ‖∇speed‖ [Pa].
    /// NaN at boundaries and cells with NaN neighbors.
    pub wall_shear_stress: Array2<f64>,
    /// Number of velocity estimates accumulated per cell.
    pub count: Array2<u32>,
}

// ─── Mapper ───────────────────────────────────────────────────────────────────

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
    count: Array2<u32>,
}

impl VelocityMapper {
    /// Create a new velocity mapper.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] if `pixel_size ≤ 0`, extents ≤ 0,
    /// or `frame_dt ≤ 0`.
    pub fn new(config: VelocityMapConfig) -> KwaversResult<Self> {
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
        if config.frame_dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "frame_dt must be > 0".to_string(),
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
                    speed[[ix, iz]] = (mvx * mvx + mvz * mvz).sqrt();
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
                wss[[ix, iz]] = mu * (gx * gx + gz * gz).sqrt();
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clinical::imaging::functional_ultrasound::ulm::microbubble_detection::BubbleDetection;
    use crate::clinical::imaging::functional_ultrasound::ulm::tracking::BubbleTrack;

    fn make_track_at(positions: &[(f64, f64)]) -> BubbleTrack {
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
    fn test_velocity_known_lateral_motion() {
        // Bubble moving at Δx = 1 mm per frame, frame_dt = 1e-3 s → vx = 1 m/s.
        let config = VelocityMapConfig {
            x_extent: 5e-3,
            z_extent: 5e-3,
            pixel_size: 50e-6,
            frame_dt: 1e-3,
            min_count: 1,
            ..Default::default()
        };
        let mut mapper = VelocityMapper::new(config).unwrap();
        let track = make_track_at(&[(1e-3, 2e-3), (2e-3, 2e-3), (3e-3, 2e-3)]);
        mapper.accumulate(&[track]);
        let map = mapper.compute();

        // Midpoint of first segment: (1.5 mm, 2 mm)
        let ix = (1.5e-3_f64 / 50e-6) as usize;
        let iz = (2e-3_f64 / 50e-6) as usize;
        let vx = map.vx[[ix, iz]];
        assert!((vx - 1.0).abs() < 1e-9, "Expected vx=1.0 m/s, got {vx:.6}");
        assert!(map.vz[[ix, iz]].abs() < 1e-9, "vz should be 0");
        let s = map.speed[[ix, iz]];
        assert!((s - 1.0).abs() < 1e-9, "speed should be 1 m/s, got {s}");
    }

    #[test]
    fn test_velocity_known_axial_motion() {
        // Bubble moving at Δz = 1 mm per frame → vz = 1 m/s.
        let config = VelocityMapConfig {
            x_extent: 5e-3,
            z_extent: 5e-3,
            pixel_size: 50e-6,
            frame_dt: 1e-3,
            min_count: 1,
            ..Default::default()
        };
        let mut mapper = VelocityMapper::new(config).unwrap();
        let track = make_track_at(&[(2e-3, 1e-3), (2e-3, 2e-3)]);
        mapper.accumulate(&[track]);
        let map = mapper.compute();

        let ix = (2e-3_f64 / 50e-6) as usize;
        let iz = (1.5e-3_f64 / 50e-6) as usize;
        let vz = map.vz[[ix, iz]];
        assert!((vz - 1.0).abs() < 1e-9, "Expected vz=1.0 m/s, got {vz}");
        assert!(map.vx[[ix, iz]].abs() < 1e-9, "vx should be 0");
    }

    #[test]
    fn test_velocity_direction_diagonal() {
        // Bubble moving equal parts x and z → direction = atan2(1, 1) = π/4.
        let config = VelocityMapConfig {
            x_extent: 5e-3,
            z_extent: 5e-3,
            pixel_size: 50e-6,
            frame_dt: 1e-3,
            min_count: 1,
            ..Default::default()
        };
        let mut mapper = VelocityMapper::new(config).unwrap();
        let track = make_track_at(&[(1e-3, 1e-3), (2e-3, 2e-3)]);
        mapper.accumulate(&[track]);
        let map = mapper.compute();

        let ix = (1.5e-3_f64 / 50e-6) as usize;
        let iz = (1.5e-3_f64 / 50e-6) as usize;
        let dir = map.direction[[ix, iz]];
        assert!(
            (dir - std::f64::consts::FRAC_PI_4).abs() < 1e-9,
            "Expected π/4 rad, got {dir}"
        );
    }

    #[test]
    fn test_min_count_mask() {
        // Cell with only 1 vote and min_count=3 must be NaN.
        let config = VelocityMapConfig {
            x_extent: 1e-3,
            z_extent: 1e-3,
            pixel_size: 50e-6,
            frame_dt: 1e-3,
            min_count: 3,
            ..Default::default()
        };
        let mut mapper = VelocityMapper::new(config).unwrap();
        let track = make_track_at(&[(100e-6, 200e-6), (150e-6, 200e-6)]);
        mapper.accumulate(&[track]);
        let map = mapper.compute();

        let ix = (125e-6_f64 / 50e-6) as usize;
        let iz = (200e-6_f64 / 50e-6) as usize;
        assert!(
            map.speed[[ix, iz]].is_nan(),
            "Cell with count < min_count must be NaN"
        );
    }

    #[test]
    fn test_count_accumulation() {
        // 5 tracks all passing through the same cell — count should be 5.
        // Δx = 100 μm, frame_dt = 100 μs → vx = 100e-6 / 100e-6 = 1 m/s.
        let config = VelocityMapConfig {
            x_extent: 1e-3,
            z_extent: 1e-3,
            pixel_size: 100e-6,
            frame_dt: 100e-6,
            min_count: 1,
            ..Default::default()
        };
        let mut mapper = VelocityMapper::new(config).unwrap();
        for _ in 0..5 {
            let track = make_track_at(&[(200e-6, 400e-6), (300e-6, 400e-6)]);
            mapper.accumulate(&[track]);
        }
        let map = mapper.compute();
        let ix = (250e-6_f64 / 100e-6) as usize;
        let iz = (400e-6_f64 / 100e-6) as usize;
        assert_eq!(map.count[[ix, iz]], 5, "Expected 5 votes");
        // Mean velocity should be 1 m/s (Δx=100 μm, dt=100 μs)
        let vx = map.vx[[ix, iz]];
        assert!((vx - 1.0).abs() < 1e-9, "Mean vx should be 1 m/s, got {vx}");
    }

    #[test]
    fn test_wall_shear_stress_uniform_field_is_zero() {
        // Uniform velocity everywhere → ∇speed = 0 → WSS = 0 in interior cells.
        let config = VelocityMapConfig {
            x_extent: 1e-3,
            z_extent: 1e-3,
            pixel_size: 50e-6,
            frame_dt: 1e-3,
            min_count: 1,
            ..Default::default()
        };
        let mut mapper = VelocityMapper::new(config).unwrap();
        // Dense uniform-motion tracks covering a 5×5 block.
        let mut tracks = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                let x0 = i as f64 * 50e-6 + 100e-6;
                let z0 = j as f64 * 50e-6 + 100e-6;
                // 4 frames of uniform x-motion at 50 μm/frame
                tracks.push(make_track_at(&[
                    (x0, z0),
                    (x0 + 50e-6, z0),
                    (x0 + 100e-6, z0),
                    (x0 + 150e-6, z0),
                ]));
            }
        }
        mapper.accumulate(&tracks);
        let map = mapper.compute();

        // Check a fully interior cell of the block.
        let ix = (275e-6_f64 / 50e-6) as usize;
        let iz = (275e-6_f64 / 50e-6) as usize;
        let wss = map.wall_shear_stress[[ix, iz]];
        if !wss.is_nan() {
            assert!(
                wss.abs() < 1e-6,
                "WSS for uniform flow must be ≈0, got {wss}"
            );
        }
    }

    #[test]
    fn test_zero_length_track_skipped() {
        // A track with only one detection has no velocity segments — must not panic.
        let config = VelocityMapConfig {
            x_extent: 1e-3,
            z_extent: 1e-3,
            pixel_size: 50e-6,
            frame_dt: 1e-3,
            min_count: 1,
            ..Default::default()
        };
        let mut mapper = VelocityMapper::new(config).unwrap();
        let track = make_track_at(&[(500e-6, 500e-6)]);
        mapper.accumulate(&[track]); // Should not panic
        let total: u32 = mapper.count.iter().sum();
        assert_eq!(
            total, 0,
            "Single-detection track contributes no velocity segments"
        );
    }
}
