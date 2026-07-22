//! `RealTimeSirtPipeline` — streaming SIRT reconstruction logic.
//!
//! SRP: changes when the SIRT update equation, projection model dispatch,
//! postprocessing order, or quality assessment formula changes.
//!
//! ## SIRT Update (simultaneous-rows form)
//!
//! ```text
//! x^(k+1) = x^(k) + λ · D_R · Aᵀ · (b − A·x^(k))
//! ```
//!
//! Two forward-projection models are selected by `RealTimeSirtConfig::transducer_geometry`:
//! - `None`  — legacy column-sum: `A[s,(i,j,k)] = 1` iff `s = i·ny + j`
//! - `Some`  — acoustic ray-tracing: `A[s,v] = exp(−2αf_c r) / r`
//!   Row normalisation `D_R(s) = 1/‖A_row_s‖²` (Dines & Kak 1979, §III).

use super::config::RealTimeSirtConfig;
use super::types::{FrameQuality, ReconstructionFrame};
use crate::reconstruction::acoustic_projection::{
    backproject_acoustic, project_acoustic, AcousticProjectionGeometry,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, Array3};
use moirai_parallel::{for_each_chunk_mut_enumerated_with, map_collect_index_with, Adaptive};
use std::time::Instant;

/// Compute the per-sensor squared row norm `‖A_row_s‖²` for the acoustic projection matrix.
///
/// ## Theorem: Row-norm preconditioning in SIRT (Dines & Kak 1979 §III)
///
/// The SIRT update
/// ```text
/// x^(k+1) = x^(k) + λ · D_R · Aᵀ · (b − A·x^(k))
/// ```
/// requires the diagonal preconditioner `D_R`s` = 1/‖A_row_s‖²` so that each
/// sensor is scaled by its own energy; sensors with large geometric coverage do
/// not dominate the gradient update.
///
/// For the acoustic projection model `A[s,(i,j,k)] = exp(−2αf_c r) / r`:
/// ```text
/// ‖A_row_s‖² = Σ_{i,j,k} (exp(−2αf_c r_{s,v}) / r_{s,v})²
/// ```
///
/// This is computed once per geometry/grid-shape pair and cached in
/// `RealTimeSirtPipeline::row_norm_sq_cache`. Parallelised over sensors via Moirai.
/// The lower clamp at `f64::EPSILON` prevents a zero-denominator singularity
/// for far-field sensors whose stencil weight is below double-precision roundoff.
///
/// # References
/// - Dines KA, Kak AC (1979). "Ultrasonic attenuation tomography of soft tissues."
///   *Ultrasonic Imaging* 1(1):16–33, §III.
fn compute_row_norm_sq(
    geom: &AcousticProjectionGeometry,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let (dx, dy, dz) = geom.voxel_spacing;
    let alpha = geom.alpha_nepers_per_m_per_hz();
    let f_c = geom.center_frequency_hz;
    let zs = geom.element_z;

    map_collect_index_with::<Adaptive, _, _>(geom.element_x.len(), |sensor_idx| {
        let xs = geom.element_x[sensor_idx];
        let mut norm_sq = 0.0_f64;
        for i in 0..nx {
            let xv = i as f64 * dx;
            let dx2 = (xv - xs) * (xv - xs);
            for j in 0..ny {
                let yv = j as f64 * dy;
                let dxy2 = yv.mul_add(yv, dx2);
                for k in 0..nz {
                    let zv = k as f64 * dz;
                    let r = (zv - zs).mul_add(zv - zs, dxy2).sqrt().max(1e-6);
                    let weight = (-2.0 * alpha * f_c * r).exp() / r;
                    norm_sq += weight * weight;
                }
            }
        }
        // Clamp away from zero to prevent div-by-zero in D_R[s] = 1/‖A_row_s‖²
        norm_sq.max(f64::EPSILON)
    })
}

fn smooth_x_pass(
    a: &Array3<f64>,
    b: &mut Array3<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    wn: f64,
    w0: f64,
) {
    let plane_len = ny * nz;
    let a_slice = a
        .as_slice_memory_order()
        .expect("invariant: Array3 smoothing read buffer is contiguous");
    let b_slice = b
        .as_slice_memory_order_mut()
        .expect("invariant: Array3 smoothing write buffer is contiguous");
    let interior = &mut b_slice[plane_len..(nx - 1) * plane_len];

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
        interior,
        plane_len,
        |chunk_idx, plane| {
            let i = chunk_idx + 1;
            let center_base = i * plane_len;
            let left_base = center_base - plane_len;
            let right_base = center_base + plane_len;

            for (offset, dst) in plane.iter_mut().enumerate() {
                *dst = wn * a_slice[left_base + offset]
                    + w0 * a_slice[center_base + offset]
                    + wn * a_slice[right_base + offset];
            }
        },
    );
}

fn smooth_y_pass(a: &Array3<f64>, b: &mut Array3<f64>, ny: usize, nz: usize, wn: f64, w0: f64) {
    let plane_len = ny * nz;
    let a_slice = a
        .as_slice_memory_order()
        .expect("invariant: Array3 smoothing read buffer is contiguous");
    let b_slice = b
        .as_slice_memory_order_mut()
        .expect("invariant: Array3 smoothing write buffer is contiguous");

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(b_slice, plane_len, |i, plane| {
        let plane_base = i * plane_len;
        for j in 1..ny - 1 {
            let row_base = j * nz;
            let left_base = plane_base + (j - 1) * nz;
            let center_base = plane_base + row_base;
            let right_base = plane_base + (j + 1) * nz;

            for k in 0..nz {
                plane[row_base + k] = wn * a_slice[left_base + k]
                    + w0 * a_slice[center_base + k]
                    + wn * a_slice[right_base + k];
            }
        }
    });
}

fn smooth_z_pass(a: &Array3<f64>, b: &mut Array3<f64>, nz: usize, wn: f64, w0: f64) {
    let a_slice = a
        .as_slice_memory_order()
        .expect("invariant: Array3 smoothing read buffer is contiguous");
    let b_slice = b
        .as_slice_memory_order_mut()
        .expect("invariant: Array3 smoothing write buffer is contiguous");

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(b_slice, nz, |row_idx, row| {
        let row_base = row_idx * nz;
        for k in 1..nz - 1 {
            row[k] = wn * a_slice[row_base + k - 1]
                + w0 * a_slice[row_base + k]
                + wn * a_slice[row_base + k + 1];
        }
    });
}

/// Streaming SIRT reconstruction pipeline.
///
/// ## Row-norm cache
///
/// `row_norm_sq_cache` stores the per-sensor squared row norms of the acoustic
/// projection operator together with the grid shape for which they were computed.
/// Computing these norms requires O(n_sensors × NX×NY×NZ) work — equivalent to
/// one full forward projection.  Recomputing them on every frame is wasteful
/// because the geometry and grid shape are fixed for the pipeline lifetime.
/// The cache is invalidated only when `expected_grid_size` changes (uncommon).
#[derive(Debug)]
pub struct RealTimeSirtPipeline {
    config: RealTimeSirtConfig,
    current_image: Option<Array3<f64>>,
    frame_count: usize,
    start_time: Instant,
    frame_history: Vec<ReconstructionFrame>,
    /// Cached `(norms, grid_shape)`; recomputed only when `grid_shape` changes.
    row_norm_sq_cache: Option<(Vec<f64>, (usize, usize, usize))>,
}

impl RealTimeSirtPipeline {
    /// Create a new streaming SIRT pipeline.
    #[must_use]
    pub fn new(config: RealTimeSirtConfig) -> Self {
        Self {
            config,
            current_image: None,
            frame_count: 0,
            start_time: Instant::now(),
            frame_history: Vec::new(),
            row_norm_sq_cache: None,
        }
    }

    /// Process one RF measurement frame and return a reconstructed image.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn process_frame(
        &mut self,
        rf_data: &Array1<f64>,
        expected_grid_size: (usize, usize, usize),
    ) -> KwaversResult<ReconstructionFrame> {
        let frame_start = Instant::now();
        if self.config.enable_safety_checks {
            Self::validate_input(rf_data)?;
        }
        let preprocessed = if self.config.enable_preprocessing {
            Self::preprocess(rf_data)?
        } else {
            rf_data.clone()
        };
        if self.current_image.is_none() {
            self.current_image = Some(Array3::zeros(expected_grid_size));
        }

        let mut image = self.current_image.clone().unwrap();
        let [nx, ny, nz] = image.shape();
        let relaxation = self.config.sirt_config.relaxation_factor;
        let n_meas = preprocessed.len();
        let meas_norm = {
            let ss: f64 = preprocessed.iter().map(|&v| v * v).sum();
            ss.sqrt().max(1e-30)
        };
        let mut convergence_error = f64::INFINITY;

        if let Some(ref geom) = self.config.transducer_geometry.clone() {
            // ── Acoustic physics-based SIRT ───────────────────────────────────
            // Row normalisation D_R[s] = 1/‖A_row_s‖² (Dines & Kak 1979 §III).
            //
            // Row norms depend only on the geometry and grid shape, not on the
            // RF data.  Cache them: first call computes in parallel (Moirai over
            // elements); subsequent calls reuse the cached vector.  Invalidate
            // when grid shape changes (geometry is fixed for the pipeline lifetime).
            let grid_shape = (nx, ny, nz);
            if self
                .row_norm_sq_cache
                .as_ref()
                .is_none_or(|(_, s)| *s != grid_shape)
            {
                let norms = compute_row_norm_sq(geom, nx, ny, nz);
                self.row_norm_sq_cache = Some((norms, grid_shape));
            }
            let row_norm_sq = &self.row_norm_sq_cache.as_ref().unwrap().0;
            let n_sensors = geom.element_x.len();

            for _ in 0..self.config.sirt_config.max_iterations {
                let projection = project_acoustic(&image, geom);
                let mut residual = Array1::<f64>::zeros(n_sensors);
                for idx in 0..n_sensors.min(n_meas) {
                    residual[idx] = preprocessed[idx] - projection[idx];
                }
                let res_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
                convergence_error = res_norm / meas_norm;
                let scaled = Array1::from_shape_fn(n_sensors, |[s]| residual[s] / row_norm_sq[s]);
                let backproj = backproject_acoustic(&scaled, (nx, ny, nz), geom);
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            image[[i, j, k]] += relaxation * backproj[[i, j, k]];
                        }
                    }
                }
            }
        } else {
            // ── Legacy column-sum SIRT ────────────────────────────────────────
            // A[s,(i,j,k)] = 1 iff s = i·ny+j; D_R = (1/nz)·I.
            let proj_len = nx * ny;
            let inv_nz = 1.0 / nz.max(1) as f64;
            let mut projection = Array1::<f64>::zeros(proj_len);
            let mut residual = Array1::<f64>::zeros(proj_len);

            for _ in 0..self.config.sirt_config.max_iterations {
                projection.fill(0.0);
                for i in 0..nx {
                    for j in 0..ny {
                        let mut sum = 0.0_f64;
                        for k in 0..nz {
                            sum += image[[i, j, k]];
                        }
                        projection[i * ny + j] = sum;
                    }
                }
                residual.fill(0.0);
                for idx in 0..proj_len.min(n_meas) {
                    residual[idx] = preprocessed[idx] - projection[idx];
                }
                let res_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
                convergence_error = res_norm / meas_norm;
                for i in 0..nx {
                    for j in 0..ny {
                        let r = residual[i * ny + j] * inv_nz;
                        for k in 0..nz {
                            image[[i, j, k]] += relaxation * r;
                        }
                    }
                }
            }
        }

        self.current_image = Some(image.clone());

        let output = match self.config.output_smoothing_sigma {
            Some(sigma) => Self::apply_smoothing(&image, sigma)?,
            None => image,
        };
        let output = match self.config.intensity_threshold {
            Some(threshold) => output.mapv(|x| if x > threshold { x } else { 0.0 }),
            None => output,
        };
        let quality = if self.config.enable_quality_monitoring {
            Some(Self::assess_quality(&output))
        } else {
            None
        };

        let frame = ReconstructionFrame {
            timestamp: self.start_time.elapsed().as_secs_f64(),
            image: output,
            iterations: self.config.sirt_config.max_iterations,
            computation_time_ms: frame_start.elapsed().as_secs_f64() * 1000.0,
            convergence_error,
            quality_metrics: quality,
        };
        self.frame_history.push(frame.clone());
        self.frame_count += 1;
        Ok(frame)
    }

    fn validate_input(rf_data: &Array1<f64>) -> KwaversResult<()> {
        if rf_data.is_empty() {
            return Err(KwaversError::InvalidInput("Empty RF data".to_owned()));
        }
        for &val in rf_data.iter() {
            if !val.is_finite() {
                return Err(KwaversError::InvalidInput(
                    "RF data contains NaN or Inf".to_owned(),
                ));
            }
        }
        Ok(())
    }

    fn preprocess(rf_data: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        let max_val = rf_data.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if max_val > 0.0 {
            Ok(rf_data / max_val)
        } else {
            Ok(rf_data.clone())
        }
    }

    /// Separable 3-point Gaussian-weighted smoothing (σ in grid points).
    ///
    /// ## Implementation
    ///
    /// Three sequential 1-D passes (X → Y → Z) using a 2-buffer ping-pong
    /// pattern: two `Array3` allocations (down from four in the original), with
    /// `Array3::assign` (memcpy, no allocation) used to propagate edge-plane
    /// values between passes.  Each pass interior is computed in parallel via
    /// Moirai chunk dispatch.
    ///
    /// ## Correctness invariant
    ///
    /// Edge planes in each dimension (index 0 and dim−1) are not written by
    /// their respective pass, preserving the same boundary semantics as the
    /// original sequential implementation.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    fn apply_smoothing(image: &Array3<f64>, sigma: f64) -> KwaversResult<Array3<f64>> {
        if sigma <= 0.0 {
            return Ok(image.clone());
        }
        let [nx, ny, nz] = image.shape();
        if nx < 3 || ny < 3 || nz < 3 {
            return Ok(image.clone());
        }
        let wn = (-0.5 / (sigma * sigma)).exp();
        let norm_w = 2.0f64.mul_add(wn, 1.0);
        let w0 = 1.0 / norm_w;
        let wn = wn / norm_w;

        // Two allocations total (down from four).  After each pass we swap
        // buffers and `assign` so that edge-plane values in the write buffer
        // always reflect the most recent read buffer state before interior
        // positions are overwritten.
        let mut a = image.clone(); // read buffer
        let mut b = a.clone(); // write buffer (edge planes pre-filled)

        // --- X pass: interior i ∈ [1, nx−2] ---
        smooth_x_pass(&a, &mut b, nx, ny, nz, wn, w0);
        std::mem::swap(&mut a, &mut b);
        // Propagate current result (in a) into b so j/k edge planes are correct.
        b.assign(&a);

        // --- Y pass: interior j ∈ [1, ny−2] ---
        smooth_y_pass(&a, &mut b, ny, nz, wn, w0);
        std::mem::swap(&mut a, &mut b);
        b.assign(&a);

        // --- Z pass: interior k ∈ [1, nz−2] ---
        smooth_z_pass(&a, &mut b, nz, wn, w0);
        Ok(b)
    }

    fn assess_quality(image: &Array3<f64>) -> FrameQuality {
        let min_val = image.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = image.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean: f64 = image.iter().sum::<f64>() / image.len() as f64;
        let variance: f64 =
            image.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / image.len() as f64;
        let snr = if variance > 0.0 {
            10.0 * (mean * mean / variance).log10()
        } else {
            0.0
        };
        FrameQuality {
            snr_estimate: snr,
            artifact_level: 0.0,
            spatial_smoothness: 0.0,
            dynamic_range: max_val - min_val,
            converged: true,
        }
    }

    /// Chronological slice of all processed frames.
    #[must_use]
    pub fn frame_history(&self) -> &[ReconstructionFrame] {
        &self.frame_history
    }

    /// Average throughput since pipeline creation (fps).
    #[must_use]
    pub fn avg_frame_rate(&self) -> f64 {
        if self.frame_count == 0 {
            return 0.0;
        }
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.frame_count as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Total number of frames processed.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }
}
