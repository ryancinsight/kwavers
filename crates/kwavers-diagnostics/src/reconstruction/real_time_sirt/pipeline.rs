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
use super::smoothing::{apply_smoothing, compute_row_norm_sq};
use super::types::{FrameQuality, ReconstructionFrame};
use crate::reconstruction::acoustic_projection::{
    backproject_acoustic_into, project_acoustic_into,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, Array3};
use std::time::Instant;

/// Maximum number of frames retained by [`RealTimeSirtPipeline::frame_history`].
///
/// The pipeline is a streaming component: without a bound, every processed
/// frame clones a full 3-D image into `frame_history`, so a long-running scan
/// grows the heap without limit. History beyond this window is dropped; the
/// aggregate counters (`frame_count`, `avg_frame_rate`) are unaffected.
pub(crate) const FRAME_HISTORY_LIMIT: usize = 64;

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
    /// Reused projection scratch (zeroed before each use).
    projection_scratch: Option<Array1<f64>>,
    /// Reused backprojection scratch (zeroed before each use).
    backprojection_scratch: Option<Array3<f64>>,
    /// Reused residual / preconditioned-residual scratch.
    residual_scratch: Option<Array1<f64>>,
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
            projection_scratch: None,
            backprojection_scratch: None,
            residual_scratch: None,
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

        // Move the image out of `self` for the duration of the update instead
        // of cloning it: state is fully restored below, and the per-frame heap
        // cost no longer scales with the grid volume.
        let mut image = self
            .current_image
            .take()
            .expect("current_image initialised above");
        let [nx, ny, nz] = image.shape();
        let relaxation = self.config.sirt_config.relaxation_factor;
        let n_meas = preprocessed.len();
        let meas_norm = {
            let ss: f64 = preprocessed.iter().map(|&v| v * v).sum();
            ss.sqrt().max(1e-30)
        };
        let mut convergence_error = f64::INFINITY;

        // Lend the geometry for the duration of the update (restored below),
        // avoiding a full geometry clone — sensor arrays included — per frame.
        let geometry = self.config.transducer_geometry.take();
        if let Some(geom) = geometry.as_ref() {
            // ── Acoustic physics-based SIRT ───────────────────────────────────
            // Row normalisation D_R[s] = 1/‖A_row_s‖² (Dines & Kak 1979 §III).
            //
            // Row norms depend only on the geometry and grid shape, not on the
            // RF data.  Cache them: first call computes in parallel (Moirai over
            // elements); subsequent calls reuse the cached vector.  Invalidate
            // when grid shape changes (geometry is fixed for the pipeline lifetime).
            let grid_shape = (nx, ny, nz);
            // Take ownership of the row-norm cache for the duration of the
            // update so the iteration scratch buffers can be borrowed from
            // `self` concurrently; it is restored after the update below.
            let (row_norm_sq, cached_shape) = match self.row_norm_sq_cache.take() {
                Some((norms, shape)) if shape == grid_shape => (norms, shape),
                _ => {
                    let norms = compute_row_norm_sq(geom, nx, ny, nz);
                    (norms, grid_shape)
                }
            };
            let n_sensors = geom.element_x.len();

            // Provision (once) and reuse the iteration scratch buffers across
            // iterations and frames; reallocation happens only when the sensor
            // count or grid shape changes.
            if self
                .projection_scratch
                .as_ref()
                .is_none_or(|p| p.len() != n_sensors)
            {
                self.projection_scratch = Some(Array1::zeros(n_sensors));
            }
            if self
                .residual_scratch
                .as_ref()
                .is_none_or(|p| p.len() != n_sensors)
            {
                self.residual_scratch = Some(Array1::zeros(n_sensors));
            }
            if self
                .backprojection_scratch
                .as_ref()
                .is_none_or(|b| b.shape() != [nx, ny, nz])
            {
                self.backprojection_scratch = Some(Array3::zeros((nx, ny, nz)));
            }
            let proj = self.projection_scratch.as_mut().expect("provisioned above");
            let residual = self.residual_scratch.as_mut().expect("provisioned above");
            let backproj = self
                .backprojection_scratch
                .as_mut()
                .expect("provisioned above");

            for _ in 0..self.config.sirt_config.max_iterations {
                proj.fill(0.0);
                project_acoustic_into(&image, geom, proj);
                // Residual r = b − A·x over the measured sensor range; sensors
                // without a measurement keep a zero residual (no update source).
                let usable = n_sensors.min(n_meas);
                let mut res_norm_sq = 0.0_f64;
                for idx in 0..usable {
                    let r = preprocessed[idx] - proj[idx];
                    residual[idx] = r;
                    res_norm_sq += r * r;
                }
                for idx in usable..n_sensors {
                    residual[idx] = 0.0;
                }
                convergence_error = res_norm_sq.sqrt() / meas_norm;
                // In-place row-norm preconditioning: scaled = D_R · r.
                for (s, r) in residual.iter_mut().enumerate() {
                    *r /= row_norm_sq[s];
                }
                backproj.fill(0.0);
                backproject_acoustic_into(residual, geom, backproj);
                let image_slice = image
                    .as_slice_mut()
                    .expect("invariant: SIRT image is C-contiguous");
                let bp_slice = backproj
                    .as_slice()
                    .expect("invariant: backprojection scratch is C-contiguous");
                for (dst, &src) in image_slice.iter_mut().zip(bp_slice.iter()) {
                    *dst += relaxation * src;
                }
            }
            // Return the row-norm cache to the pipeline for reuse by later
            // frames (same grid shape expected).
            self.row_norm_sq_cache = Some((row_norm_sq, cached_shape));
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

        // Restore the lent geometry and the working image before postprocessing.
        self.config.transducer_geometry = geometry;
        self.current_image = Some(image.clone());

        let output = match self.config.output_smoothing_sigma {
            Some(sigma) => apply_smoothing(image, sigma)?,
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
        if self.frame_history.len() == FRAME_HISTORY_LIMIT {
            self.frame_history.remove(0);
        }
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
