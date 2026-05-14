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
use crate::clinical::imaging::reconstruction::acoustic_projection::{
    backproject_acoustic, project_acoustic,
};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array3};
use std::time::Instant;

/// Streaming SIRT reconstruction pipeline.
#[derive(Debug)]
pub struct RealTimeSirtPipeline {
    config: RealTimeSirtConfig,
    current_image: Option<Array3<f64>>,
    frame_count: usize,
    start_time: Instant,
    frame_history: Vec<ReconstructionFrame>,
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
        }
    }

    /// Process one RF measurement frame and return a reconstructed image.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
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
        let (nx, ny, nz) = image.dim();
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
            let n_sensors = geom.element_x.len();
            let alpha = geom.alpha_nepers_per_m_per_hz();
            let f_c = geom.center_frequency_hz;
            let (dx, dy, dz) = geom.voxel_spacing;
            let zs_elem = geom.element_z;

            let row_norm_sq: Vec<f64> = geom
                .element_x
                .iter()
                .map(|&xs| {
                    let mut sq = 0.0_f64;
                    for i in 0..nx {
                        let xv = i as f64 * dx;
                        let dx2 = (xv - xs) * (xv - xs);
                        for j in 0..ny {
                            let yv = j as f64 * dy;
                            let dxy2 = yv.mul_add(yv, dx2);
                            for k in 0..nz {
                                let zv = k as f64 * dz;
                                let r = (zv - zs_elem).mul_add(zv - zs_elem, dxy2).sqrt().max(1e-6);
                                let w = (-2.0 * alpha * f_c * r).exp() / r;
                                sq += w * w;
                            }
                        }
                    }
                    sq.max(1e-30)
                })
                .collect();

            for _ in 0..self.config.sirt_config.max_iterations {
                let projection = project_acoustic(&image, geom);
                let mut residual = Array1::<f64>::zeros(n_sensors);
                for idx in 0..n_sensors.min(n_meas) {
                    residual[idx] = preprocessed[idx] - projection[idx];
                }
                let res_norm: f64 = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();
                convergence_error = res_norm / meas_norm;
                let scaled = Array1::from_shape_fn(n_sensors, |s| residual[s] / row_norm_sq[s]);
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
        for &val in rf_data {
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_smoothing(image: &Array3<f64>, sigma: f64) -> KwaversResult<Array3<f64>> {
        if sigma <= 0.0 {
            return Ok(image.clone());
        }
        let (nx, ny, nz) = image.dim();
        if nx < 3 || ny < 3 || nz < 3 {
            return Ok(image.clone());
        }
        let wn = (-0.5 / (sigma * sigma)).exp();
        let norm = 2.0f64.mul_add(wn, 1.0);
        let w0 = 1.0 / norm;
        let wn = wn / norm;

        let mut s = image.clone();
        let mut tmp = s.clone();
        for i in 1..nx - 1 {
            for j in 0..ny {
                for k in 0..nz {
                    tmp[[i, j, k]] =
                        wn * s[[i - 1, j, k]] + w0 * s[[i, j, k]] + wn * s[[i + 1, j, k]];
                }
            }
        }
        s = tmp;
        let mut tmp = s.clone();
        for i in 0..nx {
            for j in 1..ny - 1 {
                for k in 0..nz {
                    tmp[[i, j, k]] =
                        wn * s[[i, j - 1, k]] + w0 * s[[i, j, k]] + wn * s[[i, j + 1, k]];
                }
            }
        }
        s = tmp;
        let mut tmp = s.clone();
        for i in 0..nx {
            for j in 0..ny {
                for k in 1..nz - 1 {
                    tmp[[i, j, k]] =
                        wn * s[[i, j, k - 1]] + w0 * s[[i, j, k]] + wn * s[[i, j, k + 1]];
                }
            }
        }
        Ok(tmp)
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
