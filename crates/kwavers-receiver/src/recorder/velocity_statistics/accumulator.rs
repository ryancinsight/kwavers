//! Per-component velocity statistics accumulator.
//!
//! # Statistical definitions (Treeby & Cox 2010, sensor.record API)
//!
//! For a velocity component `u_α(x, t)` sampled at discrete times `t_n`:
//!
//! ## Maximum
//! ```text
//!   u_max(x) = max_n u_α(x, t_n)
//! ```
//!
//! ## Minimum
//! ```text
//!   u_min(x) = min_n u_α(x, t_n)
//! ```
//!
//! ## Root-mean-square
//! ```text
//!   u_rms(x) = sqrt( (1/N) · Σ_n u_α(x, t_n)² )
//! ```
//!
//! All three statistics are maintained online (single-pass, constant extra memory)
//! via the incremental update:
//! ```text
//!   u_max_{n+1} = max(u_max_n, u_new)
//!   u_min_{n+1} = min(u_min_n, u_new)
//!   sq_sum_{n+1} = sq_sum_n + u_new²
//! ```
//!
//! # References
//! - Treeby & Cox (2010). k-Wave MATLAB toolbox documentation, `sensor.record`.

use kwavers_core::error::KwaversResult;
use moirai_parallel::{for_each_chunk_triple_mut_enumerated_with, Adaptive};
use ndarray::{Array1, Array3, Zip};

use super::super::STATISTICS_CHUNK_SIZE;
use super::helpers::{fill_field_at_positions, validate_sample_output_len};

/// Per-component velocity statistics accumulator.
///
/// Stores running max, min, and squared sum for one velocity component
/// (ux, uy, or uz). Standard-layout updates are provider-parallelized over the
/// spatial domain through Moirai.
#[derive(Debug, Clone)]
pub struct VelocityComponentStats {
    /// Element-wise maximum u_α over all recorded time steps.
    pub u_max: Array3<f64>,
    /// Element-wise minimum u_α over all recorded time steps.
    pub u_min: Array3<f64>,
    /// Running sum of u_α² for RMS computation.
    u_squared_sum: Array3<f64>,
    /// Number of time steps accumulated.
    pub time_step_count: usize,
    shape: (usize, usize, usize),
}

impl VelocityComponentStats {
    /// Create a new accumulator for a grid of shape `(nx, ny, nz)`.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            u_max: Array3::from_elem((nx, ny, nz), f64::NEG_INFINITY),
            u_min: Array3::from_elem((nx, ny, nz), f64::INFINITY),
            u_squared_sum: Array3::zeros((nx, ny, nz)),
            time_step_count: 0,
            shape: (nx, ny, nz),
        }
    }

    /// Accumulate one time step of data.
    ///
    /// Updates max, min, and squared sum element-wise. O(N^3), with Moirai
    /// chunk dispatch for standard layouts and no allocation.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[inline]
    pub fn update(&mut self, field: &Array3<f64>) {
        debug_assert_eq!(
            field.dim(),
            self.shape,
            "velocity field shape mismatch in VelocityComponentStats::update"
        );

        match (
            self.u_max.as_slice_mut(),
            self.u_min.as_slice_mut(),
            self.u_squared_sum.as_slice_mut(),
            field.as_slice(),
        ) {
            (Some(u_max), Some(u_min), Some(u_squared_sum), Some(field)) => {
                for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                    u_max,
                    u_min,
                    u_squared_sum,
                    STATISTICS_CHUNK_SIZE,
                    |chunk_index, u_max, u_min, u_squared_sum| {
                        let base = chunk_index * STATISTICS_CHUNK_SIZE;
                        for lane in 0..u_max.len() {
                            let u = field[base + lane];
                            if u > u_max[lane] {
                                u_max[lane] = u;
                            }
                            if u < u_min[lane] {
                                u_min[lane] = u;
                            }
                            u_squared_sum[lane] += u * u;
                        }
                    },
                );
            }
            _ => Zip::from(&mut self.u_max)
                .and(&mut self.u_min)
                .and(&mut self.u_squared_sum)
                .and(field)
                .for_each(|u_max, u_min, sq_sum, &u| {
                    if u > *u_max {
                        *u_max = u;
                    }
                    if u < *u_min {
                        *u_min = u;
                    }
                    *sq_sum += u * u;
                }),
        }

        self.time_step_count += 1;
    }

    /// Compute the RMS field.
    ///
    /// Returns a zero array when no steps have been accumulated.
    #[must_use]
    pub fn u_rms(&self) -> Array3<f64> {
        if self.time_step_count == 0 {
            return Array3::zeros(self.shape);
        }
        let n = self.time_step_count as f64;
        self.u_squared_sum.mapv(|sq| (sq / n).sqrt())
    }

    // ── Sensor-position sampling ─────────────────────────────────────────────

    /// Sample the per-component max at sensor positions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn sample_max(&self, positions: &[(usize, usize, usize)]) -> Array1<f64> {
        let mut out = Array1::zeros(positions.len());
        let _ = self.fill_max(positions, &mut out);
        out
    }

    /// Fill caller-owned storage with per-component max at sensor positions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_max(
        &self,
        positions: &[(usize, usize, usize)],
        out: &mut Array1<f64>,
    ) -> KwaversResult<()> {
        fill_field_at_positions(&self.u_max, positions, out)
    }

    /// Sample the per-component min at sensor positions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn sample_min(&self, positions: &[(usize, usize, usize)]) -> Array1<f64> {
        let mut out = Array1::zeros(positions.len());
        let _ = self.fill_min(positions, &mut out);
        out
    }

    /// Fill caller-owned storage with per-component min at sensor positions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_min(
        &self,
        positions: &[(usize, usize, usize)],
        out: &mut Array1<f64>,
    ) -> KwaversResult<()> {
        fill_field_at_positions(&self.u_min, positions, out)
    }

    /// Sample the per-component RMS at sensor positions.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn sample_rms(&self, positions: &[(usize, usize, usize)]) -> Array1<f64> {
        let mut out = Array1::zeros(positions.len());
        let _ = self.fill_rms(positions, &mut out);
        out
    }

    /// Fill caller-owned storage with per-component RMS at sensor positions.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn fill_rms(
        &self,
        positions: &[(usize, usize, usize)],
        out: &mut Array1<f64>,
    ) -> KwaversResult<()> {
        validate_sample_output_len(positions, out)?;
        if self.time_step_count == 0 {
            out.fill(0.0);
            return Ok(());
        }
        let n = self.time_step_count as f64;
        for (row, &(i, j, k)) in positions.iter().enumerate() {
            out[row] = (self.u_squared_sum[[i, j, k]] / n).sqrt();
        }
        Ok(())
    }

    // ── Global aggregates ────────────────────────────────────────────────────

    /// Reset all accumulators to their initial state.
    pub fn reset(&mut self) {
        self.u_max.fill(f64::NEG_INFINITY);
        self.u_min.fill(f64::INFINITY);
        self.u_squared_sum.fill(0.0);
        self.time_step_count = 0;
    }

    /// Global maximum u across the entire spatial domain.
    #[must_use]
    pub fn global_max(&self) -> f64 {
        self.u_max.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Global minimum u across the entire spatial domain.
    #[must_use]
    pub fn global_min(&self) -> f64 {
        self.u_min.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }

    /// Shape of the velocity field arrays.
    #[must_use]
    pub fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }
}
