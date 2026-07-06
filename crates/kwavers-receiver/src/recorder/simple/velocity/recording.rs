//! Velocity recording kernel for [`SensorRecorder`].
//!
//! Implements [`record_velocity_step`](SensorRecorder::record_velocity_step),
//! the non-staggered collocated-velocity kernel, and the intensity accumulator.
//!
//! # Non-staggered velocity interpolation
//!
//! **Theorem (half-cell backward shift).**  The staggered velocity field
//! `u_stag[i]` is defined at half-integer grid positions `(i + ½)·Δx`.  The
//! pressure-grid collocated value at integer position `i·Δx` is
//!
//! ```text
//!   u_ns[i] = (u_stag[i-1] + u_stag[i]) / 2,   i ≥ 1
//!   u_ns[0] = u_stag[0] / 2                      (ghost cell = 0)
//! ```
//!
//! Evaluating this expression directly at each sensor position is algebraically
//! identical to materialising the full collocated field and sampling it, but
//! avoids a full-grid transient allocation.
//!
//! # Acoustic intensity
//!
//! **Theorem (instantaneous intensity).**  The acoustic intensity vector at
//! sensor position `s` and time step `n` is
//!
//! ```text
//!   I(s, n) = p(s, n) · u(s, n)
//! ```
//!
//! where `p(s, n)` is already stored in `pressure[[row, col]]` by the preceding
//! `record_step` call, and `u(s, n)` is the staggered velocity component.

use super::super::SensorRecorder;
use crate::recorder::fields::SensorRecordField;
use kwavers_core::error::KwaversResult;
use ndarray::{Array1, Array2, Array3};

impl SensorRecorder {
    /// Record one time step of velocity data.
    ///
    /// Samples staggered velocity at sensor positions for time-series recording
    /// and updates per-component statistics accumulators. Must be called
    /// immediately after [`record_step`](SensorRecorder::record_step) so that
    /// the shared `next_step` counter is already incremented.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn record_velocity_step(
        &mut self,
        ux: &Array3<f64>,
        uy: &Array3<f64>,
        uz: &Array3<f64>,
    ) -> KwaversResult<()> {
        // Full-grid statistics (O(N^3), provider-parallelized inside accumulator).
        if let Some(ref mut s) = self.ux_stats {
            s.update(ux);
        }
        if let Some(ref mut s) = self.uy_stats {
            s.update(uy);
        }
        if let Some(ref mut s) = self.uz_stats {
            s.update(uz);
        }

        if self.sensor_indices.is_empty() || self.next_step == 0 {
            return Ok(());
        }
        // `next_step` was already incremented by `record_step`.
        let col = self.next_step - 1;
        if col >= self.expected_steps {
            return Ok(());
        }

        // Staggered velocity time series.
        if let Some(ref mut buf) = self.ux_data {
            for (row, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
                buf[[row, col]] = ux[[i, j, k]];
            }
        }
        if let Some(ref mut buf) = self.uy_data {
            for (row, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
                buf[[row, col]] = uy[[i, j, k]];
            }
        }
        if let Some(ref mut buf) = self.uz_data {
            for (row, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
                buf[[row, col]] = uz[[i, j, k]];
            }
        }

        // Non-staggered velocity overrides the staggered sample in the same
        // buffer using the half-cell backward interpolation kernel.
        if self
            .record_spec
            .contains(SensorRecordField::VelocityNonStaggeredX)
        {
            if let Some(ref mut buf) = self.ux_data {
                for (row, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
                    buf[[row, col]] = sample_collocated_velocity(ux, 0, i, j, k);
                }
            }
        }
        if self
            .record_spec
            .contains(SensorRecordField::VelocityNonStaggeredY)
        {
            if let Some(ref mut buf) = self.uy_data {
                for (row, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
                    buf[[row, col]] = sample_collocated_velocity(uy, 1, i, j, k);
                }
            }
        }
        if self
            .record_spec
            .contains(SensorRecordField::VelocityNonStaggeredZ)
        {
            if let Some(ref mut buf) = self.uz_data {
                for (row, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
                    buf[[row, col]] = sample_collocated_velocity(uz, 2, i, j, k);
                }
            }
        }

        // Acoustic intensity accumulation.
        if self.record_spec.records_intensity_x() {
            record_intensity_component(
                self.pressure.as_ref(),
                &mut self.ix_data,
                &mut self.ix_sum,
                ux,
                &self.sensor_indices,
                col,
            );
        }
        if self.record_spec.records_intensity_y() {
            record_intensity_component(
                self.pressure.as_ref(),
                &mut self.iy_data,
                &mut self.iy_sum,
                uy,
                &self.sensor_indices,
                col,
            );
        }
        if self.record_spec.records_intensity_z() {
            record_intensity_component(
                self.pressure.as_ref(),
                &mut self.iz_data,
                &mut self.iz_sum,
                uz,
                &self.sensor_indices,
                col,
            );
        }

        Ok(())
    }
}

// ── Module-level helpers ──────────────────────────────────────────────────────

/// Sample a staggered velocity component at a pressure-grid position.
///
/// Applies the half-cell backward interpolation `u_ns[i] = (u[i-1] + u[i]) / 2`
/// with ghost cell = 0 at the lower boundary.  Direct sensor-position evaluation
/// is algebraically identical to materialising the full collocated field and
/// sampling it, but eliminates the full-grid transient allocation.
#[inline]
fn sample_collocated_velocity(u: &Array3<f64>, axis: usize, i: usize, j: usize, k: usize) -> f64 {
    let current = u[[i, j, k]];
    let previous = match axis {
        0 if i > 0 => u[[i - 1, j, k]],
        1 if j > 0 => u[[i, j - 1, k]],
        2 if k > 0 => u[[i, j, k - 1]],
        _ => 0.0,
    };
    0.5 * (previous + current)
}

/// Accumulate one intensity component into the time-series buffer and running sum.
///
/// The pressure value is read from the already-recorded `pressure` column `col`
/// (populated by the preceding `record_step` call), then multiplied by the
/// velocity component at each sensor grid point.
fn record_intensity_component(
    pressure: Option<&Array2<f64>>,
    data: &mut Option<Array2<f64>>,
    sum: &mut Option<Array1<f64>>,
    velocity: &Array3<f64>,
    sensor_indices: &[(usize, usize, usize)],
    col: usize,
) {
    let Some(pressure) = pressure else {
        return;
    };

    for (row, &(i, j, k)) in sensor_indices.iter().enumerate() {
        let intensity = pressure[[row, col]] * velocity[[i, j, k]];
        if let Some(buf) = data.as_mut() {
            buf[[row, col]] = intensity;
        }
        if let Some(total) = sum.as_mut() {
            total[row] += intensity;
        }
    }
}
