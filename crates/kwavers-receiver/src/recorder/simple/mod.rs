//! Simple (array-mask) sensor recorder.
//!
//! Supports k-Wave's `sensor.record` API.  Pressure time-series and spatial
//! statistics are always available.  Full k-Wave parity (velocity time-series,
//! velocity stats, non-staggered velocity) requires [`SensorRecorder::with_spec`].
//!
//! # Module layout
//!
//! | Sub-module      | Responsibility                                |
//! |-----------------|-----------------------------------------------|
//! | `construction`  | Constructors (`new`, `with_modes`, `with_spec`) |
//! | `extraction`    | Pressure data/stat accessors                  |
//! | `recording`     | `record_step`                                 |
//! | `velocity`      | Velocity recording and extraction             |
//! | `checkpoint`    | Checkpoint state serialisation                |

use crate::recorder::fields::SensorRecordSpec;
use crate::recorder::pressure_statistics::PressureFieldStatistics;
use crate::recorder::velocity_statistics::VelocityComponentStats;
use ndarray::{Array1, Array2};

mod checkpoint;
mod construction;
mod extraction;
mod recording;
mod velocity;

#[cfg(test)]
mod tests;

/// Sensor recorder: accumulates pressure and/or velocity data at sensor positions.
///
/// # Invariants
///
/// * `sensor_indices` is ordered x-fastest (Fortran order) matching k-Wave.
/// * `pressure`, `ux_data`, `uy_data`, `uz_data` have shape
///   `(n_sensors, expected_steps)` when allocated.
/// * `ix_sum`, `iy_sum`, `iz_sum` have length `n_sensors` when allocated.
/// * `next_step ≤ expected_steps` at all times.
#[derive(Debug, Clone)]
pub struct SensorRecorder {
    pub(super) sensor_indices: Vec<(usize, usize, usize)>,
    pub(super) pressure: Option<Array2<f64>>,
    pub(super) expected_steps: usize,
    pub(super) next_step: usize,
    pub(super) stats: Option<PressureFieldStatistics>,
    /// k-Wave record specification — determines what quantities are accumulated.
    pub(super) record_spec: SensorRecordSpec,
    /// Staggered ux time series at sensor positions: `(n_sensors, expected_steps)`.
    pub(super) ux_data: Option<Array2<f64>>,
    /// Staggered uy time series at sensor positions.
    pub(super) uy_data: Option<Array2<f64>>,
    /// Staggered uz time series at sensor positions.
    pub(super) uz_data: Option<Array2<f64>>,
    /// Acoustic x-intensity time series at sensor positions.
    pub(super) ix_data: Option<Array2<f64>>,
    /// Acoustic y-intensity time series at sensor positions.
    pub(super) iy_data: Option<Array2<f64>>,
    /// Acoustic z-intensity time series at sensor positions.
    pub(super) iz_data: Option<Array2<f64>>,
    /// Running sum of x-intensity for time averages.
    pub(super) ix_sum: Option<Array1<f64>>,
    /// Running sum of y-intensity for time averages.
    pub(super) iy_sum: Option<Array1<f64>>,
    /// Running sum of z-intensity for time averages.
    pub(super) iz_sum: Option<Array1<f64>>,
    /// Running ux statistics (max/min/rms) over the full velocity grid.
    pub(super) ux_stats: Option<VelocityComponentStats>,
    /// Running uy statistics.
    pub(super) uy_stats: Option<VelocityComponentStats>,
    /// Running uz statistics.
    pub(super) uz_stats: Option<VelocityComponentStats>,
}
