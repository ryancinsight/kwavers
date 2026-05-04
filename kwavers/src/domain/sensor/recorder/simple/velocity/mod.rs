//! Velocity recording methods for [`SensorRecorder`].
//!
//! Implements time-series accumulation and statistics for all three velocity
//! components and acoustic intensity, matching k-Wave's `sensor.record` API:
//! `'ux'`, `'uy'`, `'uz'`, `'ux_max'`, `'ux_rms'`, `'ux_non_staggered'`,
//! `'Ix'`, `'Iy'`, `'Iz'`, `'I_avg_x'`, `'I_avg_y'`, `'I_avg_z'`.
//!
//! # Module structure
//!
//! | Sub-module   | Responsibility                                           |
//! |-------------|----------------------------------------------------------|
//! | `series`     | `ux` / `uy` / `uz` time-series views and owned extraction |
//! | `intensity`  | `ix` / `iy` / `iz` time-series and time-averaged intensity |
//! | `stats`      | Per-component velocity max / min / rms statistics          |
//! | `recording`  | [`record_velocity_step`] and collocated-velocity kernel    |
//!
//! [`record_velocity_step`]: super::SensorRecorder::record_velocity_step

mod intensity;
mod recording;
mod series;
mod stats;

#[cfg(test)]
mod tests;
