//! Sensor record field specification — k-Wave parity API.
//!
//! # Mathematical Background
//!
//! Defines the set of physical quantities that can be recorded at sensor
//! positions during a time-domain acoustic simulation, following the k-Wave
//! MATLAB toolbox convention (`sensor.record` cell array).
//!
//! ## Available Fields
//!
//! ### Pressure (scalar, at pressure-grid positions)
//! - **`Pressure`**: Time series `p[sensor, t]`  Pa
//! - **`PressureMax`**: `max_t p[sensor]`         Pa
//! - **`PressureMin`**: `min_t p[sensor]`         Pa
//! - **`PressureRms`**: `sqrt(mean(p²))[sensor]`  Pa
//! - **`PressureFinal`**: `p[sensor, t_end]`      Pa
//! - **`PressureMaxAll`**: `max_sensor max_t p`   Pa (global)
//! - **`PressureMinAll`**: `min_sensor min_t p`   Pa (global)
//!
//! ### Particle Velocity (staggered grid)
//!
//! Velocity components are on a **staggered grid** (Yee/k-Wave convention).
//! For an N×1×1 grid with spacing Δx:
//! - Pressure `p`i`` lives at position `i·Δx`
//! - Velocity `ux`i`` lives at position `(i + ½)·Δx`
//!
//! - **`VelocityX/Y/Z`**: Time series `u_α[sensor, t]`  m/s
//! - **`VelocityMaxX/Y/Z`**: `max_t u_α[sensor]`        m/s
//! - **`VelocityMinX/Y/Z`**: `min_t u_α[sensor]`        m/s
//! - **`VelocityRmsX/Y/Z`**: `sqrt(mean(u_α²))[sensor]` m/s
//!
//! ### Non-staggered Velocity (interpolated to pressure-grid positions)
//!
//! ## Theorem: Spectral interpolation to pressure grid
//! For ux on the staggered grid (position i+½), the value at the collocated
//! pressure position i is obtained by a half-cell backward shift in k-space:
//! ```text
//!   ux_ns`i` = IFFT( exp(−i·kx·Δx/2) · FFT(ux) )`i`
//! ```
//! which is equivalent to `(ux[i−1] + ux`i`) / 2` for constant-spacing grids
//! under band-limited interpolation (Boyd 2001, §3.2).
//!
//! - **`VelocityNonStaggeredX/Y/Z`**: `ux_ns[sensor, t]` m/s
//!
//! ### Acoustic Intensity
//!
//! Time-domain intensity vector `I = p · u` (W/m²):
//! - **`IntensityX/Y/Z`**: `Ix = p·ux[sensor, t]`  W/m²
//! - **`IntensityAvgX/Y/Z`**: `<Ix>_t[sensor]`      W/m²
//!
//! ## References
//!
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.
//! - Boyd (2001). Chebyshev and Fourier Spectral Methods. Dover.
//! - k-Wave MATLAB Toolbox: `sensor.record` documentation.

mod record_field;
mod record_spec;
#[cfg(test)]
mod tests;

pub use record_field::SensorRecordField;
pub use record_spec::SensorRecordSpec;
