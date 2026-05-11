//! `VelocityMapConfig` — velocity mapping configuration.

/// Configuration for velocity mapping.
#[derive(Debug, Clone)]
pub struct VelocityMapConfig {
    /// Physical extent of the lateral (x) dimension (m).
    pub x_extent: f64,
    /// Physical extent of the axial (z) dimension (m).
    pub z_extent: f64,
    /// Grid pixel size (m). Default: 10 μm.
    pub pixel_size: f64,
    /// Frame duration Δt (s) (= 1 / frame_rate). Default: 1e-3 s (1 kHz).
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
