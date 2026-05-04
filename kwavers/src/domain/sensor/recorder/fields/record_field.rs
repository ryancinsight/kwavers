//! [`SensorRecordField`] enum and its classification methods.

/// Physical quantity to record at sensor positions during simulation.
///
/// Corresponds directly to k-Wave's `sensor.record` cell array entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorRecordField {
    // ── Pressure ────────────────────────────────────────────────────────────
    /// Pressure time series: `p[sensor, t]`  Pa  (default)
    Pressure,
    /// Spatial maximum pressure at each sensor over all time: `max_t p`  Pa
    PressureMax,
    /// Spatial minimum pressure at each sensor over all time: `min_t p`  Pa
    PressureMin,
    /// RMS pressure at each sensor over all time: `sqrt(mean(p²))`  Pa
    PressureRms,
    /// Pressure at the final time step: `p[sensor, t_end]`  Pa
    PressureFinal,
    /// Domain-global maximum pressure (scalar): `max_sensor max_t p`  Pa
    PressureMaxAll,
    /// Domain-global minimum pressure (scalar): `min_sensor min_t p`  Pa
    PressureMinAll,

    // ── Staggered particle velocity ──────────────────────────────────────────
    /// x-component velocity time series (staggered at i+½): `ux[sensor, t]`  m/s
    VelocityX,
    /// y-component velocity time series (staggered at j+½): `uy[sensor, t]`  m/s
    VelocityY,
    /// z-component velocity time series (staggered at k+½): `uz[sensor, t]`  m/s
    VelocityZ,

    /// Maximum ux at each sensor over all time  m/s
    VelocityMaxX,
    /// Maximum uy at each sensor over all time  m/s
    VelocityMaxY,
    /// Maximum uz at each sensor over all time  m/s
    VelocityMaxZ,

    /// Minimum ux at each sensor over all time  m/s
    VelocityMinX,
    /// Minimum uy at each sensor over all time  m/s
    VelocityMinY,
    /// Minimum uz at each sensor over all time  m/s
    VelocityMinZ,

    /// RMS ux at each sensor over all time  m/s
    VelocityRmsX,
    /// RMS uy at each sensor over all time  m/s
    VelocityRmsY,
    /// RMS uz at each sensor over all time  m/s
    VelocityRmsZ,

    // ── Non-staggered (interpolated) velocity ────────────────────────────────
    /// ux interpolated to pressure grid positions (half-cell back-shift)  m/s
    VelocityNonStaggeredX,
    /// uy interpolated to pressure grid positions  m/s
    VelocityNonStaggeredY,
    /// uz interpolated to pressure grid positions  m/s
    VelocityNonStaggeredZ,

    // ── Acoustic intensity ───────────────────────────────────────────────────
    /// Time-domain x-intensity: `Ix = p·ux[sensor, t]`  W/m²
    IntensityX,
    /// Time-domain y-intensity: `Iy = p·uy[sensor, t]`  W/m²
    IntensityY,
    /// Time-domain z-intensity: `Iz = p·uz[sensor, t]`  W/m²
    IntensityZ,

    /// Time-averaged x-intensity at each sensor: `<p·ux>_t`  W/m²
    IntensityAvgX,
    /// Time-averaged y-intensity at each sensor: `<p·uy>_t`  W/m²
    IntensityAvgY,
    /// Time-averaged z-intensity at each sensor: `<p·uz>_t`  W/m²
    IntensityAvgZ,
}

impl SensorRecordField {
    /// Returns `true` if this is a velocity component (staggered or non-staggered).
    #[must_use]
    pub fn is_velocity(&self) -> bool {
        matches!(
            self,
            Self::VelocityX
                | Self::VelocityY
                | Self::VelocityZ
                | Self::VelocityMaxX
                | Self::VelocityMaxY
                | Self::VelocityMaxZ
                | Self::VelocityMinX
                | Self::VelocityMinY
                | Self::VelocityMinZ
                | Self::VelocityRmsX
                | Self::VelocityRmsY
                | Self::VelocityRmsZ
                | Self::VelocityNonStaggeredX
                | Self::VelocityNonStaggeredY
                | Self::VelocityNonStaggeredZ
        )
    }

    /// Returns `true` if this field requires a velocity time series buffer.
    #[must_use]
    pub fn needs_velocity_time_series(&self) -> bool {
        matches!(
            self,
            Self::VelocityX
                | Self::VelocityY
                | Self::VelocityZ
                | Self::VelocityNonStaggeredX
                | Self::VelocityNonStaggeredY
                | Self::VelocityNonStaggeredZ
        )
    }

    /// Returns `true` if this field requires velocity statistics accumulators.
    #[must_use]
    pub fn needs_velocity_stats(&self) -> bool {
        matches!(
            self,
            Self::VelocityMaxX
                | Self::VelocityMaxY
                | Self::VelocityMaxZ
                | Self::VelocityMinX
                | Self::VelocityMinY
                | Self::VelocityMinZ
                | Self::VelocityRmsX
                | Self::VelocityRmsY
                | Self::VelocityRmsZ
        )
    }

    /// Returns `true` if this field requires pressure statistics accumulators.
    #[must_use]
    pub fn needs_pressure_stats(&self) -> bool {
        matches!(
            self,
            Self::PressureMax
                | Self::PressureMin
                | Self::PressureRms
                | Self::PressureFinal
                | Self::PressureMaxAll
                | Self::PressureMinAll
        )
    }

    /// Returns the k-Wave MATLAB field name string for this record field.
    #[must_use]
    pub fn kwave_name(&self) -> &'static str {
        match self {
            Self::Pressure => "p",
            Self::PressureMax => "p_max",
            Self::PressureMin => "p_min",
            Self::PressureRms => "p_rms",
            Self::PressureFinal => "p_final",
            Self::PressureMaxAll => "p_max_all",
            Self::PressureMinAll => "p_min_all",
            Self::VelocityX => "ux",
            Self::VelocityY => "uy",
            Self::VelocityZ => "uz",
            Self::VelocityMaxX => "ux_max",
            Self::VelocityMaxY => "uy_max",
            Self::VelocityMaxZ => "uz_max",
            Self::VelocityMinX => "ux_min",
            Self::VelocityMinY => "uy_min",
            Self::VelocityMinZ => "uz_min",
            Self::VelocityRmsX => "ux_rms",
            Self::VelocityRmsY => "uy_rms",
            Self::VelocityRmsZ => "uz_rms",
            Self::VelocityNonStaggeredX => "ux_non_staggered",
            Self::VelocityNonStaggeredY => "uy_non_staggered",
            Self::VelocityNonStaggeredZ => "uz_non_staggered",
            Self::IntensityX => "Ix",
            Self::IntensityY => "Iy",
            Self::IntensityZ => "Iz",
            Self::IntensityAvgX => "I_avg_x",
            Self::IntensityAvgY => "I_avg_y",
            Self::IntensityAvgZ => "I_avg_z",
        }
    }
}
