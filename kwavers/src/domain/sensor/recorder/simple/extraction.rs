//! Pressure data extraction and accessor methods for SensorRecorder.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::sensor::recorder::pressure_statistics::{
    PressureFieldStatistics, SampledStatistics,
};
use ndarray::{Array1, Array2, ArrayView2};

use super::SensorRecorder;

impl SensorRecorder {
    /// Sensor indices in Fortran order (x-fastest).
    #[must_use] 
    pub fn sensor_indices(&self) -> &[(usize, usize, usize)] {
        &self.sensor_indices
    }

    /// Returns `true` if any velocity quantity is requested by the record spec.
    #[must_use]
    pub fn needs_velocity(&self) -> bool {
        self.record_spec.needs_any_velocity()
    }

    // ── Pressure extraction ──────────────────────────────────────────────────

    /// Clone the full recorded pressure buffer `(n_sensors, expected_steps)`.
    #[must_use] 
    pub fn extract_pressure_data(&self) -> Option<Array2<f64>> {
        self.pressure.clone()
    }

    /// Borrow the full pressure buffer without cloning.
    ///
    /// Includes not-yet-recorded columns (still zero-initialised).  Use
    /// [`recorded_pressure_view`](Self::recorded_pressure_view) for the
    /// prefix actually populated.
    #[must_use]
    pub fn pressure_data_view(&self) -> Option<ArrayView2<'_, f64>> {
        Some(self.pressure.as_ref()?.view())
    }

    /// Borrow only the recorded pressure prefix `(n_sensors, next_step)`.
    #[must_use]
    pub fn recorded_pressure_view(&self) -> Option<ArrayView2<'_, f64>> {
        Some(
            self.pressure
                .as_ref()?
                .slice(ndarray::s![.., ..self.next_step]),
        )
    }

    // ── Pressure statistics ──────────────────────────────────────────────────

    /// Borrow the full-grid `PressureFieldStatistics` accumulator, if one
    /// was requested. Each field inside is `Array3<f64>` of shape `(nx,
    /// ny, nz)` covering the entire simulation grid (not just sensor
    /// positions). Returns `None` when the recorder was constructed
    /// without any statistics modes.
    ///
    /// This is the canonical access path for cavitation-kernel
    /// generation: `p_min` is the per-voxel peak rarefactional
    /// pressure that drives intrinsic-threshold cavitation per
    /// Maxwell 2013, `p_max` is the peak compressional pressure, and
    /// `p_rms` is the time-averaged RMS used for thermal-dose proxies.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn full_pressure_statistics(&self) -> Option<&PressureFieldStatistics> {
        self.stats.as_ref()
    }

    /// Extract p max.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_p_max(&self) -> Option<Array1<f64>> {
        Some(self.stats.as_ref()?.sample_p_max(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled maximum pressure.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_p_max(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_pressure_stat(out, "p_max", PressureFieldStatistics::fill_p_max)
    }

    /// Extract p min.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_p_min(&self) -> Option<Array1<f64>> {
        Some(self.stats.as_ref()?.sample_p_min(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled minimum pressure.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_p_min(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_pressure_stat(out, "p_min", PressureFieldStatistics::fill_p_min)
    }

    /// Extract p rms.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_p_rms(&self) -> Option<Array1<f64>> {
        Some(self.stats.as_ref()?.sample_p_rms(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled RMS pressure.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_p_rms(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_pressure_stat(out, "p_rms", PressureFieldStatistics::fill_p_rms)
    }

    /// Extract p final.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_p_final(&self) -> Option<Array1<f64>> {
        Some(self.stats.as_ref()?.sample_p_final(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled final pressure.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_p_final(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_pressure_stat(out, "p_final", PressureFieldStatistics::fill_p_final)
    }

    /// Extract all stats.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_all_stats(&self) -> Option<SampledStatistics> {
        Some(
            self.stats
                .as_ref()?
                .sample_at_positions(&self.sensor_indices),
        )
    }

    // ── Shared helper ────────────────────────────────────────────────────────

    fn fill_pressure_stat(
        &self,
        out: &mut Array1<f64>,
        field: &str,
        fill: fn(
            &PressureFieldStatistics,
            &[(usize, usize, usize)],
            &mut Array1<f64>,
        ) -> KwaversResult<()>,
    ) -> KwaversResult<()> {
        let Some(stats) = self.stats.as_ref() else {
            return Err(KwaversError::InvalidInput(format!(
                "{field} was not requested by the recorder"
            )));
        };
        fill(stats, &self.sensor_indices, out)
    }
}
