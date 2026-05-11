//! Per-component velocity statistics accessors for [`SensorRecorder`].
//!
//! Exposes running max, min, and RMS accumulators sampled at sensor positions,
//! matching k-Wave's `'ux_max'`, `'ux_min'`, `'ux_rms'` (and uy / uz variants)
//! record fields.

use super::super::SensorRecorder;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::sensor::recorder::velocity_statistics::{
    SampledVelocityStats, VelocityComponentStats,
};
use ndarray::Array1;

impl SensorRecorder {
    // ── ux statistics ─────────────────────────────────────────────────────────

    /// Extract ux max.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_ux_max(&self) -> Option<Array1<f64>> {
        Some(self.ux_stats.as_ref()?.sample_max(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled ux maximum values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_ux_max(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_velocity_stat(
            out,
            "ux_max",
            self.ux_stats.as_ref(),
            VelocityComponentStats::fill_max,
        )
    }

    /// Extract ux min.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_ux_min(&self) -> Option<Array1<f64>> {
        Some(self.ux_stats.as_ref()?.sample_min(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled ux minimum values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_ux_min(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_velocity_stat(
            out,
            "ux_min",
            self.ux_stats.as_ref(),
            VelocityComponentStats::fill_min,
        )
    }

    /// Extract ux rms.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_ux_rms(&self) -> Option<Array1<f64>> {
        Some(self.ux_stats.as_ref()?.sample_rms(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled ux RMS values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_ux_rms(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_velocity_stat(
            out,
            "ux_rms",
            self.ux_stats.as_ref(),
            VelocityComponentStats::fill_rms,
        )
    }

    // ── uy statistics ─────────────────────────────────────────────────────────

    /// Extract uy max.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_uy_max(&self) -> Option<Array1<f64>> {
        Some(self.uy_stats.as_ref()?.sample_max(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled uy maximum values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_uy_max(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_velocity_stat(
            out,
            "uy_max",
            self.uy_stats.as_ref(),
            VelocityComponentStats::fill_max,
        )
    }

    /// Extract uy min.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_uy_min(&self) -> Option<Array1<f64>> {
        Some(self.uy_stats.as_ref()?.sample_min(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled uy minimum values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_uy_min(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_velocity_stat(
            out,
            "uy_min",
            self.uy_stats.as_ref(),
            VelocityComponentStats::fill_min,
        )
    }

    /// Extract uy rms.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_uy_rms(&self) -> Option<Array1<f64>> {
        Some(self.uy_stats.as_ref()?.sample_rms(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled uy RMS values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_uy_rms(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_velocity_stat(
            out,
            "uy_rms",
            self.uy_stats.as_ref(),
            VelocityComponentStats::fill_rms,
        )
    }

    // ── uz statistics ─────────────────────────────────────────────────────────

    /// Extract uz max.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_uz_max(&self) -> Option<Array1<f64>> {
        Some(self.uz_stats.as_ref()?.sample_max(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled uz maximum values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_uz_max(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_velocity_stat(
            out,
            "uz_max",
            self.uz_stats.as_ref(),
            VelocityComponentStats::fill_max,
        )
    }

    /// Extract uz min.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_uz_min(&self) -> Option<Array1<f64>> {
        Some(self.uz_stats.as_ref()?.sample_min(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled uz minimum values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_uz_min(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_velocity_stat(
            out,
            "uz_min",
            self.uz_stats.as_ref(),
            VelocityComponentStats::fill_min,
        )
    }

    /// Extract uz rms.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[must_use]
    pub fn extract_uz_rms(&self) -> Option<Array1<f64>> {
        Some(self.uz_stats.as_ref()?.sample_rms(&self.sensor_indices))
    }

    /// Fill caller-owned storage with sampled uz RMS values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_uz_rms(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_velocity_stat(
            out,
            "uz_rms",
            self.uz_stats.as_ref(),
            VelocityComponentStats::fill_rms,
        )
    }

    // ── aggregate ─────────────────────────────────────────────────────────────

    /// Sample velocity statistics at sensor positions for all three components.
    ///
    /// Returns `None` unless stats accumulators for **all three** components were
    /// initialised via [`with_spec`](SensorRecorder::with_spec). Callers that
    /// request only per-component stats should use the individual extractors
    /// (`extract_ux_max`, etc.) instead.
    #[must_use]
    pub fn extract_sampled_velocity_stats(&self) -> Option<SampledVelocityStats> {
        let ux = self.ux_stats.as_ref()?;
        let uy = self.uy_stats.as_ref()?;
        let uz = self.uz_stats.as_ref()?;
        let pos = &self.sensor_indices;
        Some(SampledVelocityStats {
            ux_max: ux.sample_max(pos),
            ux_min: ux.sample_min(pos),
            ux_rms: ux.sample_rms(pos),
            uy_max: uy.sample_max(pos),
            uy_min: uy.sample_min(pos),
            uy_rms: uy.sample_rms(pos),
            uz_max: uz.sample_max(pos),
            uz_min: uz.sample_min(pos),
            uz_rms: uz.sample_rms(pos),
        })
    }

    fn fill_velocity_stat(
        &self,
        out: &mut Array1<f64>,
        field: &str,
        stats: Option<&VelocityComponentStats>,
        fill: fn(
            &VelocityComponentStats,
            &[(usize, usize, usize)],
            &mut Array1<f64>,
        ) -> KwaversResult<()>,
    ) -> KwaversResult<()> {
        let Some(stats) = stats else {
            return Err(KwaversError::InvalidInput(format!(
                "{field} was not requested by SensorRecordSpec"
            )));
        };
        fill(stats, &self.sensor_indices, out)
    }
}
