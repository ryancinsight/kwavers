//! Staggered velocity time-series accessors for [`SensorRecorder`].
//!
//! Provides owned-clone and borrowed-view access to the recorded ux / uy / uz
//! time-series buffers, matching k-Wave's `'ux'`, `'uy'`, `'uz'` record fields.

use super::super::SensorRecorder;
use leto::{Array2, ArrayView2};

impl SensorRecorder {
    // ── ux ───────────────────────────────────────────────────────────────────

    /// Staggered ux time series: `(n_sensors, recorded_steps)`.
    #[must_use]
    pub fn extract_ux_data(&self) -> Option<Array2<f64>> {
        self.ux_data.clone()
    }

    /// Borrow ux time-series data without cloning.
    #[must_use]
    pub fn ux_data_view(&self) -> Option<ArrayView2<'_, f64>> {
        Some(self.ux_data.as_ref()?.view())
    }

    /// Borrow the populated ux time-series prefix without cloning.
    #[must_use]
    pub fn recorded_ux_view(&self) -> Option<ArrayView2<'_, f64>> {
        let data = self.ux_data.as_ref()?;
        data.slice(&[(0, data.shape()[0], 1), (0, self.next_step, 1)])
            .ok()
    }

    // ── uy ───────────────────────────────────────────────────────────────────

    /// Staggered uy time series: `(n_sensors, recorded_steps)`.
    #[must_use]
    pub fn extract_uy_data(&self) -> Option<Array2<f64>> {
        self.uy_data.clone()
    }

    /// Borrow uy time-series data without cloning.
    #[must_use]
    pub fn uy_data_view(&self) -> Option<ArrayView2<'_, f64>> {
        Some(self.uy_data.as_ref()?.view())
    }

    /// Borrow the populated uy time-series prefix without cloning.
    #[must_use]
    pub fn recorded_uy_view(&self) -> Option<ArrayView2<'_, f64>> {
        let data = self.uy_data.as_ref()?;
        data.slice(&[(0, data.shape()[0], 1), (0, self.next_step, 1)])
            .ok()
    }

    // ── uz ───────────────────────────────────────────────────────────────────

    /// Staggered uz time series: `(n_sensors, recorded_steps)`.
    #[must_use]
    pub fn extract_uz_data(&self) -> Option<Array2<f64>> {
        self.uz_data.clone()
    }

    /// Borrow uz time-series data without cloning.
    #[must_use]
    pub fn uz_data_view(&self) -> Option<ArrayView2<'_, f64>> {
        Some(self.uz_data.as_ref()?.view())
    }

    /// Borrow the populated uz time-series prefix without cloning.
    #[must_use]
    pub fn recorded_uz_view(&self) -> Option<ArrayView2<'_, f64>> {
        let data = self.uz_data.as_ref()?;
        data.slice(&[(0, data.shape()[0], 1), (0, self.next_step, 1)])
            .ok()
    }
}
