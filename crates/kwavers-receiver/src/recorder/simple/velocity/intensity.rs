//! Acoustic intensity time-series and time-average accessors for [`SensorRecorder`].
//!
//! Acoustic intensity at sensor position `s` and time step `n` is:
//!
//! ```text
//!   I_x(s, n) = p(s, n) · u_x(s, n)
//!   I_y(s, n) = p(s, n) · u_y(s, n)
//!   I_z(s, n) = p(s, n) · u_z(s, n)
//! ```
//!
//! The time-averaged intensity accumulates the running sum
//! `Σ_{n} p(s,n)·u(s,n)` and divides by the recorded step count on extraction,
//! matching k-Wave's `'Ix'`, `'Iy'`, `'Iz'`, `'I_avg_x'`, `'I_avg_y'`, `'I_avg_z'`
//! record fields.

use super::super::SensorRecorder;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, Array2, ArrayView2};

impl SensorRecorder {
    // ── ix / iy / iz time series ──────────────────────────────────────────────

    /// Acoustic x-intensity time series: `Ix = p * ux`.
    #[must_use]
    pub fn extract_ix_data(&self) -> Option<Array2<f64>> {
        self.ix_data.clone()
    }

    /// Borrow x-intensity time-series data without cloning.
    #[must_use]
    pub fn recorded_ix_view(&self) -> Option<ArrayView2<'_, f64>> {
        let data = self.ix_data.as_ref()?;
        data.slice(&[(0, data.shape()[0], 1), (0, self.next_step, 1)])
            .ok()
    }

    /// Acoustic y-intensity time series: `Iy = p * uy`.
    #[must_use]
    pub fn extract_iy_data(&self) -> Option<Array2<f64>> {
        self.iy_data.clone()
    }

    /// Borrow y-intensity time-series data without cloning.
    #[must_use]
    pub fn recorded_iy_view(&self) -> Option<ArrayView2<'_, f64>> {
        let data = self.iy_data.as_ref()?;
        data.slice(&[(0, data.shape()[0], 1), (0, self.next_step, 1)])
            .ok()
    }

    /// Acoustic z-intensity time series: `Iz = p * uz`.
    #[must_use]
    pub fn extract_iz_data(&self) -> Option<Array2<f64>> {
        self.iz_data.clone()
    }

    /// Borrow z-intensity time-series data without cloning.
    #[must_use]
    pub fn recorded_iz_view(&self) -> Option<ArrayView2<'_, f64>> {
        let data = self.iz_data.as_ref()?;
        data.slice(&[(0, data.shape()[0], 1), (0, self.next_step, 1)])
            .ok()
    }

    // ── time-averaged intensity ───────────────────────────────────────────────

    /// Time-averaged x-intensity: `<p * ux>_t`.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    #[must_use]
    pub fn extract_i_avg_x(&self) -> Option<Array1<f64>> {
        Some(self.average_intensity(self.ix_sum.as_ref()?))
    }

    /// Fill caller-owned storage with time-averaged x-intensity.
    ///
    /// This avoids allocating a new `Array1` on repeated diagnostic reads.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_i_avg_x(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_average_intensity_component(self.ix_sum.as_ref(), out, "I_avg_x")
    }

    /// Time-averaged y-intensity: `<p * uy>_t`.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    #[must_use]
    pub fn extract_i_avg_y(&self) -> Option<Array1<f64>> {
        Some(self.average_intensity(self.iy_sum.as_ref()?))
    }

    /// Fill caller-owned storage with time-averaged y-intensity.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_i_avg_y(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_average_intensity_component(self.iy_sum.as_ref(), out, "I_avg_y")
    }

    /// Time-averaged z-intensity: `<p * uz>_t`.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    #[must_use]
    pub fn extract_i_avg_z(&self) -> Option<Array1<f64>> {
        Some(self.average_intensity(self.iz_sum.as_ref()?))
    }

    /// Fill caller-owned storage with time-averaged z-intensity.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn fill_i_avg_z(&self, out: &mut Array1<f64>) -> KwaversResult<()> {
        self.fill_average_intensity_component(self.iz_sum.as_ref(), out, "I_avg_z")
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Divide a running intensity sum by the number of recorded steps.
    ///
    /// Returns a zero vector before any step has been recorded.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn average_intensity(&self, sum: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros([sum.len()]);
        let _ = self.fill_average_intensity(sum, &mut out);
        out
    }

    fn fill_average_intensity_component(
        &self,
        sum: Option<&Array1<f64>>,
        out: &mut Array1<f64>,
        field: &str,
    ) -> KwaversResult<()> {
        let Some(sum) = sum else {
            return Err(KwaversError::InvalidInput(format!(
                "{field} was not requested by SensorRecordSpec"
            )));
        };
        self.fill_average_intensity(sum, out)
    }

    fn fill_average_intensity(
        &self,
        sum: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> KwaversResult<()> {
        if out.len() != sum.len() {
            return Err(KwaversError::DimensionMismatch(format!(
                "intensity average output length {} != sensor count {}",
                out.len(),
                sum.len()
            )));
        }

        if self.next_step == 0 {
            out.fill(0.0);
            return Ok(());
        }

        let inv_steps = 1.0 / self.next_step as f64;
        for (dst, &value) in out.iter_mut().zip(sum.iter()) {
            *dst = value * inv_steps;
        }
        Ok(())
    }
}