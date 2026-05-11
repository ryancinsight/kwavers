/// Checkpoint save/restore for [`SensorRecorder`].
///
/// # Theorem (checkpoint invariant)
///
/// Let `R` be a `SensorRecorder` that has recorded `k` steps into a buffer of
/// capacity `N` (so `R.next_step = k`, `R.expected_steps = N`).  Define the
/// checkpoint triple `(D, k, N)` where `D = R.pressure[.., ..k]`.
///
/// **Restoration contract**: given a fresh `SensorRecorder R'` with
/// `R'.expected_steps = N` and the same sensor index set, calling
/// `R'.restore_from_checkpoint(D, k)` establishes:
///   1. `R'.next_step = k`
///   2. `R'.pressure[.., ..k] = D`  (column-wise equality)
///   3. `R'.pressure[.., k..N]` remains initialised to zero (ready for the
///      remaining `N - k` recording steps).
///
/// These three invariants together guarantee bit-exact continuation: any step
/// appended after restoration writes to column `k` (and beyond) in both the
/// original and the restored recorder.
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array2, ArrayView2};

use super::SensorRecorder;

impl SensorRecorder {
    // ── Checkpoint ───────────────────────────────────────────────────────────

    /// Borrow partial sensor data accumulated so far, for checkpointing.
    ///
    /// Returns `(view, next_step, expected_steps)` where `view` has shape
    /// `(n_sensors, next_step)`. This is the zero-copy checkpoint source of
    /// truth; owned checkpoint creation composes this method with `to_owned()`.
    #[must_use]
    pub fn checkpoint_state_view(&self) -> Option<(ArrayView2<'_, f64>, usize, usize)> {
        Some((
            self.recorded_pressure_view()?,
            self.next_step,
            self.expected_steps,
        ))
    }

    /// Return partial sensor data accumulated so far, for checkpointing.
    ///
    /// Returns `(data, next_step, expected_steps)` where `data` has shape
    /// `(n_sensors, next_step)`. Returns `None` when no sensor mask is active.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use] 
    pub fn checkpoint_state(&self) -> Option<(Array2<f64>, usize, usize)> {
        self.checkpoint_state_view()
            .map(|(view, next_step, expected_steps)| (view.to_owned(), next_step, expected_steps))
    }

    /// Restore recorder state from a checkpoint.
    ///
    /// `data` must have shape `(n_sensors, next_step)`.
    /// # Errors
    /// - Returns [`KwaversError::DimensionMismatch`] if the precondition for mismatched array or grid dimensions is violated.
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn restore_from_checkpoint(
        &mut self,
        data: Array2<f64>,
        next_step: usize,
    ) -> KwaversResult<()> {
        let (n_sensors, n_recorded) = data.dim();
        if n_sensors != self.sensor_indices.len() {
            return Err(KwaversError::DimensionMismatch(format!(
                "checkpoint sensor count {n_sensors} ≠ recorder sensor count {}",
                self.sensor_indices.len()
            )));
        }
        if next_step > self.expected_steps {
            return Err(KwaversError::InvalidInput(format!(
                "checkpoint next_step {next_step} exceeds expected_steps {}",
                self.expected_steps
            )));
        }
        if n_recorded != next_step {
            return Err(KwaversError::InvalidInput(format!(
                "checkpoint data has {n_recorded} columns but next_step={next_step}"
            )));
        }
        let pressure = self
            .pressure
            .get_or_insert_with(|| Array2::zeros((n_sensors, self.expected_steps)));
        pressure
            .slice_mut(ndarray::s![.., ..next_step])
            .assign(&data);
        self.next_step = next_step;
        Ok(())
    }
}
