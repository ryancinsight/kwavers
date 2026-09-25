//! HIFU thermal-dose accumulation.
//!
//! CEM43 evaluation delegates to Asclepius over trapezoidal interval-average
//! temperatures. A one-minute exposure at 44 deg C therefore contributes two
//! equivalent minutes at 43 deg C.
//!
//! Reference: Sapareto & Dewey (1984), Int. J. Radiat. Oncol. Biol. Phys.
//! 10(6), 787-800.

use kwavers_core::constants::medical::THERMAL_DOSE_THRESHOLD;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use leto::Array3;

use crate::thermal::response::{CelsiusStorage, Cem43Accumulator};

/// Thermal dose calculation in cumulative equivalent minutes at 43 deg C.
///
/// A thin typed wrapper over the shared [`Cem43Accumulator`] in interval
/// (trapezoidal) mode: it retains the single previous Celsius measurement so
/// each new observation contributes the midpoint equivalent minutes between
/// the two samples.
#[derive(Debug, Clone)]
pub struct HifuThermalDose {
    accumulator: Cem43Accumulator<CelsiusStorage, 1>,
}

impl HifuThermalDose {
    /// Create new thermal dose calculator.
    #[must_use]
    pub fn new(grid: &Grid) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        Self {
            accumulator: Cem43Accumulator::new([nx, ny, nz]),
        }
    }

    /// Add a temperature measurement at time `time_s` seconds.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions differ, time is non-finite, or
    /// Asclepius rejects an absolute temperature. Persistent history and dose
    /// remain unchanged on failure.
    ///
    /// # Panics
    ///
    /// Panics only if a dose or temperature array violates the dense-storage
    /// invariant required by the zero-copy accumulation path.
    pub fn add_temperature_measurement(
        &mut self,
        temperature: Array3<f64>,
        time_s: f64,
    ) -> KwaversResult<()> {
        if temperature.shape() != self.accumulator.shape() {
            return Err(KwaversError::DimensionMismatch(format!(
                "HIFU temperature shape {:?} does not match dose shape {:?}",
                temperature.shape(),
                self.accumulator.shape()
            )));
        }
        if !time_s.is_finite() {
            return Err(KwaversError::InvalidInput(
                "HIFU measurement time must be finite".to_string(),
            ));
        }

        self.accumulator.accumulate_interval(temperature, time_s)
    }

    /// Cumulative equivalent minutes at 43 deg C.
    #[must_use]
    pub fn cem43(&self) -> &Array3<f64> {
        self.accumulator.dose()
    }

    /// Get thermal dose at a grid location.
    #[must_use]
    pub fn dose_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.accumulator.dose_at(i, j, k)
    }

    /// Check if ablation threshold reached (CEM43 > 240 CEM43 min).
    #[must_use]
    pub fn ablation_threshold_reached(&self) -> Array3<bool> {
        self.accumulator
            .dose()
            .mapv(|dose| dose > THERMAL_DOSE_THRESHOLD)
    }
}
