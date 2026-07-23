//! HIFU thermal-dose accumulation.
//!
//! CEM43 evaluation delegates to Asclepius over trapezoidal interval-average
//! temperatures. A one-minute exposure at 44 deg C therefore contributes two
//! equivalent minutes at 43 deg C.
//!
//! Reference: Sapareto & Dewey (1984), Int. J. Radiat. Oncol. Biol. Phys.
//! 10(6), 787-800.

use aequitas::systems::si::quantities::{ThermodynamicTemperature, Time};
use asclepius::response::thermal::Cem43;
use kwavers_core::constants::medical::THERMAL_DOSE_THRESHOLD;
use kwavers_core::constants::numerical::SECONDS_PER_MINUTE;
use kwavers_core::constants::thermodynamic::KELVIN_OFFSET_C;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use leto::Array3;

use crate::thermal::CumulativeEquivalentMinutes;

/// Thermal dose calculation in cumulative equivalent minutes at 43 deg C.
#[derive(Debug, Clone)]
pub struct HifuThermalDose {
    /// Cumulative equivalent minutes at 43 deg C.
    cem43: Array3<f64>,
    /// Reusable checked interval increments.
    increments: Array3<f64>,
    /// Temperature history [deg C].
    temperature_history: Vec<Array3<f64>>,
    /// Measurement times `s`.
    time_points: Vec<Time<f64>>,
}

impl HifuThermalDose {
    /// Create new thermal dose calculator.
    #[must_use]
    pub fn new(grid: &Grid) -> Self {
        Self {
            cem43: Array3::zeros(grid.dimensions()),
            increments: Array3::zeros(grid.dimensions()),
            temperature_history: Vec::new(),
            time_points: Vec::new(),
        }
    }

    /// Add a temperature measurement at typed time `time`.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions differ, time is non-finite or not
    /// increasing, or Asclepius rejects an absolute temperature. Persistent
    /// history and dose remain unchanged on failure.
    pub fn add_temperature_measurement(
        &mut self,
        temperature: Array3<f64>,
        time: Time<f64>,
    ) -> KwaversResult<()> {
        if temperature.shape() != self.cem43.shape() {
            return Err(KwaversError::DimensionMismatch(format!(
                "HIFU temperature shape {:?} does not match dose shape {:?}",
                temperature.shape(),
                self.cem43.shape()
            )));
        }
        let time_s = time.into_base();
        if !time_s.is_finite() || time_s < 0.0 {
            return Err(KwaversError::InvalidInput(
                "HIFU measurement time must be finite".to_string(),
            ));
        }

        if let (Some(previous), Some(&previous_time)) =
            (self.temperature_history.last(), self.time_points.last())
        {
            let step = Time::from_base(time_s - previous_time.into_base());
            let law = Cem43::<f64>::canonical();
            let previous = previous
                .as_slice()
                .expect("invariant: HIFU temperature history is dense");
            let current = temperature
                .as_slice()
                .expect("invariant: HIFU temperature measurement is dense");
            let increments = self
                .increments
                .as_slice_mut()
                .expect("invariant: HIFU increment field is dense");

            for ((increment, &previous_c), &current_c) in
                increments.iter_mut().zip(previous).zip(current)
            {
                let average_c = previous_c.midpoint(current_c);
                *increment = law
                    .increment(
                        ThermodynamicTemperature::from_base(average_c + KELVIN_OFFSET_C),
                        step,
                    )
                    .map_err(|source| {
                        KwaversError::InvalidInput(format!(
                            "HIFU CEM43 observation is invalid: {source}"
                        ))
                    })?
                    .get()
                    .into_base()
                    / SECONDS_PER_MINUTE;
            }

            let dose = self
                .cem43
                .as_slice_mut()
                .expect("invariant: HIFU dose field is dense");
            for (value, increment) in dose.iter_mut().zip(increments) {
                *value += *increment;
            }
        } else {
            let law = Cem43::<f64>::canonical();
            for &temperature_c in temperature.iter() {
                law.rate(ThermodynamicTemperature::from_base(
                    temperature_c + KELVIN_OFFSET_C,
                ))
                .map_err(|source| {
                    KwaversError::InvalidInput(format!(
                        "HIFU CEM43 observation is invalid: {source}"
                    ))
                })?;
            }
        }

        self.temperature_history.push(temperature);
        self.time_points.push(time);
        Ok(())
    }

    /// Get thermal dose at a grid location.
    pub fn dose_at(
        &self,
        i: usize,
        j: usize,
        k: usize,
    ) -> KwaversResult<CumulativeEquivalentMinutes> {
        let value = self.cem43.get([i, j, k]).map_err(|source| {
            KwaversError::InvalidInput(format!(
                "HIFU CEM43 dose index [{i}, {j}, {k}] is outside the grid: {source}"
            ))
        })?;
        CumulativeEquivalentMinutes::try_from_minutes(*value)
    }

    /// Check if ablation threshold reached (CEM43 > 240 CEM43 min).
    #[must_use]
    pub fn ablation_threshold_reached(&self) -> Array3<bool> {
        let threshold = CumulativeEquivalentMinutes::try_from_minutes(THERMAL_DOSE_THRESHOLD)
            .expect("invariant: canonical CEM43 threshold is finite and non-negative")
            .as_minutes();
        let [nx, ny, nz] = self.cem43.shape();
        let mut result = Array3::from_elem([nx, ny, nz], false);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    result[[i, j, k]] = self.cem43[[i, j, k]] > threshold;
                }
            }
        }
        result
    }
}
