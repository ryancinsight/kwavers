//! HIFU thermal-dose accumulation.
//!
//! CEM43 follows Sapareto-Dewey equivalent minutes at 43 deg C:
//!
//! ```text
//! CEM43 = sum dt_min * R^(43 - T_avg)
//! R = 0.5  for T_avg >= 43 deg C
//! R = 0.25 for T_avg <  43 deg C
//! ```
//!
//! A one-minute exposure at 44 deg C therefore contributes two equivalent
//! minutes at 43 deg C.
//!
//! Reference: Sapareto & Dewey (1984), Int. J. Radiat. Oncol. Biol. Phys.
//! 10(6), 787-800.

use kwavers_core::constants::medical::{
    THERMAL_DOSE_REFERENCE_TEMP_C, THERMAL_DOSE_R_ABOVE_43C, THERMAL_DOSE_R_BELOW_43C,
    THERMAL_DOSE_THRESHOLD,
};
use kwavers_core::constants::numerical::SECONDS_PER_MINUTE;
use kwavers_grid::Grid;
use ndarray::Array3;

/// Thermal dose calculation in cumulative equivalent minutes at 43 deg C.
#[derive(Debug, Clone)]
pub struct HifuThermalDose {
    /// Cumulative equivalent minutes at 43 deg C.
    pub cem43: Array3<f64>,
    /// Temperature history [deg C].
    temperature_history: Vec<Array3<f64>>,
    /// Measurement times [s].
    time_points_s: Vec<f64>,
}

impl HifuThermalDose {
    /// Create new thermal dose calculator.
    #[must_use]
    pub fn new(grid: &Grid) -> Self {
        Self {
            cem43: Array3::zeros(grid.dimensions()),
            temperature_history: Vec::new(),
            time_points_s: Vec::new(),
        }
    }

    /// Add a temperature measurement at time `time_s` seconds.
    pub fn add_temperature_measurement(&mut self, temperature: Array3<f64>, time_s: f64) {
        debug_assert_eq!(
            temperature.dim(),
            self.cem43.dim(),
            "HIFU temperature measurement must match dose-grid dimensions"
        );
        self.temperature_history.push(temperature);
        self.time_points_s.push(time_s);
        self.update_cem43();
    }

    /// Get thermal dose at a grid location.
    #[must_use]
    pub fn dose_at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.cem43[[i, j, k]]
    }

    /// Check if ablation threshold reached (CEM43 > 240 CEM43 min).
    #[must_use]
    pub fn ablation_threshold_reached(&self) -> Array3<bool> {
        self.cem43.mapv(|dose| dose > THERMAL_DOSE_THRESHOLD)
    }

    fn update_cem43(&mut self) {
        if self.temperature_history.len() < 2 {
            return;
        }

        self.cem43.fill(0.0);

        for sample in 1..self.temperature_history.len() {
            let dt_min =
                (self.time_points_s[sample] - self.time_points_s[sample - 1]) / SECONDS_PER_MINUTE;
            if !dt_min.is_finite() || dt_min <= 0.0 {
                continue;
            }

            let temp_prev = &self.temperature_history[sample - 1];
            let temp_curr = &self.temperature_history[sample];
            let prev_slice = temp_prev
                .as_slice()
                .expect("contiguous temperature history");
            let curr_slice = temp_curr
                .as_slice()
                .expect("contiguous temperature history");
            let dose_slice = self.cem43.as_slice_mut().expect("contiguous dose field");

            for index in 0..dose_slice.len() {
                let t_avg = (prev_slice[index] + curr_slice[index]) * 0.5;
                dose_slice[index] += cem43_increment_minutes(t_avg, dt_min);
            }
        }
    }
}

#[inline]
pub(super) fn cem43_increment_minutes(temperature_c: f64, dt_min: f64) -> f64 {
    let r = if temperature_c >= THERMAL_DOSE_REFERENCE_TEMP_C {
        THERMAL_DOSE_R_ABOVE_43C
    } else {
        THERMAL_DOSE_R_BELOW_43C
    };
    dt_min * r.powf(THERMAL_DOSE_REFERENCE_TEMP_C - temperature_c)
}
