//! Thermal dose accumulation — Sapareto & Dewey (1984) CEM43 model
//!
//! ## Mathematical Foundation
//!
//! **Theorem (Sapareto & Dewey 1984)**
//!
//! The thermal isoeffect dose normalised to 43 °C is:
//!
//! ```text
//! CEM43(x, t) = ∫ R(T(x,τ))^(43 − T(x,τ)) dτ   [cumulative equivalent minutes at 43°C]
//! ```
//!
//! The Arrhenius-derived R coefficient has a step discontinuity at 43 °C:
//!
//! ```text
//! R(T) = 0.5   for T ≥ 43 °C  (protein denaturation regime)
//! R(T) = 0.25  for T <  43 °C  (lower-temperature regime)
//! ```
//!
//! Experimentally calibrated from cell survival curves (Sapareto & Dewey 1984, Table 1;
//! confirmed in Dewhirst et al. 2003, Eq. 1).  The threshold is **43 °C**, not 37 °C.
//!
//! **Discrete accumulation** (Euler forward, timestep Δt in seconds, converted
//! internally to minutes to keep CEM43 in canonical equivalent-minute units):
//!
//! ```text
//! CEM43_{k+1}(x) = CEM43_k(x) + R(T)^(43 − T(x,t_k)) · (Δt / 60)
//! ```
//!
//! Validation points (Sapareto & Dewey 1984, Table 1):
//!
//! | T (°C) | R     | dose_rate (CEM43/min) |
//! |--------|-------|------------------------|
//! | 37     | 0.25  | 0.25^6 ≈ 2.441×10⁻⁴   |
//! | 42     | 0.25  | 0.25^1 = 0.25          |
//! | 43     | 0.5   | 0.5^0  = 1.0           |
//! | 50     | 0.5   | 0.5^(−7) = 128         |
//!
//! ## References
//! - Sapareto SA, Dewey WC (1984). "Thermal dose determination in cancer therapy."
//!   *Int J Radiat Oncol Biol Phys* 10(6):787–800.  DOI:10.1016/0360-3016(84)90379-1
//! - Dewhirst MW et al. (2003). "Basic principles of thermal dosimetry and thermal thresholds
//!   for tissue damage from hyperthermia." *Int J Hyperthermia* 19(3):267–294.

use super::monitor::TranscranialSafetyMonitor;
use aequitas::systems::si::quantities::Time;
use kwavers_core::constants::numerical::SECONDS_PER_MINUTE;
use kwavers_core::error::KwaversResult;

use crate::thermal::response::{checked_cem43_increments, CelsiusStorage};

impl TranscranialSafetyMonitor {
    /// Update CEM43 thermal dose accumulation at each grid point.
    ///
    /// Implements Sapareto & Dewey (1984) exactly:
    ///
    /// ```text
    /// dose_rate = R(T)^(43 − T)
    /// ```
    ///
    /// where `T` is in °C, `R = 0.5` for `T ≥ 43°C`, `R = 0.25` for `T < 43°C`.
    ///
    /// # Arguments
    /// * `dt` – Timestep in seconds. Internally converted to minutes so the
    ///   accumulated `CEM43` is in canonical equivalent-minutes (matching the
    ///   default `max_thermal_dose = 240` threshold from Sapareto & Dewey).
    pub(crate) fn update_thermal_dose(
        &mut self,
        temperature: &leto::Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        checked_cem43_increments::<CelsiusStorage, _>(
            self.thermal_dose_increments.view_mut(),
            temperature.view(),
            Time::from_base(dt),
            |_| true,
        )?;
        let [nx, ny, nz] = temperature.shape();

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let increment = self.thermal_dose_increments[[i, j, k]];
                    let dose_rate = increment * SECONDS_PER_MINUTE / dt;
                    self.thermal_dose.dose_rate[[i, j, k]] = dose_rate;
                    self.thermal_dose.current_dose[[i, j, k]] += increment;

                    let target_dose = self.thresholds.max_thermal_dose;
                    let current_dose = self.thermal_dose.current_dose[[i, j, k]];

                    // Remaining time to reach target dose at current instantaneous rate
                    // dose_rate is CEM43/min so dividing yields minutes; convert to seconds
                    // for the user-facing field.
                    if current_dose < target_dose && dose_rate > 0.0 {
                        self.thermal_dose.time_to_target[[i, j, k]] =
                            (target_dose - current_dose) / dose_rate * SECONDS_PER_MINUTE;
                    } else {
                        self.thermal_dose.time_to_target[[i, j, k]] = 0.0;
                    }

                    if dose_rate > 0.0 {
                        self.thermal_dose.max_safe_time[[i, j, k]] =
                            (target_dose - current_dose).max(0.0) / dose_rate * SECONDS_PER_MINUTE;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
    use leto::Array3;

    /// **Test: CEM43 dose rate at body temperature (37 °C)**
    ///
    /// Reference: Sapareto & Dewey (1984), Table 1.
    /// At T = 37°C: R = 0.25, dose_rate = 0.25^(43−37) = 0.25^6 ≈ 2.441×10⁻⁴ CEM43 s⁻¹.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_cem43_dose_rate_at_body_temperature() {
        let mut monitor = TranscranialSafetyMonitor::new((1, 1, 1), 0.01, 650e3);
        let temperature = Array3::from_elem((1, 1, 1), BODY_TEMPERATURE_C);
        let pressure = Array3::zeros([1, 1, 1]);

        monitor.update_fields(&temperature, &pressure, 1.0).unwrap();

        let expected = 0.25_f64.powf(43.0 - BODY_TEMPERATURE_C);
        let actual = monitor.thermal_dose.dose_rate[[0, 0, 0]];
        let rel_err = (actual - expected).abs() / expected;
        assert!(
            rel_err < 1e-12,
            "CEM43 dose rate at 37°C: got {actual:.6e}, expected {expected:.6e}"
        );
    }

    /// **Test: CEM43 dose rate at the 43 °C threshold**
    ///
    /// At T = 43°C: R switches to 0.5, exponent = 0 → dose_rate = 0.5^0 = 1.0 CEM43 s⁻¹.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_cem43_dose_rate_at_threshold() {
        let mut monitor = TranscranialSafetyMonitor::new((1, 1, 1), 0.01, 650e3);
        // Allow 43°C (default max_temperature = 43.0 so this is exactly at limit — use 44 limit)
        monitor.thresholds.max_temperature = 44.0;
        let temperature = Array3::from_elem((1, 1, 1), 43.0_f64);
        let pressure = Array3::zeros([1, 1, 1]);

        monitor.update_fields(&temperature, &pressure, 1.0).unwrap();

        let actual = monitor.thermal_dose.dose_rate[[0, 0, 0]];
        assert!(
            (actual - 1.0).abs() < 1e-12,
            "CEM43 dose rate at 43°C should be exactly 1.0 CEM43/s, got {actual}"
        );
    }

    /// **Test: CEM43 dose rate above threshold (HIFU ablation regime)**
    ///
    /// At T = 50°C: R = 0.5, dose_rate = 0.5^(43−50) = 0.5^(−7) = 128 CEM43 s⁻¹.
    /// Corresponds to ablative HIFU where tissue damage accumulates rapidly.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_cem43_dose_rate_hifu_ablation() {
        let mut monitor = TranscranialSafetyMonitor::new((1, 1, 1), 0.01, 650e3);
        monitor.thresholds.max_temperature = 60.0;
        monitor.thresholds.max_thermal_dose = 1e9;

        let temperature = Array3::from_elem((1, 1, 1), 50.0_f64);
        let pressure = Array3::zeros([1, 1, 1]);

        monitor.update_fields(&temperature, &pressure, 1.0).unwrap();

        let expected = 0.5_f64.powf(43.0 - 50.0);
        let actual = monitor.thermal_dose.dose_rate[[0, 0, 0]];
        let rel_err = (actual - expected).abs() / expected;
        assert!(
            rel_err < 1e-12,
            "CEM43 dose rate at 50°C: got {actual:.6e}, expected {expected:.6e} (128 CEM43/s)"
        );
    }

    /// **Test: CEM43 linear accumulation at threshold temperature**
    ///
    /// At T = 43°C, dose_rate = 1.0 CEM43/min → after N seconds: CEM43 = N/60.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_cem43_accumulation_linear_at_threshold() {
        let mut monitor = TranscranialSafetyMonitor::new((1, 1, 1), 0.01, 650e3);
        monitor.thresholds.max_temperature = 44.0;
        monitor.thresholds.max_thermal_dose = 1e9;

        let temperature = Array3::from_elem((1, 1, 1), 43.0_f64);
        let pressure = Array3::zeros([1, 1, 1]);

        let n_steps = 600usize;
        let dt = 1.0_f64;
        for _ in 0..n_steps {
            monitor.update_fields(&temperature, &pressure, dt).unwrap();
        }

        let accumulated = monitor.thermal_dose.current_dose[[0, 0, 0]];
        // 600 s at 43 °C = 10 min at 43 °C = 10 CEM43 (canonical Sapareto-Dewey)
        let expected = (n_steps as f64) / 60.0;
        let rel_err = (accumulated - expected).abs() / expected;
        assert!(
            rel_err < 1e-12,
            "Accumulated CEM43 after {n_steps}s at 43°C: got {accumulated:.4}, expected {expected:.4}"
        );
    }

    #[test]
    fn invalid_temperature_preserves_monitor_state() {
        let mut monitor = TranscranialSafetyMonitor::new((2, 1, 1), 0.01, 650e3);
        monitor.thresholds.max_temperature = 60.0;
        let reference = Array3::from_elem((2, 1, 1), 43.0);
        let pressure = Array3::zeros([2, 1, 1]);
        monitor.update_fields(&reference, &pressure, 60.0).unwrap();
        let dose_before = monitor.thermal_dose.current_dose.clone();
        let temperature_before = monitor.temperature.clone();

        let mut invalid = Array3::from_elem((2, 1, 1), 44.0);
        invalid[[1, 0, 0]] = f64::NAN;
        assert!(monitor.update_fields(&invalid, &pressure, 60.0).is_err());
        assert_eq!(monitor.thermal_dose.current_dose, dose_before);
        assert_eq!(monitor.temperature, temperature_before);
    }
}
