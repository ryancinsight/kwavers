//! Mechanical index calculation per AIUM/NEMA UD 3-2004

use super::monitor::TranscranialSafetyMonitor;
use crate::acoustics::analysis::calculate_mechanical_index;
use kwavers_core::constants::numerical::MPA_TO_PA;

impl TranscranialSafetyMonitor {
    /// Update mechanical index
    ///
    /// MI = p_peak(MPa) / √f(MHz)
    /// Reference: AIUM/NEMA UD 3-2004
    pub(crate) fn update_mechanical_index(&mut self) {
        let pressure_domain_valid = self.pressure.iter().all(|pressure| pressure.is_finite());
        let peak_pressure = if pressure_domain_valid {
            self.pressure
                .iter()
                .map(|&pressure| pressure.abs())
                .fold(0.0_f64, f64::max)
        } else {
            f64::INFINITY
        };

        self.mechanical_index.peak_pressure = peak_pressure / MPA_TO_PA; // Convert to MPa

        let mi = if pressure_domain_valid && self.frequency.is_finite() && self.frequency > 0.0 {
            calculate_mechanical_index(peak_pressure, self.frequency)
        } else {
            f64::INFINITY
        };

        self.mechanical_index.current_mi = mi;
        self.mechanical_index.safety_margin = if mi == 0.0 {
            f64::INFINITY
        } else if mi.is_finite() {
            self.thresholds.max_mechanical_index / mi
        } else {
            0.0
        };
    }
}
