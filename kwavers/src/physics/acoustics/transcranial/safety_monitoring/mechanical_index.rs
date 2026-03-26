//! Mechanical index calculation per AIUM/NEMA UD 3-2004

use super::monitor::SafetyMonitor;

impl SafetyMonitor {
    /// Update mechanical index
    ///
    /// MI = p_peak(MPa) / √f(MHz)
    /// Reference: AIUM/NEMA UD 3-2004
    pub(crate) fn update_mechanical_index(&mut self) {
        // Find peak pressure
        let peak_pressure = self
            .pressure
            .iter()
            .map(|&p| p.abs())
            .fold(0.0_f64, f64::max);

        self.mechanical_index.peak_pressure = peak_pressure / 1e6; // Convert to MPa

        // Calculate MI: p_peak / sqrt(f)
        // Reference: AIUM/NEMA UD 3-2004
        let mi = if self.frequency > 0.0 {
            self.mechanical_index.peak_pressure / self.frequency.sqrt() * 1e3 // kHz to Hz conversion
        } else {
            0.0
        };

        self.mechanical_index.current_mi = mi;
        self.mechanical_index.safety_margin = self.thresholds.max_mechanical_index / mi;
    }
}
