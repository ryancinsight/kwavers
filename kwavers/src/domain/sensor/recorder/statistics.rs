// recorder/statistics.rs - Recording statistics

use log::info;

/// Statistics collected during recording
#[derive(Debug, Clone, Default)]
pub struct RecorderStatistics {
    pub total_snapshots: usize,
    pub total_cavitation_events: usize,
    pub total_sl_events: usize,
    pub total_thermal_events: usize,
    pub max_pressure: f64,
    pub min_pressure: f64,
    pub max_temperature: f64,
    pub max_light_intensity: f64,
}

impl RecorderStatistics {
    /// Create new statistics instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_snapshots: 0,
            total_cavitation_events: 0,
            total_sl_events: 0,
            total_thermal_events: 0,
            max_pressure: f64::NEG_INFINITY,
            min_pressure: f64::INFINITY,
            max_temperature: f64::NEG_INFINITY,
            max_light_intensity: 0.0,
        }
    }

    /// Update pressure statistics
    pub fn update_pressure(&mut self, pressure: f64) {
        self.max_pressure = self.max_pressure.max(pressure);
        self.min_pressure = self.min_pressure.min(pressure);
    }

    /// Update temperature statistics
    pub fn update_temperature(&mut self, temperature: f64) {
        self.max_temperature = self.max_temperature.max(temperature);
    }

    /// Update light intensity statistics
    pub fn update_light_intensity(&mut self, intensity: f64) {
        self.max_light_intensity = self.max_light_intensity.max(intensity);
    }

    /// Print summary of statistics
    pub fn print_summary(&self) {
        info!("=== Recording Statistics ===");
        info!("Total snapshots: {}", self.total_snapshots);
        info!("Cavitation events: {}", self.total_cavitation_events);
        info!("Sonoluminescence events: {}", self.total_sl_events);
        info!("Thermal events: {}", self.total_thermal_events);
        info!(
            "Pressure range: {:.2e} to {:.2e} Pa",
            self.min_pressure, self.max_pressure
        );
        info!("Max temperature: {:.2} K", self.max_temperature);
        info!("Max light intensity: {:.2e} W/m\u{00b2}", self.max_light_intensity);
    }
}
