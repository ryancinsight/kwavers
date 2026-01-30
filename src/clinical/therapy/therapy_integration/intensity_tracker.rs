//! Real-Time Acoustic Intensity Tracking System
//!
//! This module implements continuous monitoring of acoustic intensity fields
//! with temporal averaging, peak tracking, and safety-relevant metrics.
//!
//! # Metrics Provided
//!
//! - **SPTA** (Spatial Peak Temporal Average): FDA-regulated safety metric
//! - **ISPPA** (Spatial Peak Pulse Average): Peak intensity within pulse
//! - **I_tas** (Temporal Average Spatial): Time-averaged field intensity
//! - **Thermal Dose** (CEM43): Cumulative equivalent minutes at 43°C
//! - **Peak Intensity**: Maximum instantaneous pressure²/(ρc)
//!
//! # Clinical Relevance
//!
//! - SPTA < 720 mW/cm² for FDA-compliant diagnostic ultrasound
//! - SPTA > 100 W/cm² typical for therapeutic HIFU
//! - CEM43 < 240 minutes to prevent tissue necrosis
//! - Peak intensity determines cavitation nucleation
//!
//! # References
//!
//! - IEC 62359:2010 - Ultrasonics - Field characterization
//! - FDA 510(k) Guidance - Acoustic Output Measurement
//! - Sapareto & Dewey (1984) - Thermal dose determination

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Acoustic intensity measurement at a point in time
#[derive(Debug, Clone, Copy)]
pub struct InstantaneousIntensity {
    /// Spatial peak pulse average (W/m²)
    pub isppa: f64,
    /// Spatial peak (instantaneous maximum)
    pub spatial_peak: f64,
    /// Spatial average within focal region
    pub spatial_average: f64,
    /// Measurement time (seconds)
    pub timestamp: f64,
}

/// Temporal-averaged intensity metrics
#[derive(Debug, Clone, Copy)]
pub struct TemporalIntensityMetrics {
    /// Spatial peak temporal average (FDA metric, W/m²)
    pub spta: f64,
    /// Temporal average spatial average
    pub tas: f64,
    /// Peak measured SPTA within monitoring window
    pub peak_spta: f64,
    /// Minimum SPTA (usually near zero)
    pub min_spta: f64,
    /// Number of measurements averaged
    pub sample_count: usize,
}

impl Default for TemporalIntensityMetrics {
    fn default() -> Self {
        Self {
            spta: 0.0,
            tas: 0.0,
            peak_spta: 0.0,
            min_spta: f64::MAX,
            sample_count: 0,
        }
    }
}

/// Thermal dose tracking (CEM43 model)
#[derive(Debug, Clone, Copy)]
pub struct ThermalDose {
    /// Cumulative equivalent minutes at 43°C
    pub cem43: f64,
    /// Current temperature (°C)
    pub current_temperature: f64,
    /// Maximum temperature recorded (°C)
    pub max_temperature: f64,
    /// Temperature rise above baseline (°C)
    pub temperature_rise: f64,
}

impl Default for ThermalDose {
    fn default() -> Self {
        Self {
            cem43: 0.0,
            current_temperature: 37.0, // Normal body temperature
            max_temperature: 37.0,
            temperature_rise: 0.0,
        }
    }
}

/// Real-time acoustic intensity tracker
///
/// Maintains continuous history of intensity measurements with rolling
/// temporal averaging windows for FDA compliance and safety monitoring.
#[derive(Debug, Clone)]
pub struct IntensityTracker {
    /// Maximum window size (seconds) for temporal averaging
    max_window_duration: f64,

    /// Time step between measurements (seconds)
    dt: f64,

    /// History of instantaneous intensity measurements
    intensity_history: Vec<InstantaneousIntensity>,

    /// Current temporal average metrics
    current_metrics: TemporalIntensityMetrics,

    /// Thermal dose accumulation
    thermal_dose: ThermalDose,

    /// Peak intensity ever measured (W/m²)
    peak_intensity: f64,

    /// Current simulation time (seconds)
    current_time: f64,
}

impl IntensityTracker {
    /// Create new intensity tracker
    ///
    /// # Arguments
    ///
    /// - `max_window_duration`: Maximum temporal averaging window (seconds)
    ///   - Typical: 0.1 seconds (100 ms window)
    /// - `dt`: Time between measurements (seconds)
    ///   - Typical: 10 microseconds (acoustic step)
    pub fn new(max_window_duration: f64, dt: f64) -> KwaversResult<Self> {
        if max_window_duration <= 0.0 || dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Window duration and time step must be positive".into(),
            ));
        }

        if dt > max_window_duration {
            return Err(KwaversError::InvalidInput(
                "Time step cannot exceed window duration".into(),
            ));
        }

        Ok(Self {
            max_window_duration,
            dt,
            intensity_history: Vec::new(),
            current_metrics: TemporalIntensityMetrics::default(),
            thermal_dose: ThermalDose::default(),
            peak_intensity: 0.0,
            current_time: 0.0,
        })
    }

    /// Record instantaneous intensity measurement
    ///
    /// # Arguments
    ///
    /// - `pressure_field`: 3D pressure field (Pa)
    /// - `impedance_field`: 3D acoustic impedance (kg/(m²·s))
    /// - `timestamp`: Measurement time (seconds)
    ///
    /// # Returns
    ///
    /// Updated temporal metrics after adding this measurement
    pub fn record_intensity(
        &mut self,
        pressure_field: &Array3<f64>,
        impedance_field: &Array3<f64>,
        timestamp: f64,
    ) -> KwaversResult<TemporalIntensityMetrics> {
        // Compute intensity field: I = p²/Z
        let intensity_field = pressure_field.mapv(|p| p * p) / impedance_field;

        // Find spatial peak (maximum intensity)
        let spatial_peak = intensity_field.iter().cloned().fold(0.0_f64, f64::max);

        // Compute spatial averages
        let total_intensity: f64 = intensity_field.iter().sum();
        let spatial_average = total_intensity / intensity_field.len() as f64;

        // Create measurement
        let measurement = InstantaneousIntensity {
            isppa: spatial_peak,
            spatial_peak,
            spatial_average,
            timestamp,
        };

        // Update peak tracking
        self.peak_intensity = self.peak_intensity.max(spatial_peak);
        self.current_time = timestamp;

        // Add to history
        self.intensity_history.push(measurement);

        // Trim history to window duration
        self.trim_history();

        // Update temporal metrics
        self.update_temporal_metrics();

        Ok(self.current_metrics)
    }

    /// Update temporal averaging metrics
    fn update_temporal_metrics(&mut self) {
        if self.intensity_history.is_empty() {
            self.current_metrics = TemporalIntensityMetrics::default();
            return;
        }

        // Get window bounds
        let window_start = self.current_time - self.max_window_duration;

        // Find measurements within window
        let window_measurements: Vec<_> = self
            .intensity_history
            .iter()
            .filter(|m| m.timestamp >= window_start)
            .collect();

        if window_measurements.is_empty() {
            self.current_metrics = TemporalIntensityMetrics::default();
            return;
        }

        // Compute SPTA (Spatial Peak Temporal Average)
        let total_spta: f64 = window_measurements.iter().map(|m| m.isppa).sum();
        let spta = total_spta / self.max_window_duration;

        // Compute TAS (Temporal Average Spatial)
        let total_tas: f64 = window_measurements.iter().map(|m| m.spatial_average).sum();
        let tas = total_tas / self.max_window_duration;

        // Track peak SPTA
        let peak_spta = window_measurements
            .iter()
            .map(|m| m.isppa)
            .fold(0.0_f64, f64::max);

        let min_spta = window_measurements
            .iter()
            .map(|m| m.isppa)
            .fold(f64::MAX, f64::min);

        self.current_metrics = TemporalIntensityMetrics {
            spta,
            tas,
            peak_spta,
            min_spta: min_spta.min(self.current_metrics.min_spta),
            sample_count: window_measurements.len(),
        };
    }

    /// Trim history to window duration
    fn trim_history(&mut self) {
        let cutoff = self.current_time - self.max_window_duration * 2.0; // Keep 2x window for safety
        self.intensity_history.retain(|m| m.timestamp > cutoff);
    }

    /// Update thermal dose accumulation
    ///
    /// # Arguments
    ///
    /// - `temperature_field`: 3D temperature field (°C)
    /// - `dt`: Time step duration (seconds)
    pub fn update_thermal_dose(
        &mut self,
        temperature_field: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Find maximum temperature in field
        let max_temp = temperature_field
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if !max_temp.is_finite() {
            return Err(KwaversError::InvalidInput(
                "Temperature contains NaN/Inf".into(),
            ));
        }

        // Update max temperature tracking
        self.thermal_dose.max_temperature = self.thermal_dose.max_temperature.max(max_temp);
        self.thermal_dose.current_temperature = max_temp;
        self.thermal_dose.temperature_rise = (max_temp - 37.0).max(0.0);

        // CEM43 accumulation: rate = R^(43-T)
        // where R = 0.25 for T > 43°C, R = 0.5 for T ≤ 43°C
        if max_temp > 37.0 {
            let r = if max_temp > 43.0 { 0.25 } else { 0.5 };
            let exponent = r * (43.0 - max_temp);
            let rate = exponent.exp();
            self.thermal_dose.cem43 += rate * (dt / 60.0); // Convert to minutes
        }

        Ok(())
    }

    /// Get current intensity metrics
    pub fn metrics(&self) -> TemporalIntensityMetrics {
        self.current_metrics
    }

    /// Get current thermal dose
    pub fn thermal_dose(&self) -> ThermalDose {
        self.thermal_dose
    }

    /// Get SPTA in clinically relevant units (W/cm²)
    pub fn spta_w_cm2(&self) -> f64 {
        self.current_metrics.spta / 1e4
    }

    /// Get peak intensity ever recorded (W/cm²)
    pub fn peak_intensity_w_cm2(&self) -> f64 {
        self.peak_intensity / 1e4
    }

    /// Check if thermal safety threshold exceeded
    /// CEM43 should remain < 240 minutes for safe treatment
    pub fn is_thermal_safe(&self) -> bool {
        self.thermal_dose.cem43 < 240.0
    }

    /// Get number of measurements in current window
    pub fn sample_count(&self) -> usize {
        self.current_metrics.sample_count
    }

    /// Reset all tracking
    pub fn reset(&mut self) {
        self.intensity_history.clear();
        self.current_metrics = TemporalIntensityMetrics::default();
        self.thermal_dose = ThermalDose::default();
        self.peak_intensity = 0.0;
        self.current_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_creation() {
        let tracker = IntensityTracker::new(0.1, 1e-6).unwrap();
        assert_eq!(tracker.sample_count(), 0);
        assert_eq!(tracker.peak_intensity_w_cm2(), 0.0);
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(IntensityTracker::new(-0.1, 1e-6).is_err());
        assert!(IntensityTracker::new(0.1, -1e-6).is_err());
        assert!(IntensityTracker::new(1e-7, 1e-6).is_err()); // dt > window
    }

    #[test]
    fn test_intensity_recording() {
        let mut tracker = IntensityTracker::new(0.01, 1e-6).unwrap();

        // Create simple test fields
        let pressure = Array3::from_elem([8, 8, 8], 1e6); // 1 MPa
        let impedance = Array3::from_elem([8, 8, 8], 1.5e6); // Water impedance

        let metrics = tracker
            .record_intensity(&pressure, &impedance, 0.0)
            .unwrap();

        // I = p²/Z = (1e6)² / 1.5e6 ≈ 666.7 kW/m²
        assert!(metrics.spta > 600.0);
        assert!(metrics.spta < 700.0);
    }

    #[test]
    fn test_peak_tracking() {
        let mut tracker = IntensityTracker::new(0.01, 1e-6).unwrap();

        let pressure = Array3::from_elem([4, 4, 4], 1e6);
        let impedance = Array3::from_elem([4, 4, 4], 1.5e6);

        tracker
            .record_intensity(&pressure, &impedance, 0.0)
            .unwrap();
        let peak1 = tracker.peak_intensity_w_cm2();

        // Increase pressure
        let pressure2 = Array3::from_elem([4, 4, 4], 2e6);
        tracker
            .record_intensity(&pressure2, &impedance, 1e-6)
            .unwrap();
        let peak2 = tracker.peak_intensity_w_cm2();

        assert!(peak2 > peak1);
    }

    #[test]
    fn test_thermal_dose() {
        let mut tracker = IntensityTracker::new(0.01, 1e-6).unwrap();
        let pressure = Array3::from_elem([4, 4, 4], 0.0);
        let impedance = Array3::from_elem([4, 4, 4], 1.5e6);

        // Record baseline
        tracker
            .record_intensity(&pressure, &impedance, 0.0)
            .unwrap();

        // Update thermal dose at elevated temperature
        let temperature = Array3::from_elem([4, 4, 4], 45.0); // 45°C
        tracker.update_thermal_dose(&temperature, 1.0).unwrap(); // 1 second

        let dose = tracker.thermal_dose();
        assert!(dose.cem43 > 0.0);
        assert_eq!(dose.current_temperature, 45.0);
        assert!(tracker.is_thermal_safe());
    }

    #[test]
    fn test_spta_units() {
        let mut tracker = IntensityTracker::new(0.01, 1e-6).unwrap();

        let pressure = Array3::from_elem([4, 4, 4], 1e6); // 1 MPa
        let impedance = Array3::from_elem([4, 4, 4], 1.5e6);

        tracker
            .record_intensity(&pressure, &impedance, 0.0)
            .unwrap();

        let spta_wm2 = tracker.metrics().spta;
        let spta_wcm2 = tracker.spta_w_cm2();

        // 1 W/cm² = 1e4 W/m²
        assert!((spta_wcm2 - spta_wm2 / 1e4).abs() < 0.01);
    }

    #[test]
    fn test_reset() {
        let mut tracker = IntensityTracker::new(0.01, 1e-6).unwrap();

        let pressure = Array3::from_elem([4, 4, 4], 1e6);
        let impedance = Array3::from_elem([4, 4, 4], 1.5e6);

        tracker
            .record_intensity(&pressure, &impedance, 0.0)
            .unwrap();
        assert!(tracker.sample_count() > 0);

        tracker.reset();
        assert_eq!(tracker.sample_count(), 0);
        assert_eq!(tracker.peak_intensity_w_cm2(), 0.0);
        assert_eq!(tracker.thermal_dose().cem43, 0.0);
    }
}
