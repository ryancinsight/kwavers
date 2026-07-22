//! IntensityTracker implementation.

use super::types::{InstantaneousIntensity, IntensityTrackerDose, TemporalIntensityMetrics};
use aequitas::systems::si::quantities::{ThermodynamicTemperature, Time};
use asclepius::response::thermal::Cem43;
use kwavers_core::constants::numerical::SECONDS_PER_MINUTE;
use kwavers_core::constants::thermodynamic::{BODY_TEMPERATURE_C, KELVIN_OFFSET_C};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

/// Real-time acoustic intensity tracker
///
/// Maintains continuous history of intensity measurements with rolling
/// temporal averaging windows for FDA compliance and safety monitoring.
#[derive(Debug, Clone)]
pub struct IntensityTracker {
    /// Maximum window size (seconds) for temporal averaging
    max_window_duration: f64,

    /// History of instantaneous intensity measurements
    intensity_history: Vec<InstantaneousIntensity>,

    /// Current temporal average metrics
    current_metrics: TemporalIntensityMetrics,

    /// Thermal dose accumulation
    thermal_dose: IntensityTrackerDose,

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
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
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
            intensity_history: Vec::new(),
            current_metrics: TemporalIntensityMetrics::default(),
            thermal_dose: IntensityTrackerDose::default(),
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn record_intensity(
        &mut self,
        pressure_field: &Array3<f64>,
        impedance_field: &Array3<f64>,
        timestamp: f64,
    ) -> KwaversResult<TemporalIntensityMetrics> {
        // Compute intensity field: I = p²/Z
        let p_squared = pressure_field.mapv(|p| p * p);
        let intensity_field = &p_squared / impedance_field;

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

        // Compute SPTA (Spatial Peak Temporal Average) in W/m².
        //
        // SPTA = (1/T) × ∫ I_sp dt ≈ (Σ I_sp_n × Δt) / T = avg(I_sp_n)
        // (assumes uniform Δt so the per-sample dt cancels with the sum length).
        // Ref: IEC 62359:2010 §5.2, FDA 510(k) Guidance §IV.
        let n = window_measurements.len() as f64;
        let total_spta: f64 = window_measurements.iter().map(|m| m.isppa).sum();
        let spta = total_spta / n;

        // Compute TAS (Temporal Average Spatial) in W/m².
        let total_tas: f64 = window_measurements.iter().map(|m| m.spatial_average).sum();
        let tas = total_tas / n;

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
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

        let increment = Cem43::canonical()
            .increment(
                ThermodynamicTemperature::from_base(max_temp + KELVIN_OFFSET_C),
                Time::from_base(dt),
            )
            .map_err(|source| {
                KwaversError::InvalidInput(format!(
                    "intensity-tracker CEM43 observation is invalid: {source}"
                ))
            })?
            .get()
            .into_base()
            / SECONDS_PER_MINUTE;

        // Commit only after the response observation is fully validated.
        self.thermal_dose.max_temperature = self.thermal_dose.max_temperature.max(max_temp);
        self.thermal_dose.current_temperature = max_temp;
        self.thermal_dose.temperature_rise = (max_temp - BODY_TEMPERATURE_C).max(0.0);

        if max_temp > BODY_TEMPERATURE_C {
            self.thermal_dose.cem43 += increment;
        }

        Ok(())
    }

    /// Get current intensity metrics
    pub fn metrics(&self) -> TemporalIntensityMetrics {
        self.current_metrics
    }

    /// Get current thermal dose
    pub fn thermal_dose(&self) -> IntensityTrackerDose {
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
        self.thermal_dose = IntensityTrackerDose::default();
        self.peak_intensity = 0.0;
        self.current_time = 0.0;
    }
}