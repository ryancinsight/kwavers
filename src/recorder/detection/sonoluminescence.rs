//! Sonoluminescence detection

use crate::physics::sonoluminescence_detector::DetectorConfig;
use ndarray::Array3;

/// Sonoluminescence event
#[derive(Debug, Clone)]
pub struct SonoluminescenceEvent {
    pub location: (usize, usize, usize),
    pub intensity: f64,
    pub temperature: f64,
    pub time_step: usize,
    pub duration: f64,
}

/// Sonoluminescence detector
#[derive(Debug)]
pub struct SonoluminescenceDetector {
    config: DetectorConfig,
    events: Vec<SonoluminescenceEvent>,
}

impl SonoluminescenceDetector {
    /// Create detector
    pub fn create(config: DetectorConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
        }
    }

    /// Detect events in light and temperature fields
    pub fn detect(
        &mut self,
        light: &Array3<f64>,
        temperature: &Array3<f64>,
        time_step: usize,
        dt: f64,
    ) {
        for ((i, j, k), &intensity) in light.indexed_iter() {
            // Use a default intensity threshold or derive from temperature
            let intensity_threshold = 1e-6; // W/m^2 - typical SL detection threshold
            if intensity > intensity_threshold {
                let temp = temperature[[i, j, k]];
                if temp > self.config.temperature_threshold {
                    self.events.push(SonoluminescenceEvent {
                        location: (i, j, k),
                        intensity,
                        temperature: temp,
                        time_step,
                        duration: dt,
                    });
                }
            }
        }
    }

    /// Get detected events
    pub fn events(&self) -> &[SonoluminescenceEvent] {
        &self.events
    }

    /// Clear events
    pub fn clear(&mut self) {
        self.events.clear();
    }
}
