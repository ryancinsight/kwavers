//! Cavitation event detection, classification, and frequency estimation.

use leto::Array1;
use leto::Array2;

use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::types::PamCavitationEvent;
use super::DelayAndSumPAM;

impl DelayAndSumPAM {
    /// Detect cavitation events above threshold in a precomputed intensity map.
    ///
    /// Events are sorted in descending intensity order.
    ///
    /// # Errors
    /// Returns `Err` when `intensity_map.len() != grid_points.nrows()`.
    pub fn detect_events(
        &self,
        intensity_map: &Array1<f64>,
        grid_points: &Array2<f64>,
        time: f64,
    ) -> KwaversResult<Vec<PamCavitationEvent>> {
        if intensity_map.len() != grid_points.nrows() {
            return Err(KwaversError::InvalidInput(
                "Intensity map and grid points size mismatch".to_owned(),
            ));
        }

        let threshold = self.noise_threshold(intensity_map);
        let mut events = Vec::new();

        for (idx, &intensity) in intensity_map.iter().enumerate() {
            if intensity > threshold {
                let grid_point = grid_points.row(idx);
                let position = [grid_point[0], grid_point[1], grid_point[2]];
                let coherence = self.coherence_factor(intensity, threshold);
                events.push(PamCavitationEvent {
                    position,
                    intensity,
                    time,
                    coherence,
                    peak_frequency: None,
                });
            }
        }

        events.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));

        Ok(events)
    }

    /// Detect events and estimate peak frequency from raw sensor data.
    ///
    /// Extends `detect_events` by backprojecting the beamformed signal at
    /// each detected location and extracting its dominant spectral frequency
    /// via FFT.
    ///
    /// # Errors
    /// Returns `Err` on sensor-count mismatch or size mismatch between map
    /// and grid points.
    pub fn detect_events_with_data(
        &self,
        passive_data: &Array2<f64>,
        intensity_map: &Array1<f64>,
        grid_points: &Array2<f64>,
        time: f64,
    ) -> KwaversResult<Vec<PamCavitationEvent>> {
        let (num_sensors_data, _) = passive_data.dim();
        if num_sensors_data != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Data has {} sensors but PAM configured for {}",
                num_sensors_data, self.num_sensors
            )));
        }
        if intensity_map.len() != grid_points.nrows() {
            return Err(KwaversError::InvalidInput(
                "Intensity map and grid points size mismatch".to_owned(),
            ));
        }

        let threshold = self.noise_threshold(intensity_map);
        let apodization_weights = self.compute_apodization_weights();
        let mut events = Vec::new();

        for (idx, &intensity) in intensity_map.iter().enumerate() {
            if intensity > threshold {
                let grid_point = grid_points.row(idx);
                let position = [grid_point[0], grid_point[1], grid_point[2]];
                let coherence = self.coherence_factor(intensity, threshold);
                let delays_samples = self.compute_delays(&position)?;
                let peak_frequency = self
                    .beamformed_signal_at_point(passive_data, &delays_samples, &apodization_weights)
                    .ok()
                    .and_then(|signal| self.estimate_peak_frequency(&signal));

                events.push(PamCavitationEvent {
                    position,
                    intensity,
                    time,
                    coherence,
                    peak_frequency,
                });
            }
        }

        events.sort_by(|a, b| b.intensity.total_cmp(&a.intensity));

        Ok(events)
    }
}
