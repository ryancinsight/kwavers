//! Cavitation event detection, classification, and frequency estimation.

use aequitas::systems::si::quantities::{Dimensionless, Length, Time};
use aequitas::systems::si::units::Meter;
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
    /// Returns `Err` when `intensity_map.len() != grid_points.shape()[0]`.
    pub fn detect_events(
        &self,
        intensity_map: &Array1<f64>,
        grid_points: &Array2<f64>,
        time: Time<f64>,
    ) -> KwaversResult<Vec<PamCavitationEvent>> {
        Self::validate_detection_inputs(intensity_map, grid_points)?;
        if intensity_map.is_empty() {
            return Ok(Vec::new());
        }

        let threshold = self.noise_threshold(intensity_map);
        let mut events = Vec::new();

        for (idx, &intensity) in intensity_map.iter().enumerate() {
            if intensity > threshold {
                let grid_point = grid_points
                    .index_axis::<1>(0, idx)
                    .expect("invariant: row index within bounds");
                let position = [
                    Length::from_unit::<Meter>(grid_point[0]),
                    Length::from_unit::<Meter>(grid_point[1]),
                    Length::from_unit::<Meter>(grid_point[2]),
                ];
                let coherence =
                    Dimensionless::from_base(self.coherence_factor(intensity, threshold));
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
        time: Time<f64>,
    ) -> KwaversResult<Vec<PamCavitationEvent>> {
        let [num_sensors_data, _] = passive_data.shape();
        if num_sensors_data != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Data has {} sensors but PAM configured for {}",
                num_sensors_data, self.num_sensors
            )));
        }
        Self::validate_detection_inputs(intensity_map, grid_points)?;
        if intensity_map.is_empty() {
            return Ok(Vec::new());
        }

        let threshold = self.noise_threshold(intensity_map);
        let apodization_weights = self.compute_apodization_weights();
        let mut events = Vec::new();

        for (idx, &intensity) in intensity_map.iter().enumerate() {
            if intensity > threshold {
                let grid_point = grid_points
                    .index_axis::<1>(0, idx)
                    .expect("invariant: row index within bounds");
                let position = [
                    Length::from_unit::<Meter>(grid_point[0]),
                    Length::from_unit::<Meter>(grid_point[1]),
                    Length::from_unit::<Meter>(grid_point[2]),
                ];
                let coherence =
                    Dimensionless::from_base(self.coherence_factor(intensity, threshold));
                let delays_samples = self.compute_delays(&position)?;
                let signal = self.beamformed_signal_at_point(
                    passive_data,
                    &delays_samples,
                    &apodization_weights,
                )?;
                let peak_frequency = self.estimate_peak_frequency(&signal);

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

    fn validate_detection_inputs(
        intensity_map: &Array1<f64>,
        grid_points: &Array2<f64>,
    ) -> KwaversResult<()> {
        if intensity_map.len() != grid_points.shape()[0] {
            return Err(KwaversError::InvalidInput(
                "Intensity map and grid points size mismatch".to_owned(),
            ));
        }
        if grid_points.shape()[1] != 3 {
            return Err(KwaversError::InvalidInput(format!(
                "Grid points must have shape [points x 3], got {} columns",
                grid_points.shape()[1]
            )));
        }
        if !intensity_map.iter().all(|value| value.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Intensity map must contain only finite values".to_owned(),
            ));
        }
        if !grid_points.iter().all(|value| value.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "Grid points must contain only finite coordinates".to_owned(),
            ));
        }
        Ok(())
    }
}
