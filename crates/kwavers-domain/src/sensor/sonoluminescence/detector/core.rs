//! Sonoluminescence detector implementation.

use super::constants;
use super::types::{DetectorConfig, SonoluminescenceEvent, SonoluminescenceStatistics};
use kwavers_core::constants::fundamental::{PLANCK, SPEED_OF_LIGHT, STEFAN_BOLTZMANN};
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::constants::optical::WIEN_CONSTANT;
use crate::field::BubbleStateFields;
use ndarray::Array3;
use std::collections::HashMap;

/// Sonoluminescence detector and analyzer
#[derive(Debug)]
pub struct SonoluminescenceDetector {
    /// Detector configuration
    config: DetectorConfig,
    /// Treat the emitter as an ideal blackbody (efficiency 1.0) vs. grey
    /// (0.5) when converting radiated power to photon count. Domain-local flag;
    /// the detector computes emission inline (Stefan-Boltzmann / Wien / Planck)
    /// rather than depending on the physics::optics emission model.
    use_blackbody: bool,
    /// Detected events
    events: Vec<SonoluminescenceEvent>,
    /// Event history for time-resolved analysis
    event_history: HashMap<(usize, usize, usize), Vec<f64>>,
    /// Grid spacing for position calculation
    grid_spacing: (f64, f64, f64),
    /// Current simulation time
    current_time: f64,
}

impl SonoluminescenceDetector {
    /// Create new sonoluminescence detector
    #[must_use]
    pub fn new(
        grid_shape: (usize, usize, usize),
        grid_spacing: (f64, f64, f64),
        config: DetectorConfig,
    ) -> Self {
        let _ = grid_shape; // retained for API symmetry; inline emission needs no grid
        Self {
            config,
            use_blackbody: true,
            events: Vec::new(),
            event_history: HashMap::new(),
            grid_spacing,
            current_time: 0.0,
        }
    }

    /// Detect sonoluminescence events from bubble state fields
    pub fn detect_events(
        &mut self,
        bubble_states: &BubbleStateFields,
        _pressure_field: &Array3<f64>,
        initial_radius: &Array3<f64>,
        dt: f64,
    ) -> Vec<SonoluminescenceEvent> {
        self.current_time += dt;
        let mut new_events = Vec::new();

        let (nx, ny, nz) = bubble_states.radius.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let temperature = bubble_states.temperature[[i, j, k]];
                    let pressure = bubble_states.pressure[[i, j, k]];
                    let radius = bubble_states.radius[[i, j, k]];
                    let radius_0 = initial_radius[[i, j, k]];

                    if radius_0 == 0.0 {
                        continue;
                    }

                    let compression_ratio = if radius > 0.0 { radius_0 / radius } else { 0.0 };

                    if self.check_sl_criteria(temperature, pressure, compression_ratio) {
                        let position = (i, j, k);
                        let is_distinct_event = self.is_distinct_event(position, temperature);

                        if is_distinct_event {
                            let (photon_count, peak_wavelength, energy) = self
                                .calculate_emission_characteristics(
                                    temperature,
                                    pressure,
                                    radius,
                                    dt,
                                );

                            if photon_count >= constants::MIN_PHOTON_COUNT {
                                let event = SonoluminescenceEvent {
                                    time: self.current_time,
                                    position,
                                    physical_position: (
                                        i as f64 * self.grid_spacing.0,
                                        j as f64 * self.grid_spacing.1,
                                        k as f64 * self.grid_spacing.2,
                                    ),
                                    peak_temperature: temperature,
                                    peak_pressure: pressure,
                                    compression_ratio,
                                    photon_count,
                                    peak_wavelength,
                                    duration: dt,
                                    energy,
                                };

                                new_events.push(event.clone());
                                self.events.push(event);
                            }
                        }

                        self.update_event_history(position, temperature);
                    }
                }
            }
        }

        if self.config.time_resolved {
            self.cluster_events(&mut new_events);
        }

        new_events
    }

    fn check_sl_criteria(&self, temperature: f64, pressure: f64, compression_ratio: f64) -> bool {
        temperature >= self.config.temperature_threshold
            && pressure >= self.config.pressure_threshold
            && compression_ratio >= self.config.compression_threshold
    }

    fn is_distinct_event(&self, position: (usize, usize, usize), temperature: f64) -> bool {
        match self.event_history.get(&position) {
            Some(history) => {
                if let Some(&last_temp) = history.last() {
                    temperature > last_temp * 1.5
                } else {
                    true
                }
            }
            None => true,
        }
    }

    fn calculate_emission_characteristics(
        &mut self,
        temperature: f64,
        _pressure: f64,
        radius: f64,
        dt: f64,
    ) -> (f64, f64, f64) {
        // Stefan–Boltzmann total radiated power: P = σ·A·T⁴ (SSOT σ).
        let surface_area = FOUR_PI * radius.powi(2);
        let power = STEFAN_BOLTZMANN * surface_area * temperature.powi(4);

        let efficiency = if self.use_blackbody { 1.0 } else { 0.5 };
        let energy = power * dt * efficiency;

        // Wien displacement: λ_peak = b/T (SSOT WIEN_CONSTANT, CODATA b ≈ 2.898 mm·K).
        let peak_wavelength = WIEN_CONSTANT / temperature;
        // Photon energy at the spectral peak: E_γ = h·c/λ (SSOT h, c).
        let photon_energy = PLANCK * SPEED_OF_LIGHT / peak_wavelength;
        let photon_count = energy / photon_energy;

        (photon_count, peak_wavelength, energy)
    }

    fn update_event_history(&mut self, position: (usize, usize, usize), temperature: f64) {
        self.event_history
            .entry(position)
            .or_default()
            .push(temperature);

        if let Some(history) = self.event_history.get_mut(&position) {
            if history.len() > 100 {
                history.drain(0..50);
            }
        }
    }

    fn cluster_events(&self, events: &mut Vec<SonoluminescenceEvent>) {
        let mut clustered = Vec::new();
        let mut processed = vec![false; events.len()];

        for i in 0..events.len() {
            if processed[i] {
                continue;
            }

            let mut cluster = vec![events[i].clone()];
            processed[i] = true;

            for j in i + 1..events.len() {
                if processed[j] {
                    continue;
                }

                let dx = events[i].physical_position.0 - events[j].physical_position.0;
                let dy = events[i].physical_position.1 - events[j].physical_position.1;
                let dz = events[i].physical_position.2 - events[j].physical_position.2;
                let spatial_distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
                let time_difference = (events[i].time - events[j].time).abs();

                if spatial_distance <= self.config.spatial_resolution
                    && time_difference <= self.config.time_resolution
                {
                    cluster.push(events[j].clone());
                    processed[j] = true;
                }
            }

            if cluster.len() > 1 {
                let merged = self.merge_cluster(cluster);
                clustered.push(merged);
            } else {
                clustered.push(events[i].clone());
            }
        }

        *events = clustered;
    }

    fn merge_cluster(&self, cluster: Vec<SonoluminescenceEvent>) -> SonoluminescenceEvent {
        let n = cluster.len() as f64;

        let total_photons = cluster.iter().map(|e| e.photon_count).sum();
        let total_energy = cluster.iter().map(|e| e.energy).sum();

        let peak_temperature = cluster
            .iter()
            .map(|e| e.peak_temperature)
            .fold(0.0, f64::max);
        let peak_pressure = cluster.iter().map(|e| e.peak_pressure).fold(0.0, f64::max);

        let x = cluster.iter().map(|e| e.physical_position.0).sum::<f64>() / n;
        let y = cluster.iter().map(|e| e.physical_position.1).sum::<f64>() / n;
        let z = cluster.iter().map(|e| e.physical_position.2).sum::<f64>() / n;

        let time = cluster.iter().map(|e| e.time).fold(f64::INFINITY, f64::min);
        let end_time = cluster
            .iter()
            .map(|e| e.time + e.duration)
            .fold(0.0, f64::max);
        let duration = end_time - time;

        SonoluminescenceEvent {
            time,
            position: cluster[0].position,
            physical_position: (x, y, z),
            peak_temperature,
            peak_pressure,
            compression_ratio: cluster[0].compression_ratio,
            photon_count: total_photons,
            peak_wavelength: WIEN_CONSTANT / peak_temperature,
            duration,
            energy: total_energy,
        }
    }

    /// Get all detected events
    #[must_use]
    pub fn get_events(&self) -> &[SonoluminescenceEvent] {
        &self.events
    }

    /// Get events in time window
    #[must_use]
    pub fn get_events_in_window(
        &self,
        start_time: f64,
        end_time: f64,
    ) -> Vec<&SonoluminescenceEvent> {
        self.events
            .iter()
            .filter(|e| e.time >= start_time && e.time <= end_time)
            .collect()
    }

    /// Clear event history
    pub fn clear_history(&mut self) {
        self.events.clear();
        self.event_history.clear();
    }

    /// Get statistics about detected events
    pub fn get_statistics(&self) -> SonoluminescenceStatistics {
        if self.events.is_empty() {
            return SonoluminescenceStatistics::default();
        }

        let total_events = self.events.len();
        let total_photons: f64 = self.events.iter().map(|e| e.photon_count).sum();
        let total_energy: f64 = self.events.iter().map(|e| e.energy).sum();

        let max_temperature = self
            .events
            .iter()
            .map(|e| e.peak_temperature)
            .fold(0.0, f64::max);

        let avg_temperature =
            self.events.iter().map(|e| e.peak_temperature).sum::<f64>() / total_events as f64;

        let event_rate = if self.current_time > 0.0 {
            total_events as f64 / self.current_time
        } else {
            0.0
        };

        SonoluminescenceStatistics {
            total_events,
            total_photons,
            total_energy,
            max_temperature,
            avg_temperature,
            event_rate,
        }
    }
}
