//! Sonoluminescence Detection and Analysis
//!
//! This module provides complete detection and analysis of sonoluminescence events
//! based on physical criteria from established literature.
//!
//! References:
//! - Brenner et al. (2002) "Single-bubble sonoluminescence"
//! - Yasui (1997) "Alternative model of single-bubble sonoluminescence"
//! - Gaitan et al. (1992) "Sonoluminescence and bubble dynamics"

use crate::{
    physics::bubble_dynamics::BubbleStateFields,
    physics::optics::sonoluminescence::{SonoluminescenceEmission, EmissionParameters},
};
use ndarray::Array3;
use std::collections::HashMap;

/// Physical constants for sonoluminescence
pub mod constants {
    /// Minimum temperature for sonoluminescence (K)
    /// Based on Brenner et al. (2002)
    pub const MIN_TEMPERATURE_SL: f64 = 5000.0;
    
    /// Maximum compression ratio for SL detection
    /// Based on Yasui (1997)
    pub const MAX_COMPRESSION_RATIO: f64 = 10.0;
    
    /// Minimum pressure for SL (Pa)
    /// Based on experimental observations
    pub const MIN_PRESSURE_SL: f64 = 1e6;
    
    /// Time window for event detection (s)
    pub const EVENT_TIME_WINDOW: f64 = 1e-9;
    
    /// Minimum photon count for detection
    pub const MIN_PHOTON_COUNT: f64 = 1e3;
}

/// Sonoluminescence event data
#[derive(Debug, Clone)]
pub struct SonoluminescenceEvent {
    /// Time of event (s)
    pub time: f64,
    /// Position indices (i, j, k)
    pub position: (usize, usize, usize),
    /// Physical position (x, y, z) in meters
    pub physical_position: (f64, f64, f64),
    /// Peak temperature (K)
    pub peak_temperature: f64,
    /// Peak pressure (Pa)
    pub peak_pressure: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Total photon count
    pub photon_count: f64,
    /// Peak wavelength (m)
    pub peak_wavelength: f64,
    /// Event duration (s)
    pub duration: f64,
    /// Energy released (J)
    pub energy: f64,
}

/// Sonoluminescence detector configuration
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Enable spectral analysis
    pub spectral_analysis: bool,
    /// Enable time-resolved detection
    pub time_resolved: bool,
    /// Temperature threshold (K)
    pub temperature_threshold: f64,
    /// Pressure threshold (Pa)
    pub pressure_threshold: f64,
    /// Compression ratio threshold
    pub compression_threshold: f64,
    /// Spatial resolution for clustering (m)
    pub spatial_resolution: f64,
    /// Time resolution for clustering (s)
    pub time_resolution: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            spectral_analysis: true,
            time_resolved: true,
            temperature_threshold: constants::MIN_TEMPERATURE_SL,
            pressure_threshold: constants::MIN_PRESSURE_SL,
            compression_threshold: constants::MAX_COMPRESSION_RATIO,
            spatial_resolution: 1e-6,  // 1 μm
            time_resolution: 1e-10,     // 100 ps
        }
    }
}

/// Sonoluminescence detector and analyzer
#[derive(Debug)]
pub struct SonoluminescenceDetector {
    /// Detector configuration
    config: DetectorConfig,
    /// Emission calculator
    emission_calculator: SonoluminescenceEmission,
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
    pub fn new(
        grid_shape: (usize, usize, usize),
        grid_spacing: (f64, f64, f64),
        config: DetectorConfig,
    ) -> Self {
        let emission_params = EmissionParameters {
            use_blackbody: true,
            use_bremsstrahlung: true,
            use_molecular_lines: false,
            min_temperature: config.temperature_threshold,
            ..Default::default()
        };
        
        Self {
            config,
            emission_calculator: SonoluminescenceEmission::new(grid_shape, emission_params),
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
        pressure_field: &Array3<f64>,
        initial_radius: &Array3<f64>,
        dt: f64,
    ) -> Vec<SonoluminescenceEvent> {
        self.current_time += dt;
        let mut new_events = Vec::new();
        
        let (nx, ny, nz) = bubble_states.radius.dim();
        
        // Scan all grid points for potential events
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Get state at this point
                    let temperature = bubble_states.temperature[[i, j, k]];
                    let pressure = bubble_states.pressure[[i, j, k]];
                    let radius = bubble_states.radius[[i, j, k]];
                    let radius_0 = initial_radius[[i, j, k]];
                    
                    // Skip if no bubble present
                    if radius_0 == 0.0 {
                        continue;
                    }
                    
                    // Calculate compression ratio
                    let compression_ratio = if radius > 0.0 {
                        radius_0 / radius
                    } else {
                        0.0
                    };
                    
                    // Check sonoluminescence criteria
                    if self.check_sl_criteria(temperature, pressure, compression_ratio) {
                        // Check if this is a distinct event or continuation
                        let position = (i, j, k);
                        let is_distinct_event = self.is_distinct_event(position, temperature);
                        
                        if is_distinct_event {
                            // Calculate emission characteristics
                            let (photon_count, peak_wavelength, energy) = 
                                self.calculate_emission_characteristics(
                                    temperature,
                                    pressure,
                                    radius,
                                    dt
                                );
                            
                            // Only record if above detection threshold
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
                                    duration: dt,  // Will be refined with time-resolved analysis
                                    energy,
                                };
                                
                                new_events.push(event.clone());
                                self.events.push(event);
                            }
                        }
                        
                        // Update event history
                        self.update_event_history(position, temperature);
                    }
                }
            }
        }
        
        // Perform clustering to group nearby events
        if self.config.time_resolved {
            self.cluster_events(&mut new_events);
        }
        
        new_events
    }
    
    /// Check if sonoluminescence criteria are met
    fn check_sl_criteria(&self, temperature: f64, pressure: f64, compression_ratio: f64) -> bool {
        temperature >= self.config.temperature_threshold &&
        pressure >= self.config.pressure_threshold &&
        compression_ratio >= self.config.compression_threshold
    }
    
    /// Check if this is a distinct event or continuation of existing
    fn is_distinct_event(&self, position: (usize, usize, usize), temperature: f64) -> bool {
        match self.event_history.get(&position) {
            Some(history) => {
                // Check if temperature has dropped and risen again
                if let Some(&last_temp) = history.last() {
                    temperature > last_temp * 1.5  // Significant increase
                } else {
                    true
                }
            }
            None => true,
        }
    }
    
    /// Calculate emission characteristics
    fn calculate_emission_characteristics(
        &mut self,
        temperature: f64,
        pressure: f64,
        radius: f64,
        dt: f64,
    ) -> (f64, f64, f64) {
        // Stefan-Boltzmann law for total power
        let sigma = 5.67e-8;  // Stefan-Boltzmann constant
        let surface_area = 4.0 * std::f64::consts::PI * radius.powi(2);
        let power = sigma * surface_area * temperature.powi(4);
        
        // Total energy emitted
        let energy = power * dt;
        
        // Photon count (assuming average photon energy at peak wavelength)
        let peak_wavelength = 2.898e-3 / temperature;  // Wien's law
        let h = 6.626e-34;  // Planck constant
        let c = 3e8;  // Speed of light
        let photon_energy = h * c / peak_wavelength;
        let photon_count = energy / photon_energy;
        
        (photon_count, peak_wavelength, energy)
    }
    
    /// Update event history for a position
    fn update_event_history(&mut self, position: (usize, usize, usize), temperature: f64) {
        self.event_history
            .entry(position)
            .or_insert_with(Vec::new)
            .push(temperature);
        
        // Keep only recent history
        if let Some(history) = self.event_history.get_mut(&position) {
            if history.len() > 100 {
                history.drain(0..50);
            }
        }
    }
    
    /// Cluster nearby events in space and time
    fn cluster_events(&self, events: &mut Vec<SonoluminescenceEvent>) {
        // Group events that are close in space and time
        let mut clustered = Vec::new();
        let mut processed = vec![false; events.len()];
        
        for i in 0..events.len() {
            if processed[i] {
                continue;
            }
            
            let mut cluster = vec![events[i].clone()];
            processed[i] = true;
            
            for j in i+1..events.len() {
                if processed[j] {
                    continue;
                }
                
                // Check spatial proximity
                let dx = events[i].physical_position.0 - events[j].physical_position.0;
                let dy = events[i].physical_position.1 - events[j].physical_position.1;
                let dz = events[i].physical_position.2 - events[j].physical_position.2;
                let spatial_distance = (dx*dx + dy*dy + dz*dz).sqrt();
                
                // Check temporal proximity
                let time_difference = (events[i].time - events[j].time).abs();
                
                if spatial_distance <= self.config.spatial_resolution &&
                   time_difference <= self.config.time_resolution {
                    cluster.push(events[j].clone());
                    processed[j] = true;
                }
            }
            
            // Merge cluster into single event
            if cluster.len() > 1 {
                let merged = self.merge_cluster(cluster);
                clustered.push(merged);
            } else {
                clustered.push(events[i].clone());
            }
        }
        
        *events = clustered;
    }
    
    /// Merge a cluster of events into a single event
    fn merge_cluster(&self, cluster: Vec<SonoluminescenceEvent>) -> SonoluminescenceEvent {
        let n = cluster.len() as f64;
        
        // Average properties
        let total_photons = cluster.iter().map(|e| e.photon_count).sum();
        let total_energy = cluster.iter().map(|e| e.energy).sum();
        
        // Find peak values
        let peak_temp = cluster.iter().map(|e| e.peak_temperature).fold(0.0, f64::max);
        let peak_pressure = cluster.iter().map(|e| e.peak_pressure).fold(0.0, f64::max);
        
        // Use centroid for position
        let x = cluster.iter().map(|e| e.physical_position.0).sum::<f64>() / n;
        let y = cluster.iter().map(|e| e.physical_position.1).sum::<f64>() / n;
        let z = cluster.iter().map(|e| e.physical_position.2).sum::<f64>() / n;
        
        // Use earliest time
        let time = cluster.iter().map(|e| e.time).fold(f64::INFINITY, f64::min);
        
        // Duration spans the cluster
        let end_time = cluster.iter().map(|e| e.time + e.duration).fold(0.0, f64::max);
        let duration = end_time - time;
        
        SonoluminescenceEvent {
            time,
            position: cluster[0].position,  // Use first event's grid position
            physical_position: (x, y, z),
            peak_temperature: peak_temp,
            peak_pressure: peak_pressure,
            compression_ratio: cluster[0].compression_ratio,  // Use first
            photon_count: total_photons,
            peak_wavelength: 2.898e-3 / peak_temp,  // Wien's law with peak temp
            duration,
            energy: total_energy,
        }
    }
    
    /// Get all detected events
    pub fn get_events(&self) -> &[SonoluminescenceEvent] {
        &self.events
    }
    
    /// Get events in time window
    pub fn get_events_in_window(&self, start_time: f64, end_time: f64) -> Vec<&SonoluminescenceEvent> {
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
        
        let max_temperature = self.events
            .iter()
            .map(|e| e.peak_temperature)
            .fold(0.0, f64::max);
        
        let avg_temperature = self.events
            .iter()
            .map(|e| e.peak_temperature)
            .sum::<f64>() / total_events as f64;
        
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

/// Statistics about sonoluminescence events
#[derive(Debug, Default, Clone)]
pub struct SonoluminescenceStatistics {
    /// Total number of events detected
    pub total_events: usize,
    /// Total photon count
    pub total_photons: f64,
    /// Total energy released (J)
    pub total_energy: f64,
    /// Maximum temperature observed (K)
    pub max_temperature: f64,
    /// Average temperature of events (K)
    pub avg_temperature: f64,
    /// Event rate (events/s)
    pub event_rate: f64,
}