// sensor/mod.rs

pub mod beamforming;
pub mod localization;
pub(crate) mod math;
pub mod passive_acoustic_mapping; // NEW: Multi-lateration localization system

pub use localization::{
    array::Sensor as LocalizationSensor, ArrayGeometry as LocalizationArrayGeometry,
    LocalizationResult, SensorArray,
};
// Re-export PAM components without colliding names; configs remain module-scoped.
pub use passive_acoustic_mapping::{ArrayGeometry, PAMConfig, PAMPlugin, PassiveAcousticMapper};
// Expose unified beamforming config at the sensor module root.
pub use beamforming::{BeamformingConfig, BeamformingCoreConfig};

use crate::grid::Grid;
use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Configuration for sensor setup
#[derive(Debug, Clone)]
pub struct SensorConfig {
    pub positions: Vec<(f64, f64, f64)>, // Positions in meters
    pub record_pressure: bool,
    pub record_light: bool,
}

impl SensorConfig {
    #[must_use]
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            record_pressure: true,
            record_light: true,
        }
    }

    #[must_use]
    pub fn with_positions(mut self, positions: Vec<(f64, f64, f64)>) -> Self {
        self.positions = positions;
        self
    }

    #[must_use]
    pub fn with_pressure_recording(mut self, record: bool) -> Self {
        self.record_pressure = record;
        self
    }

    #[must_use]
    pub fn with_light_recording(mut self, record: bool) -> Self {
        self.record_light = record;
        self
    }
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct Sensor {
    pub positions: Vec<(usize, usize, usize)>, // Grid indices
    pub pressure_data: Array2<f64>,            // Pressure time series
    pub light_data: Array2<f64>,               // Light fluence rate time series
    grid: Grid,
}

impl Sensor {
    /// Get number of sensors
    pub fn num_sensors(&self) -> usize {
        self.positions.len()
    }

    /// Sample field at sensor positions
    pub fn sample(&self, field: &Array3<f64>) -> Vec<f64> {
        let mut data = Vec::with_capacity(self.positions.len());
        for &(i, j, k) in &self.positions {
            if i < self.grid.nx
                && j < self.grid.ny
                && k < self.grid.nz
                && i < field.shape()[0]
                && j < field.shape()[1]
                && k < field.shape()[2]
            {
                data.push(field[[i, j, k]]);
            } else {
                data.push(0.0);
            }
        }
        data
    }
}

/// Sensor data collection for time-reversal reconstruction
#[derive(Debug, Clone)]
pub struct SensorData {
    /// Map from sensor ID to recorded data
    data: HashMap<usize, Vec<f64>>,
    /// Sensor configurations
    sensors: Vec<SensorInfo>,
}

/// Information about a single sensor
#[derive(Debug, Clone)]
pub struct SensorInfo {
    /// Sensor ID
    id: usize,
    /// Position in grid coordinates
    position: [usize; 3],
    /// Position in physical coordinates (meters)
    #[allow(dead_code)]
    position_meters: [f64; 3],
}

impl SensorInfo {
    #[must_use]
    pub fn position(&self) -> &[usize; 3] {
        &self.position
    }
}

impl SensorData {
    /// Create new sensor data collection
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            sensors: Vec::new(),
        }
    }

    /// Check if sensor data is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get sensor configurations
    #[must_use]
    pub fn sensors(&self) -> &[SensorInfo] {
        &self.sensors
    }

    /// Get data iterator
    pub fn data_iter(&self) -> impl Iterator<Item = (&usize, &Vec<f64>)> {
        self.data.iter()
    }

    /// Get data for a specific sensor
    #[must_use]
    pub fn get_data(&self, sensor_id: usize) -> Option<&Vec<f64>> {
        self.data.get(&sensor_id)
    }

    /// Add sensor data
    pub fn add_sensor_data(
        &mut self,
        sensor_id: usize,
        position: [usize; 3],
        position_meters: [f64; 3],
        data: Vec<f64>,
    ) {
        self.data.insert(sensor_id, data);
        self.sensors.push(SensorInfo {
            id: sensor_id,
            position,
            position_meters,
        });
    }
}

impl Default for SensorData {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator adapter for sensor data processing
#[derive(Debug)]
pub struct SensorDataIterator<'a> {
    data: &'a SensorData,
    sensors: Vec<&'a SensorInfo>,
    current: usize,
}

impl<'a> Iterator for SensorDataIterator<'a> {
    type Item = (&'a SensorInfo, &'a Vec<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.sensors.len() {
            return None;
        }

        let sensor = self.sensors[self.current];
        let data = self.data.data.get(&sensor.id)?;
        self.current += 1;

        Some((sensor, data))
    }
}

/// Sensor data processing methods
impl SensorData {
    /// Create an iterator over sensor data
    #[must_use]
    pub fn iter(&self) -> SensorDataIterator<'_> {
        let sensors: Vec<_> = self.sensors.iter().collect();
        SensorDataIterator {
            data: self,
            sensors,
            current: 0,
        }
    }

    /// Apply a transformation to all sensor data
    pub fn map_data<F>(&self, f: F) -> HashMap<usize, Vec<f64>>
    where
        F: Fn(&Vec<f64>) -> Vec<f64>,
    {
        self.data.iter().map(|(id, data)| (*id, f(data))).collect()
    }
}
