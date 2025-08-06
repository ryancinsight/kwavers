// sensor/mod.rs

use crate::grid::Grid;
use crate::time::Time;
use log::{debug, error, info};
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

/// Configuration for sensor setup
#[derive(Debug, Clone)]
pub struct SensorConfig {
    pub positions: Vec<(f64, f64, f64)>, // Positions in meters
    pub record_pressure: bool,
    pub record_light: bool,
}

impl SensorConfig {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            record_pressure: true,
            record_light: true,
        }
    }
    
    pub fn with_positions(mut self, positions: Vec<(f64, f64, f64)>) -> Self {
        self.positions = positions;
        self
    }
    
    pub fn with_pressure_recording(mut self, record: bool) -> Self {
        self.record_pressure = record;
        self
    }
    
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
    positions: Vec<(usize, usize, usize)>, // Grid indices
    pressure_data: Array2<f64>,            // Pressure time series
    light_data: Array2<f64>,               // Light fluence rate time series
    grid: Grid,
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
    position_meters: [f64; 3],
}

impl SensorInfo {
    pub fn position(&self) -> &[usize; 3] {
        &self.position
    }
}

impl SensorData {
    /// Create new sensor data collection
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            sensors: Vec::new(),
        }
    }
    
    /// Check if sensor data is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get sensor configurations
    pub fn sensors(&self) -> &[SensorInfo] {
        &self.sensors
    }
    
    /// Get data iterator
    pub fn data_iter(&self) -> impl Iterator<Item = (&usize, &Vec<f64>)> {
        self.data.iter()
    }
    
    /// Add sensor data
    pub fn add_sensor_data(&mut self, sensor_id: usize, position: [usize; 3], position_meters: [f64; 3], data: Vec<f64>) {
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

/// Advanced sensor data processing methods
impl SensorData {
    /// Create an iterator over sensor data
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
        self.data.iter()
            .map(|(id, data)| (*id, f(data)))
            .collect()
    }
    
    /// Filter sensors based on a predicate
    pub fn filter_sensors<F>(&self, predicate: F) -> Vec<&SensorInfo>
    where
        F: Fn(&SensorInfo) -> bool,
    {
        self.sensors.iter()
            .filter(|sensor| predicate(sensor))
            .collect()
    }
    
    /// Compute statistics for each sensor using iterator combinators
    pub fn compute_statistics(&self) -> HashMap<usize, SensorStatistics> {
        self.data.iter()
            .map(|(id, data)| {
                let stats = data.iter()
                    .fold(SensorStatisticsAccumulator::new(), |acc, &val| {
                        acc.update(val)
                    })
                    .finalize(data.len());
                (*id, stats)
            })
            .collect()
    }
    
    /// Find sensors with data exceeding a threshold
    pub fn sensors_exceeding_threshold(&self, threshold: f64) -> Vec<usize> {
        self.data.iter()
            .filter_map(|(id, data)| {
                data.iter()
                    .any(|&val| val > threshold)
                    .then_some(*id)
            })
            .collect()
    }
    
    /// Apply windowed processing to sensor data
    pub fn windowed_processing<F, T>(&self, window_size: usize, f: F) -> HashMap<usize, Vec<T>>
    where
        F: Fn(&[f64]) -> T,
    {
        self.data.iter()
            .map(|(id, data)| {
                let processed: Vec<T> = data.windows(window_size)
                    .map(&f)
                    .collect();
                (*id, processed)
            })
            .collect()
    }
    
    /// Parallel processing of sensor data using rayon
    pub fn par_process<F, T>(&self, f: F) -> HashMap<usize, T>
    where
        F: Fn(&Vec<f64>) -> T + Sync + Send,
        T: Send,
    {
        use rayon::prelude::*;
        
        self.data.par_iter()
            .map(|(id, data)| (*id, f(data)))
            .collect()
    }
    
    /// Combine data from multiple sensors using a reduction operation
    pub fn reduce_all<F, T>(&self, init: T, f: F) -> T
    where
        F: Fn(T, &Vec<f64>) -> T,
    {
        self.data.values()
            .fold(init, f)
    }
    
    /// Create a lazy iterator that applies transformations on demand
    pub fn lazy_transform<'a, F, T>(&'a self, f: F) -> impl Iterator<Item = (usize, T)> + 'a
    where
        F: Fn(&Vec<f64>) -> T + 'a,
        T: 'a,
    {
        self.data.iter()
            .map(move |(id, data)| (*id, f(data)))
    }
}

#[derive(Debug, Clone)]
pub struct SensorStatistics {
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub peak_to_peak: f64,
}

struct SensorStatisticsAccumulator {
    sum: f64,
    sum_sq: f64,
    min: f64,
    max: f64,
}

impl SensorStatisticsAccumulator {
    fn new() -> Self {
        Self {
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
    
    fn update(self, value: f64) -> Self {
        Self {
            sum: self.sum + value,
            sum_sq: self.sum_sq + value * value,
            min: self.min.min(value),
            max: self.max.max(value),
        }
    }
    
    fn finalize(self, count: usize) -> SensorStatistics {
        let mean = self.sum / count as f64;
        let variance = (self.sum_sq / count as f64) - mean * mean;
        
        SensorStatistics {
            mean,
            variance,
            std_dev: variance.sqrt(),
            min: self.min,
            max: self.max,
            peak_to_peak: self.max - self.min,
        }
    }
}



impl Sensor {
    /// Creates a new sensor with positions in meters, converted to grid indices.
    pub fn new(grid: &Grid, time: &Time, positions_meters: &[(f64, f64, f64)]) -> Self {
        let positions: Vec<(usize, usize, usize)> = positions_meters
            .iter()
            .filter_map(|&(x, y, z)| grid.to_grid_indices(x, y, z))
            .collect();

        if positions.is_empty() {
            error!("No valid sensor positions provided within grid bounds");
        }
        assert!(
            !positions.is_empty(),
            "At least one valid sensor position required"
        );

        let pressure_data = Array2::zeros((positions.len(), time.n_steps));
        let light_data = Array2::zeros((positions.len(), time.n_steps));
        info!(
            "Sensor initialized with {} positions for {} steps",
            positions.len(),
            time.n_steps
        );
        debug!(
            "Sensor data preallocated: {} sensors x {} steps",
            positions.len(),
            time.n_steps
        );

        Self {
            positions,
            pressure_data,
            light_data,
            grid: grid.clone(),
        }
    }
    
    /// Creates a single point sensor at the specified coordinates
    pub fn point(x: f64, y: f64, z: f64) -> SensorConfig {
        SensorConfig::new().with_positions(vec![(x, y, z)])
    }
    
    /// Creates a sensor from configuration
    pub fn from_config(grid: &Grid, time: &Time, config: &SensorConfig) -> Self {
        Self::new(grid, time, &config.positions)
    }

    /// Records pressure and light data at the current time step from k-space fields.
    pub fn record(
        &mut self,
        pressure_field: &Array3<f64>,
        light_field: &Array3<f64>,
        time_step: usize,
    ) {
        debug!("Recording sensor data at step {}", time_step);
        for (i, &(ix, iy, iz)) in self.positions.iter().enumerate() {
            self.pressure_data[[i, time_step]] = pressure_field[[ix, iy, iz]];
            self.light_data[[i, time_step]] = light_field[[ix, iy, iz]];
        }
    }

    // Accessors
    pub fn num_sensors(&self) -> usize {
        self.positions.len()
    }
    pub fn positions(&self) -> &[(usize, usize, usize)] {
        &self.positions
    }
    pub fn pressure_data(&self) -> &Array2<f64> {
        &self.pressure_data
    }
    pub fn light_data(&self) -> &Array2<f64> {
        &self.light_data
    }
    pub fn get_pressure_time_series(&self, sensor_idx: usize) -> Option<Array1<f64>> {
        if sensor_idx < self.num_sensors() {
            Some(self.pressure_data.row(sensor_idx).to_owned())
        } else {
            None
        }
    }
    pub fn get_light_time_series(&self, sensor_idx: usize) -> Option<Array1<f64>> {
        if sensor_idx < self.num_sensors() {
            Some(self.light_data.row(sensor_idx).to_owned())
        } else {
            None
        }
    }
    pub fn grid_dx(&self) -> f64 {
        self.grid.dx
    }
    pub fn grid_dy(&self) -> f64 {
        self.grid.dy
    }
    pub fn grid_dz(&self) -> f64 {
        self.grid.dz
    }
}
