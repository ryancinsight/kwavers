// sensor/mod.rs

use crate::grid::Grid;
use crate::time::Time;
use log::{debug, error, info};
use ndarray::{Array1, Array2, Array3};

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
