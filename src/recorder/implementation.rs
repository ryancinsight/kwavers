// recorder/implementation.rs - Main recorder implementation

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::bubble_dynamics::BubbleStateFields;
use crate::physics::field_indices::{BUBBLE_RADIUS_IDX, LIGHT_IDX, PRESSURE_IDX, TEMPERATURE_IDX};
use crate::physics::sonoluminescence_detector::{SonoluminescenceDetector, SonoluminescenceEvent};
use crate::sensor::Sensor;
use crate::time::Time;
use log::info;
use ndarray::{Array2, Array3, Array4, Axis};
use std::fs::File;
use std::io::Write;

use super::config::RecorderConfig;
use super::events::{CavitationEvent, ThermalEvent};
use super::statistics::RecorderStatistics;
use super::traits::RecorderTrait;

/// Main recorder implementation
#[derive(Debug)]
pub struct Recorder {
    pub sensor: Sensor,
    pub filename: String,
    pub record_pressure: bool,
    pub record_light: bool,
    pub record_temperature: bool,
    pub record_cavitation: bool,
    pub record_sonoluminescence: bool,
    pub fields_snapshots: Vec<(usize, Array4<f64>)>,
    pub snapshot_interval: usize,
    pub pressure_sensor_data: Vec<Vec<f64>>,
    pub light_sensor_data: Vec<Vec<f64>>,
    pub temperature_sensor_data: Vec<Vec<f64>>,
    pub recorded_steps: Vec<f64>,
    pub time: Time,

    // Cavitation tracking
    pub cavitation_events: Vec<CavitationEvent>,
    pub cavitation_threshold: f64,
    pub cavitation_map: Array3<f64>,

    // Sonoluminescence tracking
    pub sl_detector: Option<SonoluminescenceDetector>,
    pub sl_events: Vec<SonoluminescenceEvent>,
    pub sl_intensity_map: Array3<f64>,

    // Thermal tracking
    pub thermal_events: Vec<ThermalEvent>,
    pub max_temperature_map: Array3<f64>,
    pub thermal_dose_map: Array3<f64>,

    // Statistics
    pub statistics: RecorderStatistics,
}

impl Recorder {
    /// Get reference to the sensor
    pub fn sensor(&self) -> &Sensor {
        &self.sensor
    }

    /// Get pressure data as Array2 for compatibility
    pub fn pressure_data(&self) -> Option<Array2<f64>> {
        if self.pressure_sensor_data.is_empty() {
            return None;
        }

        let n_time = self.pressure_sensor_data.len();
        let n_sensors = self.pressure_sensor_data[0].len();
        let mut data = Array2::zeros((n_sensors, n_time));

        for (t, sensor_data) in self.pressure_sensor_data.iter().enumerate() {
            for (s, &value) in sensor_data.iter().enumerate() {
                data[[s, t]] = value;
            }
        }

        Some(data)
    }

    /// Get light data as Array2 for compatibility
    pub fn light_data(&self) -> Option<Array2<f64>> {
        if self.light_sensor_data.is_empty() {
            return None;
        }

        let n_time = self.light_sensor_data.len();
        let n_sensors = self.light_sensor_data[0].len();
        let mut data = Array2::zeros((n_sensors, n_time));

        for (t, sensor_data) in self.light_sensor_data.iter().enumerate() {
            for (s, &value) in sensor_data.iter().enumerate() {
                data[[s, t]] = value;
            }
        }

        Some(data)
    }

    /// Create new recorder from configuration
    pub fn from_config(config: RecorderConfig, sensor: Sensor, time: &Time, grid: &Grid) -> Self {
        let grid_shape = (grid.nx, grid.ny, grid.nz);

        let sl_detector = config.sl_detector_config.as_ref().map(|cfg| {
            let grid_shape = (grid.nx, grid.ny, grid.nz);
            let grid_spacing = (grid.dx, grid.dy, grid.dz);
            SonoluminescenceDetector::new(grid_shape, grid_spacing, cfg.clone())
        });

        Self {
            sensor,
            filename: config.filename,
            record_pressure: config.record_pressure,
            record_light: config.record_light,
            record_temperature: config.record_temperature,
            record_cavitation: config.record_cavitation,
            record_sonoluminescence: config.record_sonoluminescence,
            fields_snapshots: Vec::new(),
            snapshot_interval: config.snapshot_interval,
            pressure_sensor_data: Vec::new(),
            light_sensor_data: Vec::new(),
            temperature_sensor_data: Vec::new(),
            recorded_steps: Vec::new(),
            time: time.clone(),
            cavitation_events: Vec::new(),
            cavitation_threshold: config.cavitation_threshold,
            cavitation_map: Array3::zeros(grid_shape),
            sl_detector,
            sl_events: Vec::new(),
            sl_intensity_map: Array3::zeros(grid_shape),
            thermal_events: Vec::new(),
            max_temperature_map: Array3::zeros(grid_shape),
            thermal_dose_map: Array3::zeros(grid_shape),
            statistics: RecorderStatistics::new(),
        }
    }

    /// Record field data at current time step
    pub fn record_fields(
        &mut self,
        fields: &Array4<f64>,
        step: usize,
        time: f64,
    ) -> KwaversResult<()> {
        // Update statistics
        if self.record_pressure {
            let pressure_field = fields.index_axis(Axis(0), PRESSURE_IDX);
            let max_p = pressure_field
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let min_p = pressure_field.iter().copied().fold(f64::INFINITY, f64::min);
            self.statistics.update_pressure(max_p);
            self.statistics.update_pressure(min_p);
        }

        if self.record_temperature {
            let temp_field = fields.index_axis(Axis(0), TEMPERATURE_IDX);
            let max_t = temp_field.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            self.statistics.update_temperature(max_t);
        }

        if self.record_light {
            let light_field = fields.index_axis(Axis(0), LIGHT_IDX);
            let max_l = light_field.iter().copied().fold(0.0, f64::max);
            self.statistics.update_light_intensity(max_l);
        }

        // Store snapshot if needed
        if step.is_multiple_of(self.snapshot_interval) {
            self.fields_snapshots.push((step, fields.clone()));
            self.statistics.total_snapshots += 1;
        }

        // Detect events
        if self.record_cavitation {
            self.detect_cavitation_events(fields, step, time)?;
        }

        if self.record_sonoluminescence {
            self.detect_sonoluminescence_events(fields, step, time)?;
        }

        if self.record_temperature {
            self.detect_thermal_events(fields, step, time)?;
        }

        Ok(())
    }

    /// Detect cavitation events in the field
    fn detect_cavitation_events(
        &mut self,
        fields: &Array4<f64>,
        step: usize,
        time: f64,
    ) -> KwaversResult<()> {
        let pressure_field = fields.index_axis(Axis(0), PRESSURE_IDX);
        let bubble_field = fields.index_axis(Axis(0), BUBBLE_RADIUS_IDX);

        for ((i, j, k), &pressure) in pressure_field.indexed_iter() {
            if pressure < self.cavitation_threshold {
                let radius = bubble_field[[i, j, k]];
                self.cavitation_events.push(CavitationEvent {
                    time_step: step,
                    time,
                    position: [i, j, k],
                    pressure,
                    bubble_radius: radius,
                });
                self.cavitation_map[[i, j, k]] += 1.0;
                self.statistics.total_cavitation_events += 1;
            }
        }

        Ok(())
    }

    /// Detect sonoluminescence events
    fn detect_sonoluminescence_events(
        &mut self,
        fields: &Array4<f64>,
        _step: usize,
        _time: f64,
    ) -> KwaversResult<()> {
        if let Some(ref mut detector) = self.sl_detector {
            // Extract bubble state and pressure fields
            let pressure_field = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
            let bubble_radius = fields.index_axis(Axis(0), BUBBLE_RADIUS_IDX).to_owned();

            // Create bubble state fields (simplified - would need proper state in production)
            let bubble_states = BubbleStateFields {
                radius: bubble_radius,
                velocity: Array3::zeros(pressure_field.dim()),
                temperature: Array3::zeros(pressure_field.dim()),
                pressure: pressure_field.clone(),
                is_collapsing: Array3::zeros(pressure_field.dim()),
                compression_ratio: Array3::from_elem(pressure_field.dim(), 1.0),
            };

            // Use current radius as initial radius (simplified)
            let initial_radius = bubble_states.radius.clone();
            let dt = self.time.dt;
            let events =
                detector.detect_events(&bubble_states, &pressure_field, &initial_radius, dt);
            for event in events {
                self.sl_events.push(event.clone());
                let (i, j, k) = event.position;
                self.sl_intensity_map[[i, j, k]] += event.photon_count;
                self.statistics.total_sl_events += 1;
            }
        }
        Ok(())
    }

    /// Detect thermal events
    fn detect_thermal_events(
        &mut self,
        fields: &Array4<f64>,
        step: usize,
        time: f64,
    ) -> KwaversResult<()> {
        let temp_field = fields.index_axis(Axis(0), TEMPERATURE_IDX);
        const THERMAL_THRESHOLD: f64 = 343.15; // 70°C

        for ((i, j, k), &temperature) in temp_field.indexed_iter() {
            if temperature > THERMAL_THRESHOLD {
                self.thermal_events.push(ThermalEvent {
                    time_step: step,
                    time,
                    position: [i, j, k],
                    temperature,
                });
                self.max_temperature_map[[i, j, k]] =
                    self.max_temperature_map[[i, j, k]].max(temperature);
                self.statistics.total_thermal_events += 1;
            }
        }

        Ok(())
    }

    /// Save recorded data to file
    pub fn save_data(&self) -> KwaversResult<()> {
        info!("Saving recorded data to {}", self.filename);

        // Create output file
        let mut file = File::create(&self.filename)?;

        // Write header
        writeln!(file, "# Kwavers Simulation Output")?;
        writeln!(file, "# Time steps: {}", self.recorded_steps.len())?;
        writeln!(file, "# Sensors: {}", self.sensor.num_sensors())?;

        // Write statistics
        writeln!(file, "\n# Statistics")?;
        writeln!(
            file,
            "# Total snapshots: {}",
            self.statistics.total_snapshots
        )?;
        writeln!(
            file,
            "# Cavitation events: {}",
            self.statistics.total_cavitation_events
        )?;
        writeln!(
            file,
            "# Sonoluminescence events: {}",
            self.statistics.total_sl_events
        )?;
        writeln!(
            file,
            "# Thermal events: {}",
            self.statistics.total_thermal_events
        )?;

        // Write time series data
        writeln!(file, "\n# Time Series Data")?;
        for (i, &time) in self.recorded_steps.iter().enumerate() {
            write!(file, "{time:.6e}")?;

            if self.record_pressure && i < self.pressure_sensor_data.len() {
                for &val in &self.pressure_sensor_data[i] {
                    write!(file, "\t{val:.6e}")?;
                }
            }

            if self.record_light && i < self.light_sensor_data.len() {
                for &val in &self.light_sensor_data[i] {
                    write!(file, "\t{val:.6e}")?;
                }
            }

            writeln!(file)?;
        }

        info!("Data saved successfully");
        Ok(())
    }
}

impl RecorderTrait for Recorder {
    fn initialize(&mut self, grid: &Grid) -> KwaversResult<()> {
        info!(
            "Initializing recorder for grid {}x{}x{}",
            grid.nx, grid.ny, grid.nz
        );

        // Pre-allocate storage based on expected number of time steps
        let expected_steps = (self.time.duration() / self.time.dt) as usize;
        let expected_snapshots = expected_steps / self.snapshot_interval;

        self.fields_snapshots.reserve(expected_snapshots);
        self.recorded_steps.reserve(expected_steps);

        if self.record_pressure {
            self.pressure_sensor_data.reserve(expected_steps);
        }
        if self.record_light {
            self.light_sensor_data.reserve(expected_steps);
        }
        if self.record_temperature {
            self.temperature_sensor_data.reserve(expected_steps);
        }

        Ok(())
    }

    fn record(&mut self, fields: &Array4<f64>, step: usize) -> KwaversResult<()> {
        let time = step as f64 * self.time.dt;
        self.recorded_steps.push(time);

        // Record sensor data
        if self.record_pressure {
            let pressure_field = fields.index_axis(Axis(0), PRESSURE_IDX);
            let sensor_data = self.sensor.sample(&pressure_field.to_owned());
            self.pressure_sensor_data.push(sensor_data);
        }

        if self.record_light {
            let light_field = fields.index_axis(Axis(0), LIGHT_IDX);
            let sensor_data = self.sensor.sample(&light_field.to_owned());
            self.light_sensor_data.push(sensor_data);
        }

        if self.record_temperature {
            let temp_field = fields.index_axis(Axis(0), TEMPERATURE_IDX);
            let sensor_data = self.sensor.sample(&temp_field.to_owned());
            self.temperature_sensor_data.push(sensor_data);
        }

        // Record full fields and detect events
        self.record_fields(fields, step, time)?;

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        info!("Finalizing recording");

        // Print statistics
        self.statistics.print_summary();

        // Save data to file
        self.save_data()?;

        Ok(())
    }
}
