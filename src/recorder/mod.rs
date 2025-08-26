// recorder/mod.rs
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::bubble_dynamics::BubbleStateFields;
use crate::physics::field_indices::{BUBBLE_RADIUS_IDX, LIGHT_IDX, PRESSURE_IDX, TEMPERATURE_IDX};
use crate::physics::sonoluminescence_detector::{
    DetectorConfig, SonoluminescenceDetector, SonoluminescenceEvent,
};
use crate::sensor::Sensor;
use crate::time::Time;
use log::{debug, info};
use ndarray::{Array2, Array3, Array4, Axis};
use std::fs::File;
use std::io::{self, Write};

/// Trait for data recording (Dependency Inversion Principle)
pub trait RecorderTrait: Send + Sync {
    /// Initialize the recorder
    fn initialize(&mut self, grid: &Grid) -> KwaversResult<()>;

    /// Record data at a specific time step
    fn record(&mut self, fields: &Array4<f64>, step: usize) -> KwaversResult<()>;

    /// Finalize recording and save data
    fn finalize(&mut self) -> KwaversResult<()>;
}

/// Configuration for recorder setup
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    pub filename: String,
    pub record_pressure: bool,
    pub record_light: bool,
    pub record_temperature: bool,
    pub record_cavitation: bool,
    pub record_sonoluminescence: bool,
    pub snapshot_interval: usize,
    /// Threshold for cavitation detection (Pa)
    pub cavitation_threshold: f64,
    /// Configuration for sonoluminescence detection
    pub sl_detector_config: Option<DetectorConfig>,
}

impl RecorderConfig {
    pub fn new(filename: &str) -> Self {
        Self {
            filename: filename.to_string(),
            record_pressure: true,
            record_light: true,
            record_temperature: false,
            record_cavitation: false,
            record_sonoluminescence: false,
            snapshot_interval: 1,
            cavitation_threshold: -1e5, // -1 bar for cavitation
            sl_detector_config: None,
        }
    }

    pub fn with_pressure_recording(mut self, record: bool) -> Self {
        self.record_pressure = record;
        self
    }

    pub fn with_light_recording(mut self, record: bool) -> Self {
        self.record_light = record;
        self
    }

    pub fn with_temperature_recording(mut self, record: bool) -> Self {
        self.record_temperature = record;
        self
    }

    pub fn with_cavitation_detection(mut self, enable: bool, threshold: f64) -> Self {
        self.record_cavitation = enable;
        self.cavitation_threshold = threshold;
        self
    }

    pub fn with_sonoluminescence_detection(
        mut self,
        enable: bool,
        config: Option<DetectorConfig>,
    ) -> Self {
        self.record_sonoluminescence = enable;
        self.sl_detector_config = config;
        self
    }

    pub fn with_snapshot_interval(mut self, interval: usize) -> Self {
        self.snapshot_interval = interval;
        self
    }
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self::new("simulation_output")
    }
}

/// Cavitation event data
#[derive(Debug, Clone)]
pub struct CavitationEvent {
    pub time: f64,
    pub position: (usize, usize, usize),
    pub pressure: f64,
    pub radius: f64,
    pub collapse_rate: f64,
}

/// Thermal event data
#[derive(Debug, Clone)]
pub struct ThermalEvent {
    pub time: f64,
    pub position: (usize, usize, usize),
    pub temperature: f64,
    pub heating_rate: f64,
}

#[derive(Debug)]
pub struct Recorder {
    pub sensor: Sensor,
    pub filename: String,
    pub record_pressure: bool,
    pub record_light: bool,
    pub record_temperature: bool,
    pub record_cavitation: bool,
    pub record_sonoluminescence: bool,
    pub fields_snapshots: Vec<(usize, Array4<f64>)>, // (time_step, fields)
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
    pub thermal_dose_map: Array3<f64>, // CEM43 thermal dose

    // Statistics
    pub statistics: RecorderStatistics,
}

/// Statistics collected during recording
#[derive(Debug, Default, Clone)]
pub struct RecorderStatistics {
    pub total_cavitation_events: usize,
    pub total_sl_events: usize,
    pub total_thermal_events: usize,
    pub max_pressure: f64,
    pub min_pressure: f64,
    pub max_temperature: f64,
    pub max_light_intensity: f64,
    pub total_sl_photons: f64,
    pub total_sl_energy: f64,
}

impl Recorder {
    pub fn new(
        sensor: Sensor,
        time: &Time,
        filename: &str,
        record_pressure: bool,
        record_light: bool,
        snapshot_interval: usize,
    ) -> Self {
        info!(
            "Initialized Recorder: {} sensors, duration {:.6e} s, pressure: {}, light: {}",
            sensor.num_sensors(),
            time.duration(),
            record_pressure,
            record_light
        );
        assert!(time.dt > 0.0, "Time step must be positive");

        // Get grid dimensions from sensor
        let grid_shape = sensor.grid_shape();

        Self {
            sensor,
            filename: filename.to_string(),
            record_pressure,
            record_light,
            record_temperature: false,
            record_cavitation: false,
            record_sonoluminescence: false,
            fields_snapshots: Vec::new(),
            snapshot_interval,
            pressure_sensor_data: Vec::new(),
            light_sensor_data: Vec::new(),
            temperature_sensor_data: Vec::new(),
            recorded_steps: Vec::new(),
            time: time.clone(),
            cavitation_events: Vec::new(),
            cavitation_threshold: -1e5,
            cavitation_map: Array3::zeros(grid_shape),
            sl_detector: None,
            sl_events: Vec::new(),
            sl_intensity_map: Array3::zeros(grid_shape),
            thermal_events: Vec::new(),
            max_temperature_map: Array3::zeros(grid_shape),
            thermal_dose_map: Array3::zeros(grid_shape),
            statistics: RecorderStatistics::default(),
        }
    }

    pub fn default_sdt(sensor: Sensor, time: &Time) -> Self {
        Self::new(sensor, time, "sensor_data", true, true, 10)
    }

    /// Creates a recorder from configuration
    pub fn from_config(sensor: Sensor, time: &Time, config: &RecorderConfig) -> Self {
        let grid_shape = sensor.grid_shape();

        let mut recorder = Self::new(
            sensor,
            time,
            &config.filename,
            config.record_pressure,
            config.record_light,
            config.snapshot_interval,
        );

        recorder.record_temperature = config.record_temperature;
        recorder.record_cavitation = config.record_cavitation;
        recorder.record_sonoluminescence = config.record_sonoluminescence;
        recorder.cavitation_threshold = config.cavitation_threshold;

        // Initialize sonoluminescence detector if enabled
        if config.record_sonoluminescence {
            let sl_config = config.sl_detector_config.clone().unwrap_or_default();

            let grid_spacing = (1e-3, 1e-3, 1e-3); // Default 1mm spacing
            recorder.sl_detector = Some(SonoluminescenceDetector::new(
                grid_shape,
                grid_spacing,
                sl_config,
            ));
        }

        recorder
    }

    pub fn record(&mut self, fields: &Array4<f64>, time_step: usize, current_time: f64) {
        debug!("Recording at step {} (t = {:.6e})", time_step, current_time);

        // Record standard fields at sensor positions
        if self.record_pressure {
            let pressure = fields.index_axis(Axis(0), PRESSURE_IDX);
            let pressure_values: Vec<f64> = self
                .sensor
                .positions()
                .iter()
                .map(|&(ix, iy, iz)| pressure[[ix, iy, iz]])
                .collect();
            self.pressure_sensor_data.push(pressure_values);

            // Update statistics
            let max_p = pressure.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_p = pressure.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            self.statistics.max_pressure = self.statistics.max_pressure.max(max_p);
            self.statistics.min_pressure = self.statistics.min_pressure.min(min_p);
        }

        if self.record_light {
            let light = fields.index_axis(Axis(0), LIGHT_IDX);
            let light_values: Vec<f64> = self
                .sensor
                .positions()
                .iter()
                .map(|&(ix, iy, iz)| light[[ix, iy, iz]])
                .collect();
            self.light_sensor_data.push(light_values);

            // Update statistics
            let max_l = light.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            self.statistics.max_light_intensity = self.statistics.max_light_intensity.max(max_l);
        }

        if self.record_temperature {
            let temperature = fields.index_axis(Axis(0), TEMPERATURE_IDX);
            let temperature_values: Vec<f64> = self
                .sensor
                .positions()
                .iter()
                .map(|&(ix, iy, iz)| temperature[[ix, iy, iz]])
                .collect();
            self.temperature_sensor_data.push(temperature_values);

            // Update statistics and thermal dose
            self.update_thermal_tracking(&temperature.to_owned(), current_time);
        }

        // Detect and record cavitation events
        if self.record_cavitation {
            self.detect_cavitation_events(fields, current_time);
        }

        // Detect and record sonoluminescence events
        if self.record_sonoluminescence {
            self.detect_sonoluminescence_events(fields, current_time);
        }

        // Store field snapshots
        if time_step % self.snapshot_interval == 0 {
            self.fields_snapshots.push((time_step, fields.clone()));
        }

        self.recorded_steps.push(current_time);
    }

    /// Detect cavitation events based on pressure threshold and bubble dynamics
    fn detect_cavitation_events(&mut self, fields: &Array4<f64>, current_time: f64) {
        let pressure = fields.index_axis(Axis(0), PRESSURE_IDX);
        let bubble_radius = fields.index_axis(Axis(0), BUBBLE_RADIUS_IDX);

        let (nx, ny, nz) = pressure.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let p = pressure[[i, j, k]];
                    let r = bubble_radius[[i, j, k]];

                    // Cavitation detection criteria:
                    // 1. Pressure below threshold (tensile stress)
                    // 2. Bubble present and growing rapidly
                    if p < self.cavitation_threshold && r > 0.0 {
                        // Calculate collapse rate (simplified)
                        let collapse_rate = if r > 1e-6 {
                            (p.abs() / 101325.0) * (1e-6 / r)
                        } else {
                            0.0
                        };

                        let event = CavitationEvent {
                            time: current_time,
                            position: (i, j, k),
                            pressure: p,
                            radius: r,
                            collapse_rate,
                        };

                        self.cavitation_events.push(event);
                        self.cavitation_map[[i, j, k]] += 1.0; // Increment cavitation count
                        self.statistics.total_cavitation_events += 1;
                    }
                }
            }
        }
    }

    /// Detect sonoluminescence events using the integrated detector
    fn detect_sonoluminescence_events(&mut self, fields: &Array4<f64>, current_time: f64) {
        if let Some(ref mut detector) = self.sl_detector {
            // Extract bubble state fields
            let temperature = fields.index_axis(Axis(0), TEMPERATURE_IDX);
            let pressure = fields.index_axis(Axis(0), PRESSURE_IDX);
            let bubble_radius = fields.index_axis(Axis(0), BUBBLE_RADIUS_IDX);

            // Create bubble state fields with all required fields
            let bubble_states = BubbleStateFields {
                radius: bubble_radius.to_owned(),
                velocity: Array3::zeros(bubble_radius.dim()), // Not available here
                pressure: pressure.to_owned(),
                temperature: temperature.to_owned(),
                is_collapsing: Array3::zeros(bubble_radius.dim()), // Default to not collapsing
                compression_ratio: Array3::ones(bubble_radius.dim()), // Default to 1.0
            };

            // Use bubble radius as initial radius (simplified)
            let initial_radius = bubble_radius.mapv(|r| if r > 0.0 { r * 10.0 } else { 0.0 });

            // Detect events
            let dt = self.time.dt;
            let events =
                detector.detect_events(&bubble_states, &pressure.to_owned(), &initial_radius, dt);

            // Update intensity map and statistics
            for event in &events {
                let (i, j, k) = event.position;
                self.sl_intensity_map[[i, j, k]] += event.photon_count;
                self.statistics.total_sl_photons += event.photon_count;
                self.statistics.total_sl_energy += event.energy;
            }

            self.statistics.total_sl_events += events.len();
            self.sl_events.extend(events);
        }
    }

    /// Update thermal tracking and calculate thermal dose
    fn update_thermal_tracking(&mut self, temperature: &Array3<f64>, current_time: f64) {
        let dt = self.time.dt;
        let (nx, ny, nz) = temperature.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let temp = temperature[[i, j, k]];

                    // Update maximum temperature map
                    self.max_temperature_map[[i, j, k]] =
                        self.max_temperature_map[[i, j, k]].max(temp);

                    // Calculate CEM43 thermal dose
                    // CEM43 = Σ R^(43-T) * Δt
                    // where R = 0.5 for T > 43°C, R = 0.25 for T < 43°C
                    if temp > 37.0 {
                        // Only accumulate dose above body temperature
                        let r: f64 = if temp >= 43.0 { 0.5 } else { 0.25 };
                        let cem43_increment = r.powf(43.0 - temp) * (dt / 60.0); // Convert to minutes
                        self.thermal_dose_map[[i, j, k]] += cem43_increment;

                        // Detect significant thermal events (e.g., > 50°C)
                        if temp > 50.0 {
                            let heating_rate = 0.0; // Would need previous temperature to calculate
                            let event = ThermalEvent {
                                time: current_time,
                                position: (i, j, k),
                                temperature: temp,
                                heating_rate,
                            };
                            self.thermal_events.push(event);
                            self.statistics.total_thermal_events += 1;
                        }
                    }
                }
            }
        }

        // Update max temperature statistic
        let max_t = temperature.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        self.statistics.max_temperature = self.statistics.max_temperature.max(max_t);
    }

    pub fn save(&self) -> io::Result<()> {
        info!("Saving recorded data to {}.csv", self.filename);

        let mut file = File::create(format!("{}.csv", self.filename))?;

        // Header
        write!(file, "Time")?;
        if self.record_pressure {
            for i in 0..self.sensor.num_sensors() {
                write!(file, ",Pressure_Sensor{}", i + 1)?;
            }
        }
        if self.record_light {
            for i in 0..self.sensor.num_sensors() {
                write!(file, ",Light_Sensor{}", i + 1)?;
            }
        }
        if self.record_temperature {
            for i in 0..self.sensor.num_sensors() {
                write!(file, ",Temperature_Sensor{}", i + 1)?;
            }
        }
        writeln!(file)?;

        // Data
        for (step, &time) in self.recorded_steps.iter().enumerate() {
            write!(file, "{}", time)?;
            if self.record_pressure && step < self.pressure_sensor_data.len() {
                self.pressure_sensor_data[step]
                    .iter()
                    .try_for_each(|&val| write!(file, ",{}", val))?;
            }
            if self.record_light && step < self.light_sensor_data.len() {
                self.light_sensor_data[step]
                    .iter()
                    .try_for_each(|&val| write!(file, ",{}", val))?;
            }
            if self.record_temperature && step < self.temperature_sensor_data.len() {
                self.temperature_sensor_data[step]
                    .iter()
                    .try_for_each(|&val| write!(file, ",{}", val))?;
            }
            writeln!(file)?;
        }

        info!(
            "Saved {} time steps to {}.csv",
            self.recorded_steps.len(),
            self.filename
        );

        // Save cavitation events if recorded
        if self.record_cavitation && !self.cavitation_events.is_empty() {
            let mut cav_file = File::create(format!("{}_cavitation.csv", self.filename))?;
            writeln!(cav_file, "time,x,y,z,pressure,radius,collapse_rate")?;
            for event in &self.cavitation_events {
                writeln!(
                    cav_file,
                    "{},{},{},{},{},{},{}",
                    event.time,
                    event.position.0,
                    event.position.1,
                    event.position.2,
                    event.pressure,
                    event.radius,
                    event.collapse_rate
                )?;
            }
            info!("Saved {} cavitation events", self.cavitation_events.len());
        }

        // Save sonoluminescence events if recorded
        if self.record_sonoluminescence && !self.sl_events.is_empty() {
            let mut sl_file = File::create(format!("{}_sonoluminescence.csv", self.filename))?;
            writeln!(
                sl_file,
                "time,x,y,z,temperature,pressure,photons,wavelength,energy"
            )?;
            for event in &self.sl_events {
                writeln!(
                    sl_file,
                    "{},{},{},{},{},{},{},{},{}",
                    event.time,
                    event.position.0,
                    event.position.1,
                    event.position.2,
                    event.peak_temperature,
                    event.peak_pressure,
                    event.photon_count,
                    event.peak_wavelength,
                    event.energy
                )?;
            }
            info!("Saved {} sonoluminescence events", self.sl_events.len());
        }

        // Save thermal events if recorded
        if self.record_temperature && !self.thermal_events.is_empty() {
            let mut thermal_file = File::create(format!("{}_thermal.csv", self.filename))?;
            writeln!(thermal_file, "time,x,y,z,temperature,heating_rate")?;
            for event in &self.thermal_events {
                writeln!(
                    thermal_file,
                    "{},{},{},{},{},{}",
                    event.time,
                    event.position.0,
                    event.position.1,
                    event.position.2,
                    event.temperature,
                    event.heating_rate
                )?;
            }
            info!("Saved {} thermal events", self.thermal_events.len());
        }

        Ok(())
    }

    // Accessors
    pub fn sensor(&self) -> &Sensor {
        &self.sensor
    }

    pub fn sensor_mut(&mut self) -> &mut Sensor {
        &mut self.sensor
    }

    pub fn pressure_data(&self) -> Option<Array2<f64>> {
        if self.record_pressure && !self.pressure_sensor_data.is_empty() {
            let n_sensors = self.pressure_sensor_data[0].len();
            let n_steps = self.pressure_sensor_data.len();
            let mut data = Array2::zeros((n_sensors, n_steps));
            for (step, row) in self.pressure_sensor_data.iter().enumerate() {
                for (sensor, &val) in row.iter().enumerate() {
                    data[[sensor, step]] = val;
                }
            }
            Some(data)
        } else {
            None
        }
    }

    pub fn light_data(&self) -> Option<Array2<f64>> {
        if self.record_light && !self.light_sensor_data.is_empty() {
            let n_sensors = self.light_sensor_data[0].len();
            let n_steps = self.light_sensor_data.len();
            let mut data = Array2::zeros((n_sensors, n_steps));
            for (step, row) in self.light_sensor_data.iter().enumerate() {
                for (sensor, &val) in row.iter().enumerate() {
                    data[[sensor, step]] = val;
                }
            }
            Some(data)
        } else {
            None
        }
    }

    pub fn temperature_data(&self) -> Option<Array2<f64>> {
        if self.record_temperature && !self.temperature_sensor_data.is_empty() {
            let n_sensors = self.temperature_sensor_data[0].len();
            let n_steps = self.temperature_sensor_data.len();
            let mut data = Array2::zeros((n_sensors, n_steps));
            for (step, row) in self.temperature_sensor_data.iter().enumerate() {
                for (sensor, &val) in row.iter().enumerate() {
                    data[[sensor, step]] = val;
                }
            }
            Some(data)
        } else {
            None
        }
    }

    pub fn cavitation_map(&self) -> Option<Array3<f64>> {
        if self.record_cavitation {
            Some(self.cavitation_map.clone())
        } else {
            None
        }
    }

    pub fn sonoluminescence_intensity_map(&self) -> Option<Array3<f64>> {
        if self.record_sonoluminescence {
            Some(self.sl_intensity_map.clone())
        } else {
            None
        }
    }

    pub fn thermal_dose_map(&self) -> Option<Array3<f64>> {
        if self.record_temperature {
            Some(self.thermal_dose_map.clone())
        } else {
            None
        }
    }

    pub fn max_temperature_map(&self) -> Option<Array3<f64>> {
        if self.record_temperature {
            Some(self.max_temperature_map.clone())
        } else {
            None
        }
    }
}

impl RecorderTrait for Recorder {
    fn initialize(&mut self, _grid: &Grid) -> KwaversResult<()> {
        info!("Initializing recorder with filename: {}", self.filename);
        Ok(())
    }

    fn record(&mut self, fields: &Array4<f64>, step: usize) -> KwaversResult<()> {
        // Record at specified intervals
        if step % self.snapshot_interval == 0 {
            self.fields_snapshots.push((step, fields.clone()));
        }

        // Record sensor data - extract pressure and light fields
        use ndarray::Axis;
        let pressure_field = fields.index_axis(Axis(0), 0); // Assuming pressure is at index 0
        let light_field = fields.index_axis(Axis(0), 2); // Assuming light is at index 2
        self.sensor
            .record(&pressure_field.to_owned(), &light_field.to_owned(), step);

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.save()?;
        info!("Recorder finalized. Data saved to {}", self.filename);
        Ok(())
    }
}
