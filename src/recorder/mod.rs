// recorder/mod.rs
use crate::sensor::Sensor;
use crate::time::Time;
use log::{debug, error, info};
use ndarray::{Array2, Array4, Axis};
use std::fs::File;
use std::io::{self, Write};

// Field indices
const PRESSURE_IDX: usize = 0;
const LIGHT_IDX: usize = 1;
const TEMPERATURE_IDX: usize = 2;
const BUBBLE_RADIUS_IDX: usize = 3;

#[derive(Debug)]
pub struct Recorder {
    pub sensor: Sensor,
    pub filename: String,
    pub record_pressure: bool,
    pub record_light: bool,
    pub fields_snapshots: Vec<(usize, Array4<f64>)>, // (time_step, fields)
    pub snapshot_interval: usize,
    pub pressure_sensor_data: Vec<Vec<f64>>,
    pub light_sensor_data: Vec<Vec<f64>>,
    pub recorded_steps: Vec<f64>,
    pub time: Time,
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
        Self {
            sensor,
            filename: filename.to_string(),
            record_pressure,
            record_light,
            fields_snapshots: Vec::new(),
            snapshot_interval,
            pressure_sensor_data: Vec::new(),
            light_sensor_data: Vec::new(),
            recorded_steps: Vec::new(),
            time: time.clone(),
        }
    }

    pub fn default_sdt(sensor: Sensor, time: &Time) -> Self {
        Self::new(sensor, time, "sensor_data", true, true, 10)
    }

    pub fn record(&mut self, fields: &Array4<f64>, time_step: usize, current_time: f64) {
        debug!("Recording at step {} (t = {:.6e})", time_step, current_time);

        if self.record_pressure {
            let pressure = fields.index_axis(Axis(0), PRESSURE_IDX);
            let pressure_values: Vec<f64> = self
                .sensor
                .positions()
                .iter()
                .map(|&(ix, iy, iz)| pressure[[ix, iy, iz]])
                .collect();
            self.pressure_sensor_data.push(pressure_values);
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
        }

        if (self.record_pressure || self.record_light)
            && !self.recorded_steps.contains(&current_time)
        {
            self.recorded_steps.push(current_time);
        }

        if time_step % self.snapshot_interval == 0 {
            debug!("Saving 4D snapshot at step {}", time_step);
            self.fields_snapshots.push((time_step, fields.clone()));
        }
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
        writeln!(file)?;

        // Data
        for (step, &time_val) in self.recorded_steps.iter().enumerate() {
            write!(file, "{}", time_val)?;
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
            writeln!(file)?;
        }

        // Save 4D snapshots
        info!("Saving {} full 4D snapshots", self.fields_snapshots.len());
        for (step, fields) in &self.fields_snapshots {
            let pressure = fields.index_axis(Axis(0), PRESSURE_IDX);
            let light = fields.index_axis(Axis(0), LIGHT_IDX);
            let temperature = fields.index_axis(Axis(0), TEMPERATURE_IDX);
            let bubble_radius = fields.index_axis(Axis(0), BUBBLE_RADIUS_IDX);

            // Save pressure
            let filename = format!("snapshot_step_{}_pressure_3d.csv", step);
            let mut file = File::create(&filename)?;
            writeln!(file, "x,y,z,pressure")?;
            pressure
                .indexed_iter()
                .try_for_each(|((ix, iy, iz), &val)| {
                    let x = ix as f64 * self.sensor.grid_dx();
                    let y = iy as f64 * self.sensor.grid_dy();
                    let z = iz as f64 * self.sensor.grid_dz();
                    writeln!(file, "{},{},{},{}", x, y, z, val)
                })?;

            // Save light
            let filename = format!("snapshot_step_{}_light_3d.csv", step);
            let mut file = File::create(&filename)?;
            writeln!(file, "x,y,z,fluence_rate")?;
            light.indexed_iter().try_for_each(|((ix, iy, iz), &val)| {
                let x = ix as f64 * self.sensor.grid_dx();
                let y = iy as f64 * self.sensor.grid_dy();
                let z = iz as f64 * self.sensor.grid_dz();
                writeln!(file, "{},{},{},{}", x, y, z, val)
            })?;

            // Save temperature
            let filename = format!("snapshot_step_{}_temperature_3d.csv", step);
            let mut file = File::create(&filename)?;
            writeln!(file, "x,y,z,temperature")?;
            temperature
                .indexed_iter()
                .try_for_each(|((ix, iy, iz), &val)| {
                    let x = ix as f64 * self.sensor.grid_dx();
                    let y = iy as f64 * self.sensor.grid_dy();
                    let z = iz as f64 * self.sensor.grid_dz();
                    writeln!(file, "{},{},{},{}", x, y, z, val)
                })?;

            // Save bubble radius
            let filename = format!("snapshot_step_{}_cavitation_3d.csv", step);
            let mut file = File::create(&filename)?;
            writeln!(file, "x,y,z,bubble_radius")?;
            bubble_radius
                .indexed_iter()
                .try_for_each(|((ix, iy, iz), &val)| {
                    let x = ix as f64 * self.sensor.grid_dx();
                    let y = iy as f64 * self.sensor.grid_dy();
                    let z = iz as f64 * self.sensor.grid_dz();
                    writeln!(file, "{},{},{},{}", x, y, z, val)
                })?;
        }

        if self.fields_snapshots.is_empty() {
            error!("No 4D snapshots saved. Check recording settings.");
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
}
