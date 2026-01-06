//! Sensor recording functionality matching k-Wave
//!
//! Implements various sensor types and recording options

use ndarray::{s, Array2, Array3, Array4};

/// Sensor configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SensorConfig {
    /// Binary mask indicating sensor positions (1 for sensor, 0 otherwise)
    pub mask: Array3<bool>,

    pub record_p: bool,
    pub record_p_max: bool,
    pub record_p_min: bool,
    pub record_p_rms: bool,
    pub record_u: bool,
    pub record_u_max: bool,
    pub record_u_min: bool,
    pub record_u_rms: bool,
    pub record_i: bool,
    pub record_i_avg: bool,

    /// Frequency response filter (optional)
    pub frequency_response: Option<Vec<f64>>,
}

/// Compatibility alias for k-Wave users
pub type SpectralSensorConfig = SensorConfig;

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            mask: Array3::from_elem((1, 1, 1), false),
            record_p: true,
            record_p_max: false,
            record_p_min: false,
            record_p_rms: false,
            record_u: false,
            record_u_max: false,
            record_u_min: false,
            record_u_rms: false,
            record_i: false,
            record_i_avg: false,
            frequency_response: None,
        }
    }
}

/// Container for recorded sensor data
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct SensorData {
    /// Time varying pressure [sensor_idx, time_step]
    pub p: Option<Array2<f64>>,

    /// Maximum pressure [x, y, z] (masked) or [sensor_idx]
    pub p_max: Option<Array3<f64>>,
    pub p_min: Option<Array3<f64>>,
    pub p_rms: Option<Array3<f64>>,

    /// Time varying velocity [3, sensor_idx, time_step]
    pub u: Option<Array3<f64>>,

    pub u_max: Option<Array4<f64>>,
    pub u_min: Option<Array4<f64>>,
    pub u_rms: Option<Array4<f64>>,

    /// Time varying intensity [3, sensor_idx, time_step]
    pub i: Option<Array3<f64>>,
    pub i_avg: Option<Array4<f64>>,
}

/// Compatibility alias for k-Wave users
pub type KWaveSensorData = SensorData;

/// Handles the recording of sensor data during simulation
#[derive(Debug)]
pub struct SensorHandler {
    config: SensorConfig,
    sensor_indices: Vec<(usize, usize, usize)>,
    p_buffer: Vec<Vec<f64>>,
    ux_buffer: Vec<Vec<f64>>,
    uy_buffer: Vec<Vec<f64>>,
    uz_buffer: Vec<Vec<f64>>,
    ix_buffer: Vec<Vec<f64>>,
    iy_buffer: Vec<Vec<f64>>,
    iz_buffer: Vec<Vec<f64>>,
    p_max_acc: Array3<f64>,
    p_min_acc: Array3<f64>,
    p_sq_acc: Array3<f64>,
    ux_max_acc: Array3<f64>,
    ux_min_acc: Array3<f64>,
    ux_sq_acc: Array3<f64>,
    uy_max_acc: Array3<f64>,
    uy_min_acc: Array3<f64>,
    uy_sq_acc: Array3<f64>,
    uz_max_acc: Array3<f64>,
    uz_min_acc: Array3<f64>,
    uz_sq_acc: Array3<f64>,
    ix_avg_acc: Array3<f64>,
    iy_avg_acc: Array3<f64>,
    iz_avg_acc: Array3<f64>,
    time_steps_recorded: usize,
}

impl SensorHandler {
    pub fn new(config: SensorConfig, grid_shape: (usize, usize, usize)) -> Self {
        let mut sensor_indices = Vec::new();

        for ((i, j, k), &is_sensor) in config.mask.indexed_iter() {
            if is_sensor {
                sensor_indices.push((i, j, k));
            }
        }

        let num_sensors = sensor_indices.len();

        let p_buffer = if config.record_p {
            vec![Vec::new(); num_sensors]
        } else {
            Vec::new()
        };
        let ux_buffer = if config.record_u {
            vec![Vec::new(); num_sensors]
        } else {
            Vec::new()
        };
        let uy_buffer = if config.record_u {
            vec![Vec::new(); num_sensors]
        } else {
            Vec::new()
        };
        let uz_buffer = if config.record_u {
            vec![Vec::new(); num_sensors]
        } else {
            Vec::new()
        };

        let ix_buffer = if config.record_i {
            vec![Vec::new(); num_sensors]
        } else {
            Vec::new()
        };
        let iy_buffer = if config.record_i {
            vec![Vec::new(); num_sensors]
        } else {
            Vec::new()
        };
        let iz_buffer = if config.record_i {
            vec![Vec::new(); num_sensors]
        } else {
            Vec::new()
        };

        let p_max_acc = if config.record_p_max {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };
        let p_min_acc = if config.record_p_min {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };
        let p_sq_acc = if config.record_p_rms {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };

        let ux_max_acc = if config.record_u_max {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };
        let ux_min_acc = if config.record_u_min {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };
        let ux_sq_acc = if config.record_u_rms {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };

        let uy_max_acc = if config.record_u_max {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };
        let uy_min_acc = if config.record_u_min {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };
        let uy_sq_acc = if config.record_u_rms {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };

        let uz_max_acc = if config.record_u_max {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };
        let uz_min_acc = if config.record_u_min {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };
        let uz_sq_acc = if config.record_u_rms {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };

        let ix_avg_acc = if config.record_i_avg {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };
        let iy_avg_acc = if config.record_i_avg {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };
        let iz_avg_acc = if config.record_i_avg {
            Array3::zeros(grid_shape)
        } else {
            Array3::zeros((0, 0, 0))
        };

        Self {
            config,
            sensor_indices,
            p_buffer,
            ux_buffer,
            uy_buffer,
            uz_buffer,
            ix_buffer,
            iy_buffer,
            iz_buffer,
            p_max_acc,
            p_min_acc,
            p_sq_acc,
            ux_max_acc,
            ux_min_acc,
            ux_sq_acc,
            uy_max_acc,
            uy_min_acc,
            uy_sq_acc,
            uz_max_acc,
            uz_min_acc,
            uz_sq_acc,
            ix_avg_acc,
            iy_avg_acc,
            iz_avg_acc,
            time_steps_recorded: 0,
        }
    }

    pub fn with_expected_steps(
        config: SensorConfig,
        grid_shape: (usize, usize, usize),
        expected_steps: usize,
    ) -> Self {
        let mut handler = Self::new(config, grid_shape);
        handler.reserve_steps(expected_steps);
        handler
    }

    pub fn reserve_steps(&mut self, expected_steps: usize) {
        if expected_steps == 0 {
            return;
        }

        for buf in &mut self.p_buffer {
            buf.reserve(expected_steps);
        }
        for buf in &mut self.ux_buffer {
            buf.reserve(expected_steps);
        }
        for buf in &mut self.uy_buffer {
            buf.reserve(expected_steps);
        }
        for buf in &mut self.uz_buffer {
            buf.reserve(expected_steps);
        }
        for buf in &mut self.ix_buffer {
            buf.reserve(expected_steps);
        }
        for buf in &mut self.iy_buffer {
            buf.reserve(expected_steps);
        }
        for buf in &mut self.iz_buffer {
            buf.reserve(expected_steps);
        }
    }

    pub fn record_step(
        &mut self,
        p: &Array3<f64>,
        ux: &Array3<f64>,
        uy: &Array3<f64>,
        uz: &Array3<f64>,
    ) {
        let is_first_step = self.time_steps_recorded == 0;
        self.time_steps_recorded += 1;

        if self.config.record_p {
            for (idx, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
                self.p_buffer[idx].push(p[[i, j, k]]);
            }
        }

        if self.config.record_u {
            for (idx, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
                self.ux_buffer[idx].push(ux[[i, j, k]]);
                self.uy_buffer[idx].push(uy[[i, j, k]]);
                self.uz_buffer[idx].push(uz[[i, j, k]]);
            }
        }

        if self.config.record_p_max || self.config.record_p_min || self.config.record_p_rms {
            for &(i, j, k) in &self.sensor_indices {
                let p_val = p[[i, j, k]];
                if self.config.record_p_max {
                    let acc = &mut self.p_max_acc[[i, j, k]];
                    if is_first_step || p_val > *acc {
                        *acc = p_val;
                    }
                }
                if self.config.record_p_min {
                    let acc = &mut self.p_min_acc[[i, j, k]];
                    if is_first_step || p_val < *acc {
                        *acc = p_val;
                    }
                }
                if self.config.record_p_rms {
                    self.p_sq_acc[[i, j, k]] += p_val * p_val;
                }
            }
        }

        if self.config.record_u_max || self.config.record_u_min || self.config.record_u_rms {
            for &(i, j, k) in &self.sensor_indices {
                let ux_val = ux[[i, j, k]];
                let uy_val = uy[[i, j, k]];
                let uz_val = uz[[i, j, k]];

                if self.config.record_u_max {
                    let acc = &mut self.ux_max_acc[[i, j, k]];
                    if is_first_step || ux_val > *acc {
                        *acc = ux_val;
                    }
                    let acc = &mut self.uy_max_acc[[i, j, k]];
                    if is_first_step || uy_val > *acc {
                        *acc = uy_val;
                    }
                    let acc = &mut self.uz_max_acc[[i, j, k]];
                    if is_first_step || uz_val > *acc {
                        *acc = uz_val;
                    }
                }

                if self.config.record_u_min {
                    let acc = &mut self.ux_min_acc[[i, j, k]];
                    if is_first_step || ux_val < *acc {
                        *acc = ux_val;
                    }
                    let acc = &mut self.uy_min_acc[[i, j, k]];
                    if is_first_step || uy_val < *acc {
                        *acc = uy_val;
                    }
                    let acc = &mut self.uz_min_acc[[i, j, k]];
                    if is_first_step || uz_val < *acc {
                        *acc = uz_val;
                    }
                }

                if self.config.record_u_rms {
                    self.ux_sq_acc[[i, j, k]] += ux_val * ux_val;
                    self.uy_sq_acc[[i, j, k]] += uy_val * uy_val;
                    self.uz_sq_acc[[i, j, k]] += uz_val * uz_val;
                }
            }
        }

        if self.config.record_i {
            for (idx, &(i, j, k)) in self.sensor_indices.iter().enumerate() {
                let p_val = p[[i, j, k]];
                self.ix_buffer[idx].push(p_val * ux[[i, j, k]]);
                self.iy_buffer[idx].push(p_val * uy[[i, j, k]]);
                self.iz_buffer[idx].push(p_val * uz[[i, j, k]]);
            }
        }

        if self.config.record_i_avg {
            for &(i, j, k) in &self.sensor_indices {
                let p_val = p[[i, j, k]];
                self.ix_avg_acc[[i, j, k]] += p_val * ux[[i, j, k]];
                self.iy_avg_acc[[i, j, k]] += p_val * uy[[i, j, k]];
                self.iz_avg_acc[[i, j, k]] += p_val * uz[[i, j, k]];
            }
        }
    }

    pub fn extract_data(&self) -> SensorData {
        let mut data = SensorData::default();

        let num_sensors = self.sensor_indices.len();
        let num_steps = self.time_steps_recorded;

        if num_steps == 0 {
            return data;
        }

        if self.config.record_p && num_sensors > 0 {
            let mut arr = Array2::zeros((num_sensors, num_steps));
            for (i, time_series) in self.p_buffer.iter().enumerate() {
                for (t, &val) in time_series.iter().enumerate() {
                    arr[[i, t]] = val;
                }
            }
            data.p = Some(arr);
        }

        if self.config.record_u && num_sensors > 0 {
            let mut arr = Array3::zeros((3, num_sensors, num_steps));
            for (i, _) in self.sensor_indices.iter().enumerate() {
                for t in 0..num_steps {
                    arr[[0, i, t]] = self.ux_buffer[i][t];
                    arr[[1, i, t]] = self.uy_buffer[i][t];
                    arr[[2, i, t]] = self.uz_buffer[i][t];
                }
            }
            data.u = Some(arr);
        }

        if self.config.record_i && num_sensors > 0 {
            let mut arr = Array3::zeros((3, num_sensors, num_steps));
            for (i, _) in self.sensor_indices.iter().enumerate() {
                for t in 0..num_steps {
                    arr[[0, i, t]] = self.ix_buffer[i][t];
                    arr[[1, i, t]] = self.iy_buffer[i][t];
                    arr[[2, i, t]] = self.iz_buffer[i][t];
                }
            }
            data.i = Some(arr);
        }

        if self.config.record_p_max {
            data.p_max = Some(self.p_max_acc.clone());
        }

        if self.config.record_p_min {
            data.p_min = Some(self.p_min_acc.clone());
        }

        if self.config.record_p_rms && num_steps > 0 {
            let mut rms = self.p_sq_acc.clone();
            rms.mapv_inplace(|x| (x / num_steps as f64).sqrt());
            data.p_rms = Some(rms);
        }

        if self.config.record_u_max {
            let shape = self.ux_max_acc.shape();
            let mut arr = Array4::zeros((3, shape[0], shape[1], shape[2]));
            arr.slice_mut(s![0, .., .., ..]).assign(&self.ux_max_acc);
            arr.slice_mut(s![1, .., .., ..]).assign(&self.uy_max_acc);
            arr.slice_mut(s![2, .., .., ..]).assign(&self.uz_max_acc);
            data.u_max = Some(arr);
        }

        if self.config.record_u_min {
            let shape = self.ux_min_acc.shape();
            let mut arr = Array4::zeros((3, shape[0], shape[1], shape[2]));
            arr.slice_mut(s![0, .., .., ..]).assign(&self.ux_min_acc);
            arr.slice_mut(s![1, .., .., ..]).assign(&self.uy_min_acc);
            arr.slice_mut(s![2, .., .., ..]).assign(&self.uz_min_acc);
            data.u_min = Some(arr);
        }

        if self.config.record_u_rms && num_steps > 0 {
            let shape = self.ux_sq_acc.shape();
            let mut arr = Array4::zeros((3, shape[0], shape[1], shape[2]));

            let mut ux_rms = self.ux_sq_acc.clone();
            ux_rms.mapv_inplace(|x| (x / num_steps as f64).sqrt());

            let mut uy_rms = self.uy_sq_acc.clone();
            uy_rms.mapv_inplace(|x| (x / num_steps as f64).sqrt());

            let mut uz_rms = self.uz_sq_acc.clone();
            uz_rms.mapv_inplace(|x| (x / num_steps as f64).sqrt());

            arr.slice_mut(s![0, .., .., ..]).assign(&ux_rms);
            arr.slice_mut(s![1, .., .., ..]).assign(&uy_rms);
            arr.slice_mut(s![2, .., .., ..]).assign(&uz_rms);
            data.u_rms = Some(arr);
        }

        if self.config.record_i_avg && num_steps > 0 {
            let shape = self.ix_avg_acc.shape();
            let mut arr = Array4::zeros((3, shape[0], shape[1], shape[2]));

            let mut ix_avg = self.ix_avg_acc.clone();
            ix_avg.mapv_inplace(|x| x / num_steps as f64);

            let mut iy_avg = self.iy_avg_acc.clone();
            iy_avg.mapv_inplace(|x| x / num_steps as f64);

            let mut iz_avg = self.iz_avg_acc.clone();
            iz_avg.mapv_inplace(|x| x / num_steps as f64);

            arr.slice_mut(s![0, .., .., ..]).assign(&ix_avg);
            arr.slice_mut(s![1, .., .., ..]).assign(&iy_avg);
            arr.slice_mut(s![2, .., .., ..]).assign(&iz_avg);
            data.i_avg = Some(arr);
        }

        data
    }

    pub fn sensor_indices(&self) -> &[(usize, usize, usize)] {
        &self.sensor_indices
    }
}
