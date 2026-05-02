use super::{ThermalAcousticConfig, ThermalAcousticCoupler};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

impl ThermalAcousticCoupler {
    /// Create new thermal-acoustic coupler with default configuration
    pub fn new_default() -> Self {
        Self::new(ThermalAcousticConfig::default()).expect("default config is valid")
    }

    /// Create new thermal-acoustic coupler
    ///
    /// # Arguments
    ///
    /// * `config`: Solver configuration
    ///
    /// # Returns
    ///
    /// Initialized coupler ready for simulation
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid (CFL violation, negative dimensions, etc.)
    pub fn new(config: ThermalAcousticConfig) -> KwaversResult<Self> {
        if config.nx < 3 || config.ny < 3 || config.nz < 3 {
            return Err(KwaversError::InvalidInput(
                "Grid dimensions must be at least 3".to_string(),
            ));
        }

        if config.dx <= 0.0 || config.dy <= 0.0 || config.dz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Grid spacing must be positive".to_string(),
            ));
        }

        if config.dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Time step must be positive".to_string(),
            ));
        }

        let max_spacing = config.dx.min(config.dy).min(config.dz);
        let cfl_acoustic = config.c_ref * config.dt / max_spacing;
        if cfl_acoustic > 0.3 {
            return Err(KwaversError::InvalidInput(format!(
                "CFL violation (acoustic): {:.3} > 0.3. Reduce dt or increase dx/dy/dz",
                cfl_acoustic
            )));
        }

        let cfl_thermal = config.alpha_thermal * config.dt / (max_spacing * max_spacing);
        if cfl_thermal > 0.25 {
            return Err(KwaversError::InvalidInput(format!(
                "CFL violation (thermal): {:.3} > 0.25. Reduce dt or increase dx/dy/dz",
                cfl_thermal
            )));
        }

        let nx = config.nx;
        let ny = config.ny;
        let nz = config.nz;

        let pressure = Array3::zeros((nx, ny, nz));
        let velocity_x = Array3::zeros((nx, ny, nz));
        let velocity_y = Array3::zeros((nx, ny, nz));
        let velocity_z = Array3::zeros((nx, ny, nz));
        let temperature = Array3::from_elem((nx, ny, nz), config.t_ref);

        let density = Array3::from_elem((nx, ny, nz), config.rho_ref);
        let sound_speed = Array3::from_elem((nx, ny, nz), config.c_ref);

        Ok(Self {
            config,
            pressure: pressure.clone(),
            velocity_x: velocity_x.clone(),
            velocity_y: velocity_y.clone(),
            velocity_z: velocity_z.clone(),
            temperature: temperature.clone(),
            pressure_prev: pressure,
            velocity_x_prev: velocity_x,
            velocity_y_prev: velocity_y,
            velocity_z_prev: velocity_z,
            temperature_prev: temperature,
            density,
            sound_speed,
            acoustic_heating: Array3::zeros((nx, ny, nz)),
            step_count: 0,
            total_time: 0.0,
        })
    }

    /// Execute one coupled time step
    ///
    /// # Algorithm
    ///
    /// 1. Save current state
    /// 2. Compute material properties from current temperature
    /// 3. Advance acoustic equations (pressure and velocities)
    /// 4. Compute acoustic heating from new pressure field
    /// 5. Advance thermal equation
    /// 6. Update material properties at new temperature
    ///
    /// # Returns
    ///
    /// Ok on success, error if divergence detected
    pub fn step(&mut self) -> KwaversResult<()> {
        self.pressure_prev.assign(&self.pressure);
        self.velocity_x_prev.assign(&self.velocity_x);
        self.velocity_y_prev.assign(&self.velocity_y);
        self.velocity_z_prev.assign(&self.velocity_z);
        self.temperature_prev.assign(&self.temperature);

        if self.config.enable_temperature_coupling {
            self.update_material_properties();
        }

        self.step_acoustic();

        self.compute_acoustic_heating();

        self.step_thermal();

        self.check_stability()?;

        self.step_count += 1;
        self.total_time += self.config.dt;

        Ok(())
    }

    /// Advance acoustic equations one time step
    ///
    /// Uses FDTD-like scheme:
    /// - ρ ∂u/∂t = -∇p
    /// - ∂p/∂t = -ρc² ∇·u
    fn step_acoustic(&mut self) {
        let dt = self.config.dt;
        let dx = self.config.dx;
        let dy = self.config.dy;
        let dz = self.config.dz;

        for k in 1..self.config.nz - 1 {
            for j in 1..self.config.ny - 1 {
                for i in 1..self.config.nx - 1 {
                    let rho = self.density[[i, j, k]];
                    if rho > 0.0 {
                        let dp_dx = (self.pressure_prev[[i + 1, j, k]]
                            - self.pressure_prev[[i - 1, j, k]])
                            / (2.0 * dx);
                        let dp_dy = (self.pressure_prev[[i, j + 1, k]]
                            - self.pressure_prev[[i, j - 1, k]])
                            / (2.0 * dy);
                        let dp_dz = (self.pressure_prev[[i, j, k + 1]]
                            - self.pressure_prev[[i, j, k - 1]])
                            / (2.0 * dz);

                        self.velocity_x[[i, j, k]] =
                            self.velocity_x_prev[[i, j, k]] - (dt / rho) * dp_dx;
                        self.velocity_y[[i, j, k]] =
                            self.velocity_y_prev[[i, j, k]] - (dt / rho) * dp_dy;
                        self.velocity_z[[i, j, k]] =
                            self.velocity_z_prev[[i, j, k]] - (dt / rho) * dp_dz;
                    }
                }
            }
        }

        for k in 1..self.config.nz - 1 {
            for j in 1..self.config.ny - 1 {
                for i in 1..self.config.nx - 1 {
                    let rho_c_sq = self.density[[i, j, k]]
                        * self.sound_speed[[i, j, k]]
                        * self.sound_speed[[i, j, k]];

                    let du_dx = (self.velocity_x[[i + 1, j, k]] - self.velocity_x[[i - 1, j, k]])
                        / (2.0 * dx);
                    let dv_dy = (self.velocity_y[[i, j + 1, k]] - self.velocity_y[[i, j - 1, k]])
                        / (2.0 * dy);
                    let dw_dz = (self.velocity_z[[i, j, k + 1]] - self.velocity_z[[i, j, k - 1]])
                        / (2.0 * dz);

                    let div_u = du_dx + dv_dy + dw_dz;

                    self.pressure[[i, j, k]] =
                        self.pressure_prev[[i, j, k]] - rho_c_sq * dt * div_u;
                }
            }
        }
    }

    /// Get current pressure field
    pub fn pressure(&self) -> &Array3<f64> {
        &self.pressure
    }

    /// Get current temperature field
    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }

    /// Get current acoustic heating
    pub fn acoustic_heating(&self) -> &Array3<f64> {
        &self.acoustic_heating
    }

    /// Get total simulation time
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Get step count
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}
