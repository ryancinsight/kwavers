use super::{ThermalAcousticConfig, ThermalAcousticCoupler};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;
use moirai_parallel::{enumerate_mut_with, Adaptive};

impl ThermalAcousticCoupler {
    /// Create new thermal-acoustic coupler with default configuration
    /// # Panics
    /// - Panics if `default config is valid`.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
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
                "Grid dimensions must be at least 3".to_owned(),
            ));
        }

        if config.dx <= 0.0 || config.dy <= 0.0 || config.dz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Grid spacing must be positive".to_owned(),
            ));
        }

        if config.dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Time step must be positive".to_owned(),
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

        // 3D explicit diffusion stability: α·dt·(1/dx² + 1/dy² + 1/dz²) ≤ 1/2
        // Worst case with max_spacing = min(dx,dy,dz): 3·α·dt/max_spacing² ≤ 1/2 → cfl ≤ 1/6
        let cfl_thermal = config.alpha_thermal * config.dt / (max_spacing * max_spacing);
        if cfl_thermal > 1.0 / 6.0 {
            return Err(KwaversError::InvalidInput(format!(
                "CFL violation (thermal): {:.4} > 1/6 ≈ 0.1667 (3D explicit diffusion limit). Reduce dt or increase dx/dy/dz",
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
        let temperature = Array3::from_elem([nx, ny, nz], config.t_ref);

        let density = Array3::from_elem([nx, ny, nz], config.rho_ref);
        let sound_speed = Array3::from_elem([nx, ny, nz], config.c_ref);

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
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
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

    /// Advance acoustic equations one time step.
    ///
    /// Explicit staggered FDTD: reads from `_prev` fields (no intra-step dependencies)
    /// → all velocity components and the pressure update are fully data-parallel.
    ///
    /// Velocity components are updated in three separate parallel passes to allow
    /// simultaneous mutable access to each field while sharing immutable borrows of
    /// `pressure_prev` and `density`.
    fn step_acoustic(&mut self) {
        let dt = self.config.dt;
        let dx = self.config.dx;
        let dy = self.config.dy;
        let dz = self.config.dz;
        let nx = self.config.nx;
        let ny = self.config.ny;
        let nz = self.config.nz;

        // --- Velocity update ---
        // ρ ∂u/∂t = −∇p   (reads pressure_prev, writes velocity_{x,y,z})
        {
            let pp = &self.pressure_prev;
            let dens = &self.density;
            let vxp = &self.velocity_x_prev;
            let velocity_x = &mut self.velocity_x;
            if let (Some(velocity_x), Some(pp), Some(dens), Some(vxp)) = (
                velocity_x.as_slice_mut(),
                pp.as_slice(),
                dens.as_slice(),
                vxp.as_slice(),
            ) {
                enumerate_mut_with::<Adaptive, _, _>(velocity_x, |index, vx| {
                    let (i, j, k) = Self::cell_indices(index, ny, nz);
                    if i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz - 1 {
                        return;
                    }
                    let rho = dens[index];
                    if rho > 0.0 {
                        let dp_dx = (pp[index + ny * nz] - pp[index - ny * nz]) / (2.0 * dx);
                        *vx = (dt / rho).mul_add(-dp_dx, vxp[index]);
                    }
                });
            } else {
                velocity_x
                    .indexed_iter_mut()
                    .expect("invariant: contiguous owned velocity_x array")
                    .for_each(|([i, j, k], vx)| {
                        if i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz - 1 {
                            return;
                        }
                        let rho = dens[[i, j, k]];
                        if rho > 0.0 {
                            let dp_dx = (pp[[i + 1, j, k]] - pp[[i - 1, j, k]]) / (2.0 * dx);
                            *vx = (dt / rho).mul_add(-dp_dx, vxp[[i, j, k]]);
                        }
                    });
            }
        }
        {
            let pp = &self.pressure_prev;
            let dens = &self.density;
            let vyp = &self.velocity_y_prev;
            let velocity_y = &mut self.velocity_y;
            if let (Some(velocity_y), Some(pp), Some(dens), Some(vyp)) = (
                velocity_y.as_slice_mut(),
                pp.as_slice(),
                dens.as_slice(),
                vyp.as_slice(),
            ) {
                enumerate_mut_with::<Adaptive, _, _>(velocity_y, |index, vy| {
                    let (i, j, k) = Self::cell_indices(index, ny, nz);
                    if i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz - 1 {
                        return;
                    }
                    let rho = dens[index];
                    if rho > 0.0 {
                        let dp_dy = (pp[index + nz] - pp[index - nz]) / (2.0 * dy);
                        *vy = (dt / rho).mul_add(-dp_dy, vyp[index]);
                    }
                });
            } else {
                velocity_y
                    .indexed_iter_mut()
                    .expect("invariant: contiguous owned velocity_y array")
                    .for_each(|([i, j, k], vy)| {
                        if i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz - 1 {
                            return;
                        }
                        let rho = dens[[i, j, k]];
                        if rho > 0.0 {
                            let dp_dy = (pp[[i, j + 1, k]] - pp[[i, j - 1, k]]) / (2.0 * dy);
                            *vy = (dt / rho).mul_add(-dp_dy, vyp[[i, j, k]]);
                        }
                    });
            }
        }
        {
            let pp = &self.pressure_prev;
            let dens = &self.density;
            let vzp = &self.velocity_z_prev;
            let velocity_z = &mut self.velocity_z;
            if let (Some(velocity_z), Some(pp), Some(dens), Some(vzp)) = (
                velocity_z.as_slice_mut(),
                pp.as_slice(),
                dens.as_slice(),
                vzp.as_slice(),
            ) {
                enumerate_mut_with::<Adaptive, _, _>(velocity_z, |index, vz| {
                    let (i, j, k) = Self::cell_indices(index, ny, nz);
                    if i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz - 1 {
                        return;
                    }
                    let rho = dens[index];
                    if rho > 0.0 {
                        let dp_dz = (pp[index + 1] - pp[index - 1]) / (2.0 * dz);
                        *vz = (dt / rho).mul_add(-dp_dz, vzp[index]);
                    }
                });
            } else {
                velocity_z
                    .indexed_iter_mut()
                    .expect("invariant: contiguous owned velocity_z array")
                    .for_each(|([i, j, k], vz)| {
                        if i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz - 1 {
                            return;
                        }
                        let rho = dens[[i, j, k]];
                        if rho > 0.0 {
                            let dp_dz = (pp[[i, j, k + 1]] - pp[[i, j, k - 1]]) / (2.0 * dz);
                            *vz = (dt / rho).mul_add(-dp_dz, vzp[[i, j, k]]);
                        }
                    });
            }
        }

        // --- Pressure update ---
        // ∂p/∂t = −ρc² ∇·u  (reads velocity just written above + pressure_prev)
        {
            let vx = &self.velocity_x;
            let vy = &self.velocity_y;
            let vz = &self.velocity_z;
            let dens = &self.density;
            let ss = &self.sound_speed;
            let pp = &self.pressure_prev;
            let pressure = &mut self.pressure;
            if let (Some(pressure), Some(vx), Some(vy), Some(vz), Some(dens), Some(ss), Some(pp)) = (
                pressure.as_slice_mut(),
                vx.as_slice(),
                vy.as_slice(),
                vz.as_slice(),
                dens.as_slice(),
                ss.as_slice(),
                pp.as_slice(),
            ) {
                enumerate_mut_with::<Adaptive, _, _>(pressure, |index, p| {
                    let (i, j, k) = Self::cell_indices(index, ny, nz);
                    if i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz - 1 {
                        return;
                    }
                    let rho_c_sq = dens[index] * ss[index] * ss[index];
                    let du_dx = (vx[index + ny * nz] - vx[index - ny * nz]) / (2.0 * dx);
                    let dv_dy = (vy[index + nz] - vy[index - nz]) / (2.0 * dy);
                    let dw_dz = (vz[index + 1] - vz[index - 1]) / (2.0 * dz);
                    let div_u = du_dx + dv_dy + dw_dz;
                    *p = (rho_c_sq * dt).mul_add(-div_u, pp[index]);
                });
            } else {
                pressure
                    .indexed_iter_mut()
                    .expect("invariant: contiguous owned pressure array")
                    .for_each(|([i, j, k], p)| {
                        if i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1 || k == 0 || k >= nz - 1 {
                            return;
                        }
                        let rho_c_sq = dens[[i, j, k]] * ss[[i, j, k]] * ss[[i, j, k]];
                        let du_dx = (vx[[i + 1, j, k]] - vx[[i - 1, j, k]]) / (2.0 * dx);
                        let dv_dy = (vy[[i, j + 1, k]] - vy[[i, j - 1, k]]) / (2.0 * dy);
                        let dw_dz = (vz[[i, j, k + 1]] - vz[[i, j, k - 1]]) / (2.0 * dz);
                        let div_u = du_dx + dv_dy + dw_dz;
                        *p = (rho_c_sq * dt).mul_add(-div_u, pp[[i, j, k]]);
                    });
            }
        }
    }

    /// Get current pressure field
    #[must_use]
    pub fn pressure(&self) -> &Array3<f64> {
        &self.pressure
    }

    /// Get current temperature field
    #[must_use]
    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }

    /// Get current acoustic heating
    #[must_use]
    pub fn acoustic_heating(&self) -> &Array3<f64> {
        &self.acoustic_heating
    }

    /// Get total simulation time
    #[must_use]
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Get step count
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }
}
