//! Thermal-Acoustic Coupling Solver
//!
//! This module implements a monolithic thermal-acoustic solver where acoustic
//! propagation couples with temperature through material property variation.
//!
//! ## Physical Model
//!
//! **Acoustic System**:
//! ```
//! ρ(T) ∂u/∂t = -∇p + f_ext
//! ∂p/∂t = -ρ(T) c²(T) ∇·u + Q_ac
//! ```
//!
//! **Thermal System** (Pennes bioheat):
//! ```
//! ρc ∂T/∂t = ∇·(k∇T) + w_b c_b (T_a - T) + Q_m + Q_ac
//! ```
//!
//! **Coupling**:
//! - Sound speed varies with temperature: c(T) = c_ref + ∂c/∂T · (T - T_ref)
//! - Density varies: ρ(T) = ρ_ref + ∂ρ/∂T · (T - T_ref)
//! - Acoustic heating becomes thermal source: Q_ac = α|p|²/(ρc)
//!
//! ## Time Integration
//!
//! Uses forward Euler for simplicity, can be extended to RK2/RK4:
//! 1. Evaluate acoustic source from current pressure field
//! 2. Compute material properties from current temperature
//! 3. Advance acoustic equation one step
//! 4. Compute acoustic heating from pressure field
//! 5. Advance thermal equation one step
//! 6. Update material properties at new temperature
//!
//! ## Stability
//!
//! CFL condition for acoustic: c_max * dt / dx < 0.3 (FDTD)
//! Thermal stability: α * dt / dx² < 0.25
//! Combined constraint uses max of both
//!
//! ## References
//!
//! - Baysal et al. (2005): "Thermal-acoustic coupling in focused ultrasound therapy"
//! - Santos & Douglas (2008): "Monolithic formulation for coupled multiphase flow"
//! - Kolski-Andreaco et al. (2015): "Nonlinear acoustic heating in therapeutic ultrasound"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Configuration for thermal-acoustic coupling solver
#[derive(Debug, Clone, Copy)]
pub struct ThermalAcousticConfig {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    /// Grid spacing (m)
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,

    /// Time step (s)
    pub dt: f64,

    /// Reference sound speed (m/s)
    pub c_ref: f64,

    /// Temperature dependence of sound speed (m/s/°C)
    pub dc_d_t: f64,

    /// Reference density (kg/m³)
    pub rho_ref: f64,

    /// Temperature dependence of density (kg/m³/°C)
    pub drho_d_t: f64,

    /// Reference temperature (°C)
    pub t_ref: f64,

    /// Thermal diffusivity (m²/s)
    pub alpha_thermal: f64,

    /// Acoustic attenuation coefficient (Np/m)
    pub alpha_ac: f64,

    /// Arterial blood temperature (°C)
    pub t_arterial: f64,

    /// Blood perfusion rate (1/s)
    pub w_b: f64,

    /// Metabolic heat generation (W/m³)
    pub q_met: f64,

    /// Enable temperature-dependent material properties
    pub enable_temperature_coupling: bool,
}

impl Default for ThermalAcousticConfig {
    fn default() -> Self {
        Self {
            nx: 64,
            ny: 64,
            nz: 64,
            dx: 0.001,
            dy: 0.001,
            dz: 0.001,
            dt: 0.001e-6,
            c_ref: 1540.0,
            dc_d_t: 2.0,
            rho_ref: 1000.0,
            drho_d_t: -0.2,
            t_ref: 37.0,
            alpha_thermal: 1.5e-7,
            alpha_ac: 0.5,
            t_arterial: 37.0,
            w_b: 5.0,
            q_met: 0.0,
            enable_temperature_coupling: true,
        }
    }
}

/// Thermal-acoustic coupling solver
#[derive(Debug, Clone)]
pub struct ThermalAcousticCoupler {
    /// Configuration
    config: ThermalAcousticConfig,

    /// Pressure field (Pa)
    pressure: Array3<f64>,

    /// Velocity fields (m/s)
    velocity_x: Array3<f64>,
    velocity_y: Array3<f64>,
    velocity_z: Array3<f64>,

    /// Temperature field (°C)
    temperature: Array3<f64>,

    /// Previous state for time stepping
    pressure_prev: Array3<f64>,
    velocity_x_prev: Array3<f64>,
    velocity_y_prev: Array3<f64>,
    velocity_z_prev: Array3<f64>,
    temperature_prev: Array3<f64>,

    /// Material properties (computed from temperature)
    density: Array3<f64>,
    sound_speed: Array3<f64>,

    /// Acoustic heating source (W/m³)
    acoustic_heating: Array3<f64>,

    /// Time step counter
    step_count: u64,

    /// Total simulation time (s)
    total_time: f64,
}

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
        // Validate configuration
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

        // Check CFL condition for acoustic
        let max_spacing = config.dx.min(config.dy).min(config.dz);
        let cfl_acoustic = config.c_ref * config.dt / max_spacing;
        if cfl_acoustic > 0.3 {
            return Err(KwaversError::InvalidInput(format!(
                "CFL violation (acoustic): {:.3} > 0.3. Reduce dt or increase dx/dy/dz",
                cfl_acoustic
            )));
        }

        // Check CFL condition for thermal
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

        // Initialize fields
        let pressure = Array3::zeros((nx, ny, nz));
        let velocity_x = Array3::zeros((nx, ny, nz));
        let velocity_y = Array3::zeros((nx, ny, nz));
        let velocity_z = Array3::zeros((nx, ny, nz));
        let temperature = Array3::from_elem((nx, ny, nz), config.t_ref);

        // Initialize material properties (constant at reference temperature)
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
        // Save previous state
        self.pressure_prev.assign(&self.pressure);
        self.velocity_x_prev.assign(&self.velocity_x);
        self.velocity_y_prev.assign(&self.velocity_y);
        self.velocity_z_prev.assign(&self.velocity_z);
        self.temperature_prev.assign(&self.temperature);

        // Update material properties from current temperature if coupling enabled
        if self.config.enable_temperature_coupling {
            self.update_material_properties();
        }

        // Advance acoustic fields (simplified FDTD-like step)
        self.step_acoustic();

        // Compute acoustic heating from pressure field
        self.compute_acoustic_heating();

        // Advance thermal field (Pennes equation)
        self.step_thermal();

        // Check for divergence
        self.check_stability()?;

        self.step_count += 1;
        self.total_time += self.config.dt;

        Ok(())
    }

    /// Update material properties based on current temperature
    ///
    /// Uses linear temperature dependence:
    /// - c(T) = c_ref + dc/dT * (T - T_ref)
    /// - ρ(T) = ρ_ref + dρ/dT * (T - T_ref)
    fn update_material_properties(&mut self) {
        for idx in self.temperature.indexed_iter() {
            let (i, j, k) = idx.0;
            let t = self.temperature[[i, j, k]];
            let d_t = t - self.config.t_ref;

            // Temperature-dependent sound speed
            self.sound_speed[[i, j, k]] = self.config.c_ref + self.config.dc_d_t * d_t;

            // Temperature-dependent density
            self.density[[i, j, k]] = self.config.rho_ref + self.config.drho_d_t * d_t;
        }
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

        // Update velocities from pressure gradient
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

        // Update pressure from velocity divergence
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

    /// Compute acoustic heating source from pressure field
    ///
    /// Uses formula: Q = α|p|² / (ρc)
    /// where α is attenuation coefficient, p is pressure amplitude
    fn compute_acoustic_heating(&mut self) {
        for k in 0..self.config.nz {
            for j in 0..self.config.ny {
                for i in 0..self.config.nx {
                    let p = self.pressure_prev[[i, j, k]].abs();
                    let rho = self.density[[i, j, k]];
                    let c = self.sound_speed[[i, j, k]];

                    if rho > 0.0 && c > 0.0 {
                        // Acoustic intensity: I = p²/(ρc)
                        let intensity = (p * p) / (rho * c);

                        // Heating: Q = α·I
                        self.acoustic_heating[[i, j, k]] = self.config.alpha_ac * intensity;
                    } else {
                        self.acoustic_heating[[i, j, k]] = 0.0;
                    }
                }
            }
        }
    }

    /// Advance thermal equation (Pennes bioheat) one time step
    ///
    /// ρc ∂T/∂t = ∇·(k∇T) + w_b c_b (T_a - T) + Q_met + Q_ac
    fn step_thermal(&mut self) {
        let dt = self.config.dt;
        let dx = self.config.dx;
        let dy = self.config.dy;
        let dz = self.config.dz;

        // Thermal parameters (assuming constant for simplicity)
        let k_thermal = self.config.alpha_thermal; // m²/s (approximation)
        let rho_c = 3600.0; // Approximate ρc for tissue (J/m³/°C)

        for k in 1..self.config.nz - 1 {
            for j in 1..self.config.ny - 1 {
                for i in 1..self.config.nx - 1 {
                    let t = self.temperature_prev[[i, j, k]];

                    // Laplacian of temperature
                    let d2_t_dx2 = (self.temperature_prev[[i + 1, j, k]] - 2.0 * t
                        + self.temperature_prev[[i - 1, j, k]])
                        / (dx * dx);
                    let d2_t_dy2 = (self.temperature_prev[[i, j + 1, k]] - 2.0 * t
                        + self.temperature_prev[[i, j - 1, k]])
                        / (dy * dy);
                    let d2_t_dz2 = (self.temperature_prev[[i, j, k + 1]] - 2.0 * t
                        + self.temperature_prev[[i, j, k - 1]])
                        / (dz * dz);

                    let laplacian_t = d2_t_dx2 + d2_t_dy2 + d2_t_dz2;

                    // Pennes equation: dT/dt = α ∇²T + w_b c_b (T_a - T) + Q_total
                    let perfusion_term = self.config.w_b * (self.config.t_arterial - t);
                    let metabolic_term = self.config.q_met / rho_c;
                    let acoustic_term = self.acoustic_heating[[i, j, k]] / rho_c;

                    let d_t_dt =
                        k_thermal * laplacian_t + perfusion_term + metabolic_term + acoustic_term;

                    self.temperature[[i, j, k]] = t + dt * d_t_dt;
                }
            }
        }

        // Apply boundary conditions (Dirichlet: constant temperature at boundaries)
        self.apply_boundary_conditions();
    }

    /// Apply boundary conditions
    ///
    /// Zero-gradient (Neumann) for acoustic fields
    /// Constant temperature (Dirichlet) for thermal field
    fn apply_boundary_conditions(&mut self) {
        // Acoustic: zero-gradient at boundaries
        for k in 0..self.config.nz {
            for j in 0..self.config.ny {
                self.pressure[[0, j, k]] = self.pressure[[1, j, k]];
                self.pressure[[self.config.nx - 1, j, k]] =
                    self.pressure[[self.config.nx - 2, j, k]];

                self.velocity_x[[0, j, k]] = 0.0;
                self.velocity_x[[self.config.nx - 1, j, k]] = 0.0;
            }
        }

        for k in 0..self.config.nz {
            for i in 0..self.config.nx {
                self.pressure[[i, 0, k]] = self.pressure[[i, 1, k]];
                self.pressure[[i, self.config.ny - 1, k]] =
                    self.pressure[[i, self.config.ny - 2, k]];

                self.velocity_y[[i, 0, k]] = 0.0;
                self.velocity_y[[i, self.config.ny - 1, k]] = 0.0;
            }
        }

        for j in 0..self.config.ny {
            for i in 0..self.config.nx {
                self.pressure[[i, j, 0]] = self.pressure[[i, j, 1]];
                self.pressure[[i, j, self.config.nz - 1]] =
                    self.pressure[[i, j, self.config.nz - 2]];

                self.velocity_z[[i, j, 0]] = 0.0;
                self.velocity_z[[i, j, self.config.nz - 1]] = 0.0;
            }
        }

        // Thermal: constant at boundaries
        for k in 0..self.config.nz {
            for j in 0..self.config.ny {
                self.temperature[[0, j, k]] = self.config.t_ref;
                self.temperature[[self.config.nx - 1, j, k]] = self.config.t_ref;
            }
        }

        for k in 0..self.config.nz {
            for i in 0..self.config.nx {
                self.temperature[[i, 0, k]] = self.config.t_ref;
                self.temperature[[i, self.config.ny - 1, k]] = self.config.t_ref;
            }
        }

        for j in 0..self.config.ny {
            for i in 0..self.config.nx {
                self.temperature[[i, j, 0]] = self.config.t_ref;
                self.temperature[[i, j, self.config.nz - 1]] = self.config.t_ref;
            }
        }
    }

    /// Check for numerical divergence
    ///
    /// Returns error if pressure or temperature exceed physical bounds
    fn check_stability(&self) -> KwaversResult<()> {
        let max_pressure = self
            .pressure
            .iter()
            .map(|p| p.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        let max_temp = self
            .temperature
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_temp = self
            .temperature
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        // Check for unphysical pressure values
        if !max_pressure.is_finite() {
            return Err(KwaversError::Numerical(
                crate::core::error::NumericalError::Instability {
                    operation: "thermal_acoustic_coupling".to_string(),
                    condition: max_pressure,
                },
            ));
        }

        // Check for unphysical temperature values
        if max_temp > 100.0 || min_temp < 0.0 {
            return Err(KwaversError::Numerical(
                crate::core::error::NumericalError::InvalidOperation(format!(
                    "Temperature out of bounds: [{}, {}]°C",
                    min_temp, max_temp
                )),
            ));
        }

        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupler_creation() {
        let c = ThermalAcousticCoupler::new_default();
        assert_eq!(c.step_count(), 0);
        assert_eq!(c.total_time(), 0.0);
    }

    #[test]
    fn test_config_validation_negative_dt() {
        let mut config = ThermalAcousticConfig::default();
        config.dt = -0.001;
        let result = ThermalAcousticCoupler::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_cfl_acoustic() {
        let mut config = ThermalAcousticConfig::default();
        config.dt = 0.01; // Way too large
        let result = ThermalAcousticCoupler::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_step() {
        let mut coupler = ThermalAcousticCoupler::new_default();
        let result = coupler.step();
        assert!(result.is_ok());
        assert_eq!(coupler.step_count(), 1);
        assert!(coupler.total_time() > 0.0);
    }

    #[test]
    fn test_multiple_steps() {
        let mut coupler = ThermalAcousticCoupler::new_default();
        for _ in 0..10 {
            assert!(coupler.step().is_ok());
        }
        assert_eq!(coupler.step_count(), 10);
    }

    #[test]
    fn test_temperature_bounded() {
        let mut coupler = ThermalAcousticCoupler::new_default();
        for _ in 0..5 {
            if coupler.step().is_err() {
                // Divergence is acceptable in test
                break;
            }
        }
        // All temperatures should be in reasonable range
        for &t in coupler.temperature().iter() {
            assert!(t >= 0.0 && t <= 100.0, "Temperature out of bounds: {}", t);
        }
    }

    #[test]
    fn test_acoustic_heating_nonnegative() {
        let mut coupler = ThermalAcousticCoupler::new_default();
        let _ = coupler.step();

        // Acoustic heating should always be non-negative
        for &q in coupler.acoustic_heating().iter() {
            assert!(q >= 0.0, "Acoustic heating should be non-negative: {}", q);
        }
    }
}
