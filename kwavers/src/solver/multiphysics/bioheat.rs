// Bioheat transfer solver using Pennes equation
//
// This module implements the Pennes bioheat equation for simulating thermal
// transport in biological tissue during therapeutic ultrasound.
//
// ## Temperature-Dependent Acoustic Absorption
//
// Bamber & Hill (1979) showed that tissue absorption increases linearly with
// temperature:
//
//   α(T) = α₀ · (1 + α_T · (T − T_ref))
//
// where α_T ≈ 0.015 K⁻¹ and T_ref = 310.15 K (body temperature).
//
// The acoustic heat source from tissue absorption is:
//
//   Q_abs[i,j,k] = α(T) · p²_rms[i,j,k] / (ρ · c)
//
// Reference: Bamber, J.C. & Hill, C.R. (1979). Ultrasound Med Biol 5(2), 149–157.
// DOI: 10.1016/0301-5629(79)90083-8

/// Celsius to Kelvin offset
const C_TO_K: f64 = 273.15;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use crate::domain::medium::traits::Medium;
use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// Temperature-dependent acoustic absorption model (Bamber & Hill 1979)
///
/// ## Model
///
/// ```text
/// α(T) = α₀ · (1 + α_T · (T − T_ref))
/// ```
///
/// ## Parameters
///
/// - `alpha_0`: Reference absorption coefficient [Np/m] at `t_ref`
/// - `alpha_t`: Temperature coefficient of absorption [1/K], default 0.015 K⁻¹
/// - `t_ref`:   Reference temperature [K], default 310.15 K (body temperature)
///
/// ## Reference
///
/// Bamber, J.C. & Hill, C.R. (1979). "Acoustic properties of normal and
/// cancerous human liver — I. Dependence on pathological condition."
/// Ultrasound Med Biol 5(2), 149–157. DOI: 10.1016/0301-5629(79)90083-8
#[derive(Debug, Clone)]
pub struct AbsorptionModel {
    /// Reference absorption coefficient [Np/m]
    pub alpha_0: f64,
    /// Temperature coefficient of absorption [1/K]
    pub alpha_t: f64,
    /// Reference temperature [K]
    pub t_ref: f64,
}

impl AbsorptionModel {
    /// Create with default tissue parameters (Bamber & Hill 1979)
    #[must_use]
    pub fn tissue(alpha_0: f64) -> Self {
        Self {
            alpha_0,
            alpha_t: 0.015,
            t_ref: 310.15,
        }
    }

    /// Absorption coefficient at temperature T [K]
    ///
    /// α(T) = α₀ · (1 + α_t · (T − T_ref))
    #[must_use]
    pub fn alpha_at(&self, temperature: f64) -> f64 {
        self.alpha_0 * (1.0 + self.alpha_t * (temperature - self.t_ref))
    }
}

impl Default for AbsorptionModel {
    fn default() -> Self {
        Self::tissue(0.0)
    }
}

/// Blood properties used in Pennes equation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloodProperties {
    pub density: f64,
    pub specific_heat: f64,
    pub arterial_temperature: f64,
    pub base_perfusion: f64,
    pub perfusion_factor: f64,
}

impl Default for BloodProperties {
    fn default() -> Self {
        Self {
            density: 1060.0,
            specific_heat: 3640.0,
            arterial_temperature: 37.0 + C_TO_K,
            base_perfusion: 1.0,
            perfusion_factor: 0.01,
        }
    }
}

/// Tissue thermal properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TissueProperties {
    pub density: f64,
    pub specific_heat: f64,
    pub conductivity: f64,
    pub metabolic_rate: f64,
}

impl Default for TissueProperties {
    fn default() -> Self {
        Self {
            density: 1060.0,
            specific_heat: 3600.0,
            conductivity: 0.5,
            metabolic_rate: 420.0,
        }
    }
}

/// Bioheat solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioheatConfig {
    pub tissue: TissueProperties,
    pub blood: BloodProperties,
    pub temperature_dependent: bool,
    pub acoustic_coupling: bool,
    pub compute_thermal_dose: bool,
    pub damage_threshold: f64,
    pub dt: f64,
    pub dx: f64,
}

impl Default for BioheatConfig {
    fn default() -> Self {
        Self {
            tissue: TissueProperties::default(),
            blood: BloodProperties::default(),
            temperature_dependent: true,
            acoustic_coupling: true,
            compute_thermal_dose: true,
            damage_threshold: 240.0,
            dt: 1e-3,
            dx: 0.1e-3,
        }
    }
}

/// Thermal dose accumulator using CEM43 metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalDose {
    pub cem43: Array3<f64>,
    pub t_max: Array3<f64>,
    pub time_above_43: Array3<f64>,
    shape: (usize, usize, usize),
}

impl ThermalDose {
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            cem43: Array3::zeros((nx, ny, nz)),
            t_max: Array3::from_elem((nx, ny, nz), 0.0),
            time_above_43: Array3::zeros((nx, ny, nz)),
            shape: (nx, ny, nz),
        }
    }

    pub fn update(&mut self, temperature: &Array3<f64>, dt: f64) {
        let r_above = 0.5;
        let r_below = 0.25;
        let t43 = 43.0 + C_TO_K;
        let dt_minutes = dt / 60.0;

        for ((idx, &t), cem43_mut) in temperature.indexed_iter().zip(self.cem43.iter_mut()) {
            let (i, j, k) = idx;
            if t > self.t_max[[i, j, k]] {
                self.t_max[[i, j, k]] = t;
            }
            let r: f64 = if t > t43 { r_above } else { r_below };
            let exponent = 43.0 - (t - C_TO_K);
            *cem43_mut += dt_minutes * r.powf(exponent);
            if t > t43 {
                self.time_above_43[[i, j, k]] += dt;
            }
        }
    }

    #[must_use]
    pub fn damage_mask(&self, threshold: f64) -> Array3<bool> {
        self.cem43.mapv(|c| c > threshold)
    }
}

/// Bioheat transfer solver state
#[derive(Debug)]
pub struct BioheatSolver {
    config: BioheatConfig,
    temperature: Array3<f64>,
    temperature_prev: Array3<f64>,
    heat_source: Array3<f64>,
    thermal_dose: ThermalDose,
    grid: Grid,
    /// Optional temperature-dependent absorption model
    absorption_model: Option<AbsorptionModel>,
    /// Optional acoustic pressure field for computing Q_abs
    acoustic_pressure: Option<Array3<f64>>,
    /// Medium density for acoustic heat source [kg/m³]
    acoustic_density: f64,
    /// Medium sound speed for acoustic heat source [m/s]
    acoustic_sound_speed: f64,
}

impl BioheatSolver {
    pub fn new(config: BioheatConfig, grid: Grid) -> KwaversResult<Self> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        if config.dt <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!("dt = {} must be positive", config.dt),
                },
            ));
        }

        if config.dx <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!("dx = {} must be positive", config.dx),
                },
            ));
        }

        let thermal_diffusivity =
            config.tissue.conductivity / (config.tissue.density * config.tissue.specific_heat);
        let max_dt = config.dx * config.dx / (6.0 * thermal_diffusivity);
        if config.dt > max_dt {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "dt = {} must be < {:.3e} for thermal stability",
                        config.dt, max_dt
                    ),
                },
            ));
        }

        let thermal_dose = ThermalDose::new(nx, ny, nz);

        Ok(Self {
            config,
            temperature: Array3::from_elem((nx, ny, nz), 37.0 + C_TO_K),
            temperature_prev: Array3::from_elem((nx, ny, nz), 37.0 + C_TO_K),
            heat_source: Array3::zeros((nx, ny, nz)),
            thermal_dose,
            grid,
            absorption_model: None,
            acoustic_pressure: None,
            acoustic_density: 1000.0,
            acoustic_sound_speed: 1500.0,
        })
    }

    pub fn set_initial_temperature(&mut self, temp: Array3<f64>) {
        if temp.shape() == self.temperature.shape() {
            self.temperature = temp.clone();
            self.temperature_prev = temp;
        }
    }

    pub fn set_heat_source(&mut self, source: Array3<f64>) {
        if source.shape() == self.heat_source.shape() {
            self.heat_source = source;
        }
    }

    /// Set temperature-dependent acoustic absorption model
    pub fn set_absorption_model(&mut self, model: AbsorptionModel) {
        self.absorption_model = Some(model);
    }

    /// Set acoustic pressure field for Q_abs computation in step()
    ///
    /// When `config.temperature_dependent = true` and this pressure field is set,
    /// `step()` will recompute Q_abs = α(T) · p² / (ρc) at each cell.
    pub fn set_acoustic_pressure_field(
        &mut self,
        pressure: Array3<f64>,
        density: f64,
        sound_speed: f64,
    ) {
        if pressure.shape() == self.heat_source.shape() {
            self.acoustic_pressure = Some(pressure);
            self.acoustic_density = density;
            self.acoustic_sound_speed = sound_speed;
        }
    }

    /// Compute acoustic heat source Q_abs from pressure field using temperature-dependent α(T)
    ///
    /// ## Formula (Bamber & Hill 1979)
    ///
    /// ```text
    /// Q_abs[i,j,k] = α(T[i,j,k]) · p²[i,j,k] / (ρ · c)
    /// ```
    ///
    /// where α(T) = α₀ · (1 + α_T · (T − T_ref)).
    ///
    /// ## Arguments
    ///
    /// * `pressure`    — RMS acoustic pressure field [Pa]
    /// * `temperature` — Temperature field [K]
    /// * `absorption`  — Absorption model (Bamber & Hill 1979)
    /// * `density`     — Medium density [kg/m³]
    /// * `sound_speed` — Medium sound speed [m/s]
    ///
    /// ## Returns
    ///
    /// Volumetric heat source Q [W/m³]
    #[must_use]
    pub fn compute_acoustic_heat_source(
        &self,
        pressure: &Array3<f64>,
        temperature: &Array3<f64>,
        absorption: &AbsorptionModel,
        density: f64,
        sound_speed: f64,
    ) -> Array3<f64> {
        let rho_c = density * sound_speed;
        ndarray::Zip::from(pressure)
            .and(temperature)
            .map_collect(|&p, &t| absorption.alpha_at(t) * p * p / rho_c)
    }

    #[must_use]
    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }

    #[must_use]
    pub fn thermal_dose(&self) -> &ThermalDose {
        &self.thermal_dose
    }

    pub fn step(&mut self) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let dt = self.config.dt;
        let dx2 = self.config.dx * self.config.dx;

        let thermal_conductivity = self.config.tissue.conductivity;
        let rho_c = self.config.tissue.density * self.config.tissue.specific_heat;
        let wb_cb = self.config.blood.base_perfusion
            * self.config.blood.density
            * self.config.blood.specific_heat;
        let ta = self.config.blood.arterial_temperature;
        let qm = self.config.tissue.metabolic_rate;

        // Pre-compute temperature-dependent acoustic heat source Q_abs[i,j,k]
        // when `temperature_dependent = true` and acoustic fields are available.
        // Q_abs = α(T) · p² / (ρc)  — Bamber & Hill (1979)
        let q_abs: Option<Array3<f64>> = if self.config.temperature_dependent {
            if let (Some(ref pressure), Some(ref model)) =
                (&self.acoustic_pressure, &self.absorption_model)
            {
                let rho_c_acoustic = self.acoustic_density * self.acoustic_sound_speed;
                let q = ndarray::Zip::from(pressure)
                    .and(&self.temperature)
                    .map_collect(|&p, &t| model.alpha_at(t) * p * p / rho_c_acoustic);
                Some(q)
            } else {
                None
            }
        } else {
            None
        };

        std::mem::swap(&mut self.temperature, &mut self.temperature_prev);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for kk in 1..nz - 1 {
                    let laplacian = (self.temperature_prev[[i + 1, j, kk]]
                        + self.temperature_prev[[i - 1, j, kk]]
                        + self.temperature_prev[[i, j + 1, kk]]
                        + self.temperature_prev[[i, j - 1, kk]]
                        + self.temperature_prev[[i, j, kk + 1]]
                        + self.temperature_prev[[i, j, kk - 1]]
                        - 6.0 * self.temperature_prev[[i, j, kk]])
                        / dx2;

                    let perfusion = wb_cb * (ta - self.temperature_prev[[i, j, kk]]);
                    // Acoustic heat source: externally-set Q + temperature-dependent Q_abs
                    let q_acoustic = q_abs.as_ref().map_or(0.0, |q| q[[i, j, kk]]);
                    let q_total = qm + self.heat_source[[i, j, kk]] + q_acoustic;

                    self.temperature[[i, j, kk]] = self.temperature_prev[[i, j, kk]]
                        + (dt / rho_c) * (thermal_conductivity * laplacian + perfusion + q_total);
                }
            }
        }

        self.apply_boundary_conditions();

        if self.config.compute_thermal_dose {
            self.thermal_dose.update(&self.temperature, dt);
        }
    }

    fn apply_boundary_conditions(&mut self) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let t_boundary = 37.0 + C_TO_K;

        for j in 0..ny {
            for k in 0..nz {
                self.temperature[[0, j, k]] = t_boundary;
                self.temperature[[nx - 1, j, k]] = t_boundary;
            }
        }

        for i in 0..nx {
            for k in 0..nz {
                self.temperature[[i, 0, k]] = t_boundary;
                self.temperature[[i, ny - 1, k]] = t_boundary;
            }
        }

        for i in 0..nx {
            for j in 0..ny {
                self.temperature[[i, j, 0]] = t_boundary;
                self.temperature[[i, j, nz - 1]] = t_boundary;
            }
        }
    }

    pub fn simulate(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            self.step();
        }
    }

    #[must_use]
    pub fn damage_mask(&self) -> Array3<bool> {
        self.thermal_dose.damage_mask(self.config.damage_threshold)
    }

    #[must_use]
    pub fn max_temperature(&self) -> f64 {
        self.temperature.iter().cloned().fold(f64::NAN, f64::max)
    }

    #[must_use]
    pub fn min_temperature(&self) -> f64 {
        self.temperature.iter().cloned().fold(f64::NAN, f64::min)
    }
}

pub fn bioheat_config_from_medium<M: Medium>(
    medium: &M,
    grid: &Grid,
    dt: f64,
) -> KwaversResult<BioheatConfig> {
    let density = medium.density(0, 0, 0);
    let _sound_speed = medium.max_sound_speed();

    let tissue = TissueProperties {
        density,
        specific_heat: 3600.0,
        conductivity: 0.5,
        metabolic_rate: 420.0,
    };

    let config = BioheatConfig {
        tissue,
        blood: BloodProperties::default(),
        temperature_dependent: true,
        acoustic_coupling: true,
        compute_thermal_dose: true,
        damage_threshold: 240.0,
        dt,
        dx: grid.dx,
    };

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pennes_steady_state() {
        let config = BioheatConfig::default();
        let grid = Grid::new(11, 11, 11, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let mut solver = BioheatSolver::new(config, grid).unwrap();

        solver.set_initial_temperature(Array3::from_elem((11, 11, 11), 37.0 + C_TO_K));

        solver.simulate(10000);

        let t_avg = solver.temperature().iter().sum::<f64>() / (11.0 * 11.0 * 11.0);
        let t_arterial = 37.0 + C_TO_K;

        assert!(
            (t_avg - t_arterial).abs() < 0.1,
            "Temperature should approach arterial temperature"
        );
    }

    #[test]
    fn test_thermal_dose_accumulation() {
        let config = BioheatConfig::default();
        let grid = Grid::new(5, 5, 5, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let mut solver = BioheatSolver::new(config, grid).unwrap();

        solver.set_initial_temperature(Array3::from_elem((5, 5, 5), 50.0 + C_TO_K));

        solver.simulate(60000);

        let max_cem43 = solver
            .thermal_dose
            .cem43
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);

        assert!(max_cem43 > 0.0, "Thermal dose should accumulate");
    }

    #[test]
    fn test_perfusion_stabilization() {
        let mut config = BioheatConfig::default();
        config.blood.perfusion_factor = 0.02;
        config.dt = 1e-4;

        let grid = Grid::new(7, 7, 7, 0.5e-3, 0.5e-3, 0.5e-3).unwrap();
        let mut solver = BioheatSolver::new(config, grid).unwrap();

        let mut source = Array3::zeros((7, 7, 7));
        source[[3, 3, 3]] = 1e7;
        solver.set_heat_source(source);

        solver.simulate(1000);

        let t_max = solver.max_temperature();
        assert!(t_max < 60.0 + C_TO_K, "Temperature should stabilize");
    }

    #[test]
    fn test_damage_threshold() {
        // Sustain T_center = 60 °C with a continuous heat source.
        // For a 3×3×3 grid (dx = 0.1 mm) surrounded by 37 °C boundaries:
        //   Q_source = (6k/dx² + ω_b ρ_b c_b) × ΔT ≈ 7e9 W/m³ (see bioheat.rs derivation).
        // At 60 °C the CEM43 rate ≈ 2.18 CEM43-min/step (dt = 1e-3 s).
        // 500 steps → ~1090 CEM43-min >> threshold (240 min) → mask must be true.
        let config = BioheatConfig::default();
        let grid = Grid::new(3, 3, 3, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let mut solver = BioheatSolver::new(config, grid).unwrap();

        let mut temp = Array3::from_elem((3, 3, 3), 37.0 + C_TO_K);
        temp[[1, 1, 1]] = 60.0 + C_TO_K;
        solver.set_initial_temperature(temp);

        // Sustained heat source balances conduction + perfusion losses at 60 °C.
        let mut source = Array3::zeros((3, 3, 3));
        source[[1, 1, 1]] = 8e9_f64; // slightly above balance point ≈ 7e9 W/m³
        solver.set_heat_source(source);

        solver.simulate(500);

        let mask = solver.damage_mask();
        assert!(mask[[1, 1, 1]], "Center should be damaged");
    }

    /// Validate the discrete Pennes FD operator via the Method of Manufactured Solutions.
    ///
    /// ## Manufactured Solution
    ///
    /// ```text
    /// T_mms(i,j,k) = T_a + A · sin(πi/N) · sin(πj/N) · sin(πk/N)
    /// ```
    ///
    /// This vanishes at all six boundary faces (sin = 0), matching the solver's
    /// Dirichlet condition T_boundary = T_a = 37 °C exactly.
    ///
    /// ## Discrete Eigenvalue
    ///
    /// For the 7-point FD Laplacian acting on sin(πi/N):
    ///
    /// ```text
    /// ∇²ₕ sin(πi/N) = Λ_fd · sin(πi/N)
    /// Λ_fd = 6 · (cos(π/N) − 1) / dx²    [m⁻²]
    /// ```
    ///
    /// ## Manufactured Source
    ///
    /// Setting ∂T/∂t = 0 in the discrete Pennes equation:
    ///
    /// ```text
    /// k · Λ_fd · (T_mms − T_a) − W_b · (T_mms − T_a) + Q_mms = 0
    /// ⟹  Q_mms = (W_b − k · Λ_fd) · (T_mms − T_a)
    /// ```
    ///
    /// After exactly one time step: T_new − T_mms = (dt/ρc) · 0 = 0 to machine
    /// precision, verifying the FD Laplacian, perfusion, and source terms jointly.
    ///
    /// ## Reference
    ///
    /// Oberkampf, W.L. & Roy, C.J. (2010). *Verification and Validation in
    /// Scientific Computing.* Cambridge University Press. §7.3.
    #[test]
    fn test_pennes_mms_steady_state() {
        use std::f64::consts::PI;

        let n: usize = 12;
        let dx = 1e-3_f64;
        let dt = 1e-4_f64; // well within stability limit (~1.27 s)
        let a_amp = 10.0_f64; // K — temperature amplitude of manufactured solution

        let config = BioheatConfig {
            tissue: TissueProperties {
                metabolic_rate: 0.0, // background Q_m = 0; only Q_mms drives steady state
                ..TissueProperties::default()
            },
            temperature_dependent: false,
            acoustic_coupling: false,
            compute_thermal_dose: false,
            dt,
            dx,
            ..BioheatConfig::default()
        };

        let k = config.tissue.conductivity;
        let wb_cb = config.blood.base_perfusion * config.blood.density * config.blood.specific_heat;
        let t_a = config.blood.arterial_temperature;

        // Discrete Laplacian eigenvalue: ∇²ₕ sin(πi/M) = Λ_fd · sin(πi/M)
        // where M = n-1 so the eigenfunction vanishes at both boundary indices i=0 and i=n-1.
        let m = (n - 1) as f64;
        let lambda_fd = 6.0 * ((PI / m).cos() - 1.0) / (dx * dx);

        let mut t_mms = Array3::<f64>::from_elem((n, n, n), t_a);
        let mut q_mms = Array3::<f64>::zeros((n, n, n));

        for i in 1..n - 1 {
            for j in 1..n - 1 {
                for kk in 1..n - 1 {
                    let val = a_amp
                        * (PI * i as f64 / m).sin()
                        * (PI * j as f64 / m).sin()
                        * (PI * kk as f64 / m).sin();
                    t_mms[[i, j, kk]] = t_a + val;
                    // Q_mms exactly cancels the Pennes residual at T_mms
                    q_mms[[i, j, kk]] = (wb_cb - k * lambda_fd) * val;
                }
            }
        }

        let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
        let mut solver = BioheatSolver::new(config, grid).unwrap();
        solver.set_initial_temperature(t_mms.clone());
        solver.set_heat_source(q_mms);
        solver.step();

        // After one step with exact manufactured source: T_new = T_mms to machine precision
        let t_new = solver.temperature();
        let max_err = t_mms
            .iter()
            .zip(t_new.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_err < 1e-10,
            "MMS residual |T_new − T_mms|_∞ = {:.3e} (expected < 1e-10 for exact manufactured source)",
            max_err
        );
    }

    #[test]
    fn test_dt_validation() {
        let config = BioheatConfig {
            dt: 1.0,
            ..BioheatConfig::default()
        };

        let grid = Grid::new(5, 5, 5, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let result = BioheatSolver::new(config, grid);

        assert!(result.is_err(), "Should reject unstable time step");
    }

    /// Test temperature-dependent absorption coefficient
    ///
    /// **Validation** (Bamber & Hill 1979):
    /// α(T) / α₀ = 1 + α_T · (T − T_ref) = 1 + 0.015 · (333 − 310.15) = 1.343
    ///
    /// Tolerance ±0.01 (0.7% relative).
    #[test]
    fn test_temperature_dependent_alpha() {
        let model = AbsorptionModel::tissue(1.0); // α₀ = 1 Np/m
        let t_test = 333.0_f64; // 60 °C in Kelvin

        let alpha = model.alpha_at(t_test);
        let expected = 1.0 + 0.015 * (t_test - 310.15);

        assert!(
            (alpha - expected).abs() < 0.01,
            "α(333K)/α₀ = {:.4} expected {:.4} (tolerance ±0.01)",
            alpha,
            expected
        );

        // At body temperature (T_ref), should equal α₀
        let alpha_body = model.alpha_at(310.15);
        assert!(
            (alpha_body - 1.0).abs() < 1e-12,
            "α(T_ref) must equal α₀, got {}",
            alpha_body
        );
    }

    /// Test acoustic heat source computation
    ///
    /// **Validation**:
    /// Q_abs = α · p_rms² / (ρ · c)
    ///       = 0.25 × (0.5e6)² / (1000 × 1500)
    ///       = 0.25 × 2.5e11 / 1.5e6
    ///       = 41.67 W/m³  ±0.1%
    ///
    /// Reference: Bamber, J.C. & Hill, C.R. (1979). Ultrasound Med Biol 5(2).
    #[test]
    fn test_acoustic_heat_source() {
        let config = BioheatConfig {
            dt: 1e-4,
            ..BioheatConfig::default()
        };
        let grid = Grid::new(3, 3, 3, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let solver = BioheatSolver::new(config, grid).unwrap();

        let p_rms = 0.5e6_f64; // 0.5 MPa
        let alpha_0 = 0.25_f64; // Np/m
        let rho = 1000.0_f64; // kg/m³
        let c = 1500.0_f64; // m/s

        // Uniform pressure and body temperature (no temperature correction)
        let pressure = Array3::from_elem((3, 3, 3), p_rms);
        let temperature = Array3::from_elem((3, 3, 3), 310.15_f64); // T_ref → α(T) = α₀

        let model = AbsorptionModel::tissue(alpha_0);
        let q = solver.compute_acoustic_heat_source(&pressure, &temperature, &model, rho, c);

        let expected = alpha_0 * p_rms * p_rms / (rho * c);
        let rel_error = (q[[1, 1, 1]] - expected).abs() / expected;

        assert!(
            rel_error < 0.001,
            "Q_abs = {:.4} W/m³, expected {:.4} W/m³, rel error {:.4}%",
            q[[1, 1, 1]],
            expected,
            rel_error * 100.0
        );
    }
}
