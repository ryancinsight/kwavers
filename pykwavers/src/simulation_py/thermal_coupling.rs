//! Acoustic→thermal coupling configuration and PSTD-coupled time loop.
//!
//! `ThermalCouplingConfig` stores all parameters needed to run a coupled
//! PSTD acoustic + Pennes bioheat simulation via
//! `PSTDSolver::run_orchestrated_with_thermal`.
//!
//! ## Coupling model
//!
//! The acoustic volumetric heat source Q = 2α·c·e [W/m³] (Nyborg 1981) is
//! computed from PSTD pressure and velocity fields after every
//! `n_acoustic_per_thermal` acoustic steps and passed to
//! `ThermalDiffusionSolver::update` as Q/(ρ·cp) [K/s].

use kwavers::core::constants::fundamental::{SOUND_SPEED_TISSUE};
use kwavers::core::constants::medical::THERMAL_DOSE_REFERENCE_TEMP_C;
use kwavers::core::constants::thermodynamic::{BODY_TEMPERATURE_C, KELVIN_OFFSET_C};
use kwavers::core::error::{KwaversError, KwaversResult};
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::source::{GridSource, Source as KwaversSource};
use kwavers::physics::thermal::diffusion::ThermalDiffusionConfig;
use kwavers::solver::forward::pstd::config::CompatibilityMode;
use kwavers::solver::forward::thermal_diffusion::ThermalDiffusionSolver;
use ndarray::Array3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::medium_py::MediumInner;
use crate::sensor_py::Sensor;
use crate::simulation_result_py::{extract_full_grid_stats, SimulationRunResult};
use crate::transducer_array_py::TransducerArray2D;

use super::Simulation;

// ── Default thermal properties (soft tissue, ICRU Report 44) ──────────────────
const DEFAULT_K: f64 = 0.5;
const DEFAULT_RHO: f64 = 1000.0;
const DEFAULT_CP: f64 = 3600.0;
const DEFAULT_WB: f64 = 5e-3;
const DEFAULT_RHO_B: f64 = 1050.0; // ICRU-44 bioheat blood density (distinct from DENSITY_BLOOD=1060)
const DEFAULT_CPB: f64 = 3840.0;
const DEFAULT_TA_C: f64 = BODY_TEMPERATURE_C;

/// Configuration for acoustic→thermal coupling attached to a `Simulation`.
#[derive(Clone, Debug)]
pub(crate) struct ThermalCouplingConfig {
    pub thermal_conductivity: f64,
    pub density: f64,
    pub specific_heat: f64,
    pub enable_bioheat: bool,
    pub perfusion_rate: f64,
    pub blood_density: f64,
    pub blood_specific_heat: f64,
    pub arterial_temperature_c: f64,
    pub metabolic_heat: f64,
    pub initial_temperature_c: f64,
    pub track_thermal_dose: bool,
    /// Center frequency [Hz] for α(ω_c) computation.
    pub center_frequency_hz: f64,
    /// Acoustic steps per thermal update (≥ 1).
    pub n_acoustic_per_thermal: usize,
    /// Thermal time step [s]. `None` → `n_acoustic_per_thermal * dt_acoustic`.
    pub dt_thermal: Option<f64>,
}

impl ThermalCouplingConfig {
    /// Build a `ThermalDiffusionSolver` + matching `HomogeneousMedium` from this config.
    pub(crate) fn build_solver(
        &self,
        grid: &KwaversGrid,
    ) -> KwaversResult<(ThermalDiffusionSolver, HomogeneousMedium)> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut medium = HomogeneousMedium::new(self.density, SOUND_SPEED_TISSUE, 0.0, 0.0, grid);
        medium
            .set_thermal_properties(self.thermal_conductivity, self.specific_heat)
            .map_err(|e| KwaversError::InternalError(e.to_string()))?;

        let arterial_temp_k = self.arterial_temperature_c + KELVIN_OFFSET_C;
        let initial_temp_k = self.initial_temperature_c + KELVIN_OFFSET_C;

        let config = ThermalDiffusionConfig {
            enable_bioheat: self.enable_bioheat,
            perfusion_rate: self.perfusion_rate,
            blood_density: self.blood_density,
            blood_specific_heat: self.blood_specific_heat,
            arterial_temperature: arterial_temp_k,
            enable_hyperbolic: false,
            relaxation_time: 20.0,
            track_thermal_dose: self.track_thermal_dose,
            dose_reference_temperature: THERMAL_DOSE_REFERENCE_TEMP_C,
            spatial_order: 2,
        };

        let mut solver = ThermalDiffusionSolver::new(config, grid);
        solver.set_temperature(Array3::from_elem((nx, ny, nz), initial_temp_k));
        Ok((solver, medium))
    }
}

// ── Python API ────────────────────────────────────────────────────────────────

#[pymethods]
impl Simulation {
    /// Attach acoustic→thermal coupling to this simulation.
    ///
    /// When set, `Simulation.run()` with a PSTD solver drives the coupled time
    /// loop: acoustic heat deposition Q = 2α·c·e [W/m³] feeds the Pennes
    /// bioheat / thermal diffusion solver every `n_acoustic_per_thermal` steps.
    ///
    /// The result's `thermal_temperature` (°C) and `thermal_dose` (CEM43 min)
    /// fields are populated after the run.
    ///
    /// Parameters
    /// ----------
    /// center_frequency : float
    ///     Center frequency [Hz] for α(ω_c) evaluation (required).
    /// n_acoustic_per_thermal : int, default 1
    ///     Acoustic steps per thermal update.
    /// thermal_conductivity : float, default 0.5
    ///     k [W/(m·K)].
    /// density : float, default 1000.0
    ///     ρ [kg/m³].
    /// specific_heat : float, default 3600.0
    ///     cp [J/(kg·K)].
    /// enable_bioheat : bool, default False
    ///     Enable Pennes perfusion + metabolic terms.
    /// perfusion_rate : float, default 5e-3
    ///     w_b [1/s].
    /// blood_density : float, default 1050.0
    ///     ρ_b [kg/m³].
    /// blood_specific_heat : float, default 3840.0
    ///     c_b [J/(kg·K)].
    /// arterial_temperature : float, default 37.0
    ///     T_a [°C].
    /// metabolic_heat : float, default 0.0
    ///     Q_m [W/m³].
    /// initial_temperature : float, default 37.0
    ///     T_0 [°C].
    /// track_thermal_dose : bool, default True
    ///     Compute CEM43 dose field.
    /// dt_thermal : float or None
    ///     Override thermal time step [s]. Default: `n_acoustic_per_thermal * dt_acoustic`.
    #[pyo3(signature = (
        center_frequency,
        n_acoustic_per_thermal = 1,
        thermal_conductivity = DEFAULT_K,
        density = DEFAULT_RHO,
        specific_heat = DEFAULT_CP,
        enable_bioheat = false,
        perfusion_rate = DEFAULT_WB,
        blood_density = DEFAULT_RHO_B,
        blood_specific_heat = DEFAULT_CPB,
        arterial_temperature = DEFAULT_TA_C,
        metabolic_heat = 0.0,
        initial_temperature = DEFAULT_TA_C,
        track_thermal_dose = true,
        dt_thermal = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn set_thermal(
        &mut self,
        center_frequency: f64,
        n_acoustic_per_thermal: usize,
        thermal_conductivity: f64,
        density: f64,
        specific_heat: f64,
        enable_bioheat: bool,
        perfusion_rate: f64,
        blood_density: f64,
        blood_specific_heat: f64,
        arterial_temperature: f64,
        metabolic_heat: f64,
        initial_temperature: f64,
        track_thermal_dose: bool,
        dt_thermal: Option<f64>,
    ) -> PyResult<()> {
        if center_frequency <= 0.0 {
            return Err(PyValueError::new_err("center_frequency must be > 0"));
        }
        if n_acoustic_per_thermal == 0 {
            return Err(PyValueError::new_err("n_acoustic_per_thermal must be >= 1"));
        }
        if thermal_conductivity <= 0.0 || density <= 0.0 || specific_heat <= 0.0 {
            return Err(PyValueError::new_err(
                "thermal_conductivity, density, specific_heat must be > 0",
            ));
        }
        self.thermal = Some(ThermalCouplingConfig {
            thermal_conductivity,
            density,
            specific_heat,
            enable_bioheat,
            perfusion_rate,
            blood_density,
            blood_specific_heat,
            arterial_temperature_c: arterial_temperature,
            metabolic_heat,
            initial_temperature_c: initial_temperature,
            track_thermal_dose,
            center_frequency_hz: center_frequency,
            n_acoustic_per_thermal,
            dt_thermal,
        });
        Ok(())
    }

    /// Remove thermal coupling, reverting to acoustic-only simulation.
    pub fn clear_thermal(&mut self) {
        self.thermal = None;
    }

    /// True if thermal coupling is configured.
    #[getter]
    pub fn has_thermal(&self) -> bool {
        self.thermal.is_some()
    }
}

// ── Coupled PSTD + thermal impl ───────────────────────────────────────────────

impl Simulation {
    /// Run PSTD + thermal coupled simulation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_pstd_with_thermal_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt_acoustic: f64,
        compatibility_mode: CompatibilityMode,
        enable_nonlinear: bool,
        alpha_coeff_db: f64,
        alpha_power: f64,
        grid_source: GridSource,
        sources: Vec<Box<dyn KwaversSource>>,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        pml_size: Option<usize>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        pml_inside: bool,
        pml_alpha_xyz: Option<(f64, f64, f64)>,
        axisymmetric: bool,
        record_modes: &[String],
        record_start_index: usize,
        thermal_cfg: &ThermalCouplingConfig,
    ) -> KwaversResult<SimulationRunResult> {
        let (mut solver, _sim_grid, _sensor_mask) = Self::prepare_pstd_solver(
            grid,
            medium,
            time_steps,
            dt_acoustic,
            compatibility_mode,
            enable_nonlinear,
            alpha_coeff_db,
            alpha_power,
            grid_source,
            sources,
            sensor,
            transducer_sensor,
            pml_size,
            pml_size_xyz,
            pml_inside,
            pml_alpha_xyz,
            axisymmetric,
            record_modes,
        )?;

        let (mut thermal_solver, thermal_medium) = thermal_cfg.build_solver(grid)?;

        let omega_c = std::f64::consts::TAU * thermal_cfg.center_frequency_hz;
        let dt_thermal = thermal_cfg
            .dt_thermal
            .unwrap_or(dt_acoustic * thermal_cfg.n_acoustic_per_thermal as f64);
        let rho_cp = thermal_cfg.density * thermal_cfg.specific_heat;
        let background_heat_ks = thermal_cfg.metabolic_heat / rho_cp;

        solver.run_orchestrated_with_thermal(
            kwavers::solver::forward::pstd::implementation::core::orchestrator::thermal::ThermalOrchestrationInput {
                acoustic_steps: time_steps,
                thermal: &mut thermal_solver,
                thermal_medium: &thermal_medium,
                omega_c,
                dt_thermal,
                n_acoustic_per_thermal: thermal_cfg.n_acoustic_per_thermal,
                rho_cp,
                background_heat_ks,
            },
        )?;

        let stats = solver.sensor_recorder.extract_all_stats();
        let full_data = solver
            .sensor_recorder
            .recorded_pressure_view()
            .ok_or_else(|| KwaversError::Io(std::io::Error::other("No sensor data recorded")))?;
        let sensor_data =
            Self::trim_initial_recorder_view(full_data, time_steps, record_start_index);

        let ux_data = solver
            .sensor_recorder
            .recorded_ux_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let uy_data = solver
            .sensor_recorder
            .recorded_uy_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let uz_data = solver
            .sensor_recorder
            .recorded_uz_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let ix_data = solver
            .sensor_recorder
            .recorded_ix_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let iy_data = solver
            .sensor_recorder
            .recorded_iy_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let iz_data = solver
            .sensor_recorder
            .recorded_iz_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let i_avg_x = solver.sensor_recorder.extract_i_avg_x();
        let i_avg_y = solver.sensor_recorder.extract_i_avg_y();
        let i_avg_z = solver.sensor_recorder.extract_i_avg_z();
        let velocity_stats = solver.sensor_recorder.extract_sampled_velocity_stats();
        let full_grid_stats = extract_full_grid_stats(&solver.sensor_recorder);

        // Temperature stays in Kelvin here; °C conversion happens in result.rs.
        let thermal_temperature = Some(thermal_solver.temperature().to_owned());
        let thermal_dose = thermal_solver.thermal_dose().map(|d| d.to_owned());

        Ok(SimulationRunResult {
            sensor_data,
            stats,
            ux_data,
            uy_data,
            uz_data,
            ix_data,
            iy_data,
            iz_data,
            i_avg_x,
            i_avg_y,
            i_avg_z,
            velocity_stats,
            full_grid_stats,
            thermal_temperature,
            thermal_dose,
        })
    }
}
