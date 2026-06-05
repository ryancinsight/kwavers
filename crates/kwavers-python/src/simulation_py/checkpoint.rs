use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kwavers_core::error::KwaversError;

use crate::simulation_result_py::SimulationResult;
use crate::solver_type_bindings::SolverType;

use super::{run::kwavers_error_to_py_local, Simulation};

#[pymethods]
impl Simulation {
    /// Run PSTD for `checkpoint_steps` steps and save state to `checkpoint_path`.
    ///
    /// Only supports `SolverType.PSTD`.
    #[pyo3(signature = (time_steps, checkpoint_steps, checkpoint_path, dt=None))]
    pub(super) fn run_to_checkpoint(
        &self,
        py: Python<'_>,
        time_steps: usize,
        checkpoint_steps: usize,
        checkpoint_path: String,
        dt: Option<f64>,
    ) -> PyResult<()> {
        if !matches!(self.solver_type, SolverType::PSTD) {
            return Err(PyValueError::new_err(
                "run_to_checkpoint only supports SolverType.PSTD",
            ));
        }
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let dt_actual = dt.unwrap_or_else(|| 0.3 * dx_min / (c_max * 3.0_f64.sqrt()));

        let (grid_source, dynamic_sources) = self.build_sources(time_steps, dt_actual, c_max)?;

        let grid_clone = self.grid.inner.clone();
        let medium_clone = self.medium.inner.clone();
        let sensor_opt = self.sensor.clone();
        let transducer_sensor_opt = self.transducer_sensor.clone();
        let pml_size = self.pml_size;
        let pml_size_xyz = self.pml_size_xyz;
        let pml_inside = self.pml_inside;
        let pml_alpha_xyz = self.pml_alpha_xyz;
        let compatibility_mode = self.compatibility_mode;
        let enable_nonlinear = self.enable_nonlinear;
        let alpha_coeff = self.alpha_coeff;
        let alpha_power = self.alpha_power;
        let path = std::path::PathBuf::from(checkpoint_path);

        py.detach(move || {
            Simulation::run_pstd_to_checkpoint(
                &grid_clone,
                &medium_clone,
                time_steps,
                checkpoint_steps,
                dt_actual,
                compatibility_mode,
                enable_nonlinear,
                alpha_coeff,
                alpha_power,
                grid_source,
                dynamic_sources,
                sensor_opt.as_ref(),
                transducer_sensor_opt.as_ref(),
                pml_size,
                pml_size_xyz,
                pml_inside,
                pml_alpha_xyz,
                &path,
            )
        })
        .map_err(kwavers_error_to_py_local)
    }

    /// Resume a PSTD simulation from a checkpoint and return sensor data.
    ///
    /// Only supports `SolverType.PSTD`.
    #[pyo3(signature = (time_steps, checkpoint_path, dt=None))]
    pub(super) fn run_from_checkpoint(
        &self,
        py: Python<'_>,
        time_steps: usize,
        checkpoint_path: String,
        dt: Option<f64>,
    ) -> PyResult<SimulationResult> {
        if !matches!(self.solver_type, SolverType::PSTD) {
            return Err(PyValueError::new_err(
                "run_from_checkpoint only supports SolverType.PSTD",
            ));
        }
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let dt_actual = dt.unwrap_or_else(|| 0.3 * dx_min / (c_max * 3.0_f64.sqrt()));

        let (grid_source, dynamic_sources) = self.build_sources(time_steps, dt_actual, c_max)?;

        let grid_clone = self.grid.inner.clone();
        let medium_clone = self.medium.inner.clone();
        let sensor_opt = self.sensor.clone();
        let transducer_sensor_opt = self.transducer_sensor.clone();
        let pml_size = self.pml_size;
        let pml_size_xyz = self.pml_size_xyz;
        let pml_inside = self.pml_inside;
        let pml_alpha_xyz = self.pml_alpha_xyz;
        let compatibility_mode = self.compatibility_mode;
        let enable_nonlinear = self.enable_nonlinear;
        let alpha_coeff = self.alpha_coeff;
        let alpha_power = self.alpha_power;
        let path = std::path::PathBuf::from(checkpoint_path);
        let sensor_record_modes: Vec<String> = sensor_opt
            .as_ref()
            .map(|s| s.record_modes.clone())
            .unwrap_or_default();
        let checkpoint = kwavers_solver::forward::pstd::checkpoint::PSTDCheckpoint::load(&path)
            .map_err(kwavers_error_to_py_local)?;
        checkpoint
            .validate_restore_contract(
                self.grid.inner.nx,
                self.grid.inner.ny,
                self.grid.inner.nz,
                time_steps,
                dt_actual,
            )
            .map_err(kwavers_error_to_py_local)?;
        let remaining_steps = time_steps
            .checked_sub(checkpoint.time_step_index)
            .ok_or_else(|| {
                kwavers_error_to_py_local(KwaversError::InvalidInput(format!(
                    "checkpoint time_step_index {} exceeds solver total_steps {}",
                    checkpoint.time_step_index, time_steps
                )))
            })?;
        let shape = (self.grid.inner.nx, self.grid.inner.ny, self.grid.inner.nz);

        let run_result = py
            .detach(move || {
                Simulation::run_pstd_from_checkpoint_loaded(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    compatibility_mode,
                    enable_nonlinear,
                    alpha_coeff,
                    alpha_power,
                    grid_source,
                    dynamic_sources,
                    sensor_opt.as_ref(),
                    transducer_sensor_opt.as_ref(),
                    pml_size,
                    pml_size_xyz,
                    pml_inside,
                    pml_alpha_xyz,
                    checkpoint,
                    remaining_steps,
                    &path,
                    &sensor_record_modes,
                )
            })
            .map_err(kwavers_error_to_py_local)?;

        Simulation::simulation_run_result_to_py(py, run_result, shape, time_steps, dt_actual)
    }
}
