use pyo3::prelude::*;

use kwavers::domain::source::{GridSource, Source as KwaversSource};

use crate::simulation_result_py::SimulationResult;
use crate::solver_type_bindings::SolverType;

use super::helpers::SampledSignal;
use super::{run::kwavers_error_to_py_local, Simulation};

use std::sync::Arc;

#[pymethods]
impl Simulation {
    /// Run the simulation.
    ///
    /// Parameters
    /// ----------
    /// time_steps : int
    ///     Number of time steps
    /// dt : float, optional
    ///     Time step [s] (auto-calculated from CFL if None)
    ///
    /// Returns
    /// -------
    /// SimulationResult
    ///     Simulation results with sensor data
    ///
    /// Examples
    /// --------
    /// >>> result = sim.run(time_steps=1000, dt=1e-8)
    /// >>> print(result.sensor_data.shape)
    #[pyo3(signature = (time_steps, dt=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        time_steps: usize,
        dt: Option<f64>,
    ) -> PyResult<SimulationResult> {
        // Calculate time step from CFL condition if not provided
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let cfl = 0.3; // Conservative CFL number for 3D
        let dt_actual = dt.unwrap_or_else(|| cfl * dx_min / (c_max * 3.0_f64.sqrt()));

        let mut grid_source = GridSource::new_empty();
        let mut dynamic_sources: Vec<Box<dyn KwaversSource>> = Vec::new();
        let mut has_mask_source = false;
        let mut elastic_ivp_axis: Option<String> = None;
        let mut elastic_velocity_source: super::ElasticVelocitySource = None;

        for src in &self.sources {
            super::run::process_source_for_run(
                src,
                &self.grid,
                time_steps,
                c_max,
                &mut grid_source,
                &mut dynamic_sources,
                &mut has_mask_source,
                &mut elastic_ivp_axis,
                &mut elastic_velocity_source,
            )?;
        }

        for trans in &self.transducers {
            let mut inner_trans = trans.inner.clone();
            if let Some(ref sig_arr) = trans.input_signal {
                let sampled_sig = SampledSignal::new(sig_arr.clone(), dt_actual);
                inner_trans.set_signal(Arc::new(sampled_sig));
            }
            dynamic_sources.push(Box::new(inner_trans));
        }

        let shape = (self.grid.inner.nx, self.grid.inner.ny, self.grid.inner.nz);
        let grid_clone = self.grid.inner.clone();
        let medium_clone = self.medium.inner.clone();
        let sensor_opt = self.sensor.clone();
        let transducer_sensor_opt = self.transducer_sensor.clone();
        let pml_size = self.pml_size;
        let pml_size_xyz = self.pml_size_xyz;
        let pml_inside = self.pml_inside;
        let pml_alpha_xyz = self.pml_alpha_xyz;
        let solver_type = self.solver_type;
        let kspace_correction = self.kspace_correction.clone();
        let compatibility_mode = self.compatibility_mode;
        let enable_nonlinear = self.enable_nonlinear;
        let alpha_coeff = self.alpha_coeff;
        let alpha_power = self.alpha_power;
        let axisymmetric = self.axisymmetric;
        let thermal_cfg = self.thermal.clone();

        let sensor_record_modes: Vec<String> = sensor_opt
            .as_ref()
            .map(|s| s.record_modes.clone())
            .unwrap_or_default();
        let sensor_record_start_index: usize = sensor_opt
            .as_ref()
            .map(|s| s.record_start_index)
            .unwrap_or(1);

        let run_result = py
            .detach(move || match solver_type {
                SolverType::FDTD => Simulation::run_fdtd_impl(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    grid_source,
                    dynamic_sources,
                    sensor_opt.as_ref(),
                    transducer_sensor_opt.as_ref(),
                    pml_size,
                    pml_size_xyz,
                    pml_inside,
                    pml_alpha_xyz,
                    kspace_correction,
                    enable_nonlinear,
                    axisymmetric,
                    &sensor_record_modes,
                    sensor_record_start_index,
                ),
                SolverType::PSTD | SolverType::Hybrid => {
                    if let Some(ref cfg) = thermal_cfg {
                        Simulation::run_pstd_with_thermal_impl(
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
                            axisymmetric,
                            &sensor_record_modes,
                            sensor_record_start_index,
                            cfg,
                        )
                    } else {
                        Simulation::run_pstd_impl(
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
                            axisymmetric,
                            &sensor_record_modes,
                            sensor_record_start_index,
                        )
                    }
                }
                SolverType::PstdGpu => Simulation::run_gpu_pstd_or_cpu_fallback(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    compatibility_mode,
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
                    enable_nonlinear,
                    axisymmetric,
                    &sensor_record_modes,
                    sensor_record_start_index,
                ),
                SolverType::Elastic => Simulation::run_elastic_impl(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    grid_source,
                    sensor_opt.as_ref(),
                    pml_size,
                    pml_inside,
                    elastic_ivp_axis.as_deref(),
                    elastic_velocity_source,
                ),
                SolverType::ElasticPSTD => Simulation::run_elastic_pstd_impl(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    sensor_opt.as_ref(),
                    elastic_velocity_source,
                ),
            })
            .map_err(kwavers_error_to_py_local)?;

        Simulation::simulation_run_result_to_py(py, run_result, shape, time_steps, dt_actual)
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "Simulation(grid=Grid({nx}x{ny}x{nz}), sources={sources}, transducers={transducers}, solver={solver:?}, sensor={sensor})",
            nx = self.grid.inner.nx,
            ny = self.grid.inner.ny,
            nz = self.grid.inner.nz,
            sources = self.sources.len(),
            transducers = self.transducers.len(),
            solver = self.solver_type,
            sensor = if self.sensor.is_some() { "Sensor" } else { "Transducer" },
        )
    }
}
