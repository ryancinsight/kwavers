//! The `Simulation.run` entry point: request assembly, the GIL-free solver call, and the Python result.

use super::super::*;
use super::prepare;

#[pymethods]
impl Simulation {
    /// Run the simulation.
    ///
    /// Parameters
    /// ----------
    /// time_steps : int
    ///     Number of time steps to simulate.
    /// dt : float, optional
    ///     Time step size `s`. When ``None``, auto-calculated from the CFL
    ///     condition: ``dt = 0.3 * min(dx,dy,dz) / (c_max * sqrt(3))``.
    /// record_start_index : int, default 1
    ///     Start index for recording (k-Wave convention).
    /// record_modes : list[str] or None
    ///     Recording modes: ``["p_max", "p_min", "p_rms", "p_final", "all",
    ///     "ux", "uy", "uz", "ux_non_staggered", ...]``.
    ///
    /// Returns
    /// -------
    /// SimulationResult
    ///
    /// The numerical runner executes without the Python GIL. Python object
    /// construction remains attached to the interpreter after the Rust result
    /// is complete.
    #[pyo3(signature = (time_steps, dt=None, record_start_index=1, record_modes=None))]
    fn run(
        &mut self,
        py: Python<'_>,
        time_steps: usize,
        dt: Option<f64>,
        record_start_index: usize,
        record_modes: Option<Vec<String>>,
    ) -> PyResult<crate::simulation_result_py::SimulationResult> {
        use crate::simulation_result_py::build_simulation_result;

        // ── Guard: time_steps must be at least 1 ──────────────────────────
        if time_steps == 0 {
            return Err(PyValueError::new_err("time_steps must be at least 1"));
        }

        // ── Fall back to sensor record modes when not explicitly passed ────
        let record_modes = record_modes
            .or_else(|| self.sensor.as_ref().map(|s| s.record_modes.clone()))
            .unwrap_or_default();

        // ── Compute dt from CFL condition when not provided ───────────────
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let dt = prepare::time_step(dt, dx_min, c_max);

        // ── Resolve solver_type to kwavers SolverType ───────────────────────
        let solver_type = prepare::solver_type(self.solver_type)?;

        // ── Process sources ─────────────────────────────────────────────────
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let mut grid_source = GridSource::new_empty();
        let mut dynamic_sources: Vec<Box<dyn kwavers_source::Source>> = Vec::new();
        let mut has_mask_source = false;
        let mut elastic_ivp_axis: Option<String> = None;
        let mut elastic_velocity_source = None;

        for src in &self.sources {
            crate::simulation_py::run::sources::process_source_for_run(
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

        // ── Sensor mask / transducer indices ────────────────────────────────
        let sensor_mask = Simulation::create_sensor_mask(
            &self.grid.inner,
            self.sensor.as_ref(),
            self.transducer_sensor.as_ref(),
        );

        let transducer_ordered_indices = self
            .transducer_sensor
            .as_ref()
            .map(|ts| Simulation::create_transducer_ordered_indices(&self.grid.inner, &ts.inner));

        // ── Config references (moved from config builders, no clone) ────────
        // ── Extract transducer refs for RS solver ───────────────────────────
        let kwavers_transducers: Vec<kwavers_transducer::array_2d::TransducerArray2D> =
            self.transducers.iter().map(|t| t.inner.clone()).collect();

        // ── Convert elastic velocity source ─────────────────────────────────
        let kwavers_elastic_vsrc = prepare::elastic_velocity_source(elastic_velocity_source);

        // ── Build request ───────────────────────────────────────────────────
        let req = SimulationRunRequest {
            grid: &self.grid.inner,
            medium: self.medium.inner.as_medium(),
            time_steps,
            dt,
            solver_type,
            fft_backend: prepare::fft_backend(self.fft_backend),
            pml: self.pml_config.as_ref(),
            helmholtz: self
                .helmholtz_config
                .as_ref()
                .filter(|cfg| cfg.frequency.is_some()),
            nonlinear: self
                .nonlinear_config
                .as_ref()
                .filter(|cfg| cfg.enabled || cfg.alpha_coeff > 0.0),
            thermal: self.thermal.as_ref(),
            poroelastic: self.poroelastic.as_ref(),
            compatibility_mode: self.compatibility_mode,
            kspace_correction: self.kspace_correction.clone(),
            axisymmetric: self.axisymmetric,
            grid_source,
            sensor_mask: Some(sensor_mask),
            transducer_ordered_indices,
            record_modes,
            record_start_index,
            transducers_for_rs: &kwavers_transducers,
            elastic_velocity_source: kwavers_elastic_vsrc,
            elastic_ivp_axis: prepare::ivp_axis(elastic_ivp_axis.as_deref()),
        };

        // ── Run ─────────────────────────────────────────────────────────────
        let result = py
            .detach(|| SimulationRunner::run(&req, dynamic_sources))
            .map_err(kwavers_error_to_py)?;

        // ── Build Python result ─────────────────────────────────────────────
        Python::attach(|py| build_simulation_result(py, &result, &self.grid.inner, time_steps, dt))
    }
}
