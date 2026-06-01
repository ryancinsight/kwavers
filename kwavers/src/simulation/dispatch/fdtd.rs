//! FDTD solver dispatch.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::cpml::CPMLConfig;
use crate::domain::sensor::recorder::simple::SensorRecorder;
use crate::domain::source::Source as KwaversSource;
use crate::simulation::dispatch::shared::{recording_modes_from_strings, trim_initial_recorder_sample};
use crate::simulation::types::extract_full_grid_stats;
use crate::simulation::types::{SimulationRunRequest, SimulationRunResult};
use crate::solver::forward::fdtd::config::FdtdConfig;
use crate::solver::forward::fdtd::solver::FdtdSolver;
use crate::solver::geometry::SolverGeometry;
use crate::solver::interface::solver::Solver as SolverTrait;

/// Run an FDTD simulation for the given request.
pub fn run(
    req: &SimulationRunRequest<'_>,
    sources: Vec<Box<dyn KwaversSource>>,
) -> KwaversResult<SimulationRunResult> {
    let geometry = if req.axisymmetric {
        SolverGeometry::CylindricalAS
    } else {
        SolverGeometry::Cartesian3D
    };

    let config = FdtdConfig {
        dt: req.dt,
        nt: req.time_steps,
        spatial_order: 4,
        staggered_grid: true,
        cfl_factor: 0.3,
        subgridding: false,
        subgrid_factor: 2,
        enable_gpu_acceleration: false,
        enable_nonlinear: req.nonlinear.map_or(false, |n| n.enabled),
        kspace_correction: req.kspace_correction.clone(),
        sensor_mask: req.sensor_mask.clone(),
        geometry,
    };

    let mut solver = FdtdSolver::new(config, req.grid, req.medium, req.grid_source.clone())?;

    let modes = recording_modes_from_strings(&req.record_modes);
    if !modes.is_empty() {
        let shape = (req.grid.nx, req.grid.ny, req.grid.nz);
        solver.sensor_recorder =
            SensorRecorder::with_modes(req.sensor_mask.as_ref(), shape, req.time_steps + 1, &modes)?;
    } else if let Some(ref ordered) = req.transducer_ordered_indices {
        solver.sensor_recorder =
            SensorRecorder::from_ordered_indices(ordered.clone(), req.time_steps + 1)?;
    }

    let pml = req.pml.cloned().unwrap_or_default();
    let (default_thickness, max_allowed) = pml.effective_thickness(req.grid.nx, req.grid.ny, req.grid.nz);
    let thickness = pml.size.unwrap_or(default_thickness).min(max_allowed);

    if thickness > 0 && max_allowed > 0 {
        let mut cpml_config = if let Some((px, py, pz)) = pml.size_xyz {
            CPMLConfig::with_per_dimension_thickness(px, py, pz)
        } else {
            CPMLConfig::with_thickness(thickness)
        };
        if let Some((ax, ay, az)) = pml.alpha_xyz {
            cpml_config = cpml_config.with_alpha_xyz(ax, ay, az);
        }
        let max_c = req.medium.max_sound_speed();
        solver.enable_cpml(cpml_config, req.dt, max_c)?;
    }

    for source in sources {
        SolverTrait::add_source(&mut solver, source)?;
    }

    solver.run_orchestrated(req.time_steps)?;

    let stats = solver.sensor_recorder.extract_all_stats();
    let full_data = solver.extract_recorded_sensor_data().ok_or_else(|| {
        KwaversError::Io(std::io::Error::other("No sensor data recorded"))
    })?;
    let sensor_data = trim_initial_recorder_sample(full_data, req.time_steps, req.record_start_index);
    let full_grid_stats = extract_full_grid_stats(&solver.sensor_recorder);

    Ok(SimulationRunResult {
        sensor_data,
        stats,
        ux_data: None,
        uy_data: None,
        uz_data: None,
        ix_data: None,
        iy_data: None,
        iz_data: None,
        i_avg_x: None,
        i_avg_y: None,
        i_avg_z: None,
        velocity_stats: None,
        full_grid_stats,
        thermal_temperature: None,
        thermal_dose: None,
    })
}
