//! Nonlinear acoustic solver dispatch.
//!
//! Routes `SolverType::Nonlinear` to the Westervelt FDTD solver.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_source::Source;
use crate::dispatch::shared::trim_initial_recorder_sample;
use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_solver::forward::nonlinear::westervelt::{WesterveltFdtd, WesterveltFdtdConfig};

/// Run a Westervelt nonlinear FDTD simulation.
///
/// Uses the Westervelt FDTD solver for nonlinear acoustic propagation.
/// Sources are built from the grid source's p0 field as simple point sources.
pub fn run(req: &SimulationRunRequest<'_>) -> KwaversResult<SimulationRunResult> {
    let nl = req.nonlinear.cloned().unwrap_or_default();
    if !nl.enabled {
        return Err(KwaversError::InvalidInput(
            "Nonlinear solver requested but nonlinearity is not enabled in NonlinearConfig".into(),
        ));
    }

    let config = WesterveltFdtdConfig {
        spatial_order: 4,
        enable_absorption: nl.alpha_coeff > 0.0,
        cfl_safety: 0.3,
        artificial_viscosity: 0.0,
    };

    let mut solver = WesterveltFdtd::new(config, req.grid, req.medium);

    let sources: Vec<Box<dyn Source>> = build_sources(&req.grid_source, req.grid);

    let n_sensors = 1usize.max(
        req.sensor_mask
            .as_ref()
            .map(|m| m.iter().filter(|&&a| a).count())
            .unwrap_or(0),
    );
    let mut sensor_data = ndarray::Array2::<f64>::zeros((n_sensors, req.time_steps + 1));

    // Record initial state
    if let Some(ref mask) = req.sensor_mask {
        let p0 = solver.pressure();
        let mut sensor_idx = 0;
        for ((i, j, k), &active) in mask.indexed_iter() {
            if active {
                sensor_data[[sensor_idx, 0]] = p0[[i, j, k]];
                sensor_idx += 1;
            }
        }
    }

    let grid = req.grid;
    for step in 0..req.time_steps {
        let t = (step as f64) * req.dt;
        solver.update(req.medium, grid, &sources, t, req.dt)?;

        if let Some(ref mask) = req.sensor_mask {
            let pressure = solver.pressure();
            let mut sensor_idx = 0;
            for ((i, j, k), &active) in mask.indexed_iter() {
                if active {
                    sensor_data[[sensor_idx, step + 1]] = pressure[[i, j, k]];
                    sensor_idx += 1;
                }
            }
        }
    }

    let sensor_data =
        trim_initial_recorder_sample(sensor_data, req.time_steps, req.record_start_index);

    Ok(SimulationRunResult {
        sensor_data,
        stats: None,
        ux_data: None, uy_data: None, uz_data: None,
        ix_data: None, iy_data: None, iz_data: None,
        i_avg_x: None, i_avg_y: None, i_avg_z: None,
        velocity_stats: None,
        full_grid_stats: None,
        thermal_temperature: None,
        thermal_dose: None,
    })
}

/// Build a vector of source trait objects from a grid source.
///
/// Uses the existing `PointSource` and `NullSignal` types from the domain
/// layer.  Each non-zero node in `p0` becomes a constant-amplitude point
/// source at that cell's physical coordinates.
fn build_sources(gs: &kwavers_source::GridSource, grid: &Grid) -> Vec<Box<dyn Source>> {
    use kwavers_signal::{NullSignal, Signal};
    use kwavers_source::PointSource;
    use std::sync::Arc;

    let p0 = match gs.p0.as_ref() {
        Some(p) => p,
        None => return Vec::new(),
    };

    let signal: Arc<dyn Signal> = Arc::new(NullSignal::new());
    let mut sources: Vec<Box<dyn Source>> = Vec::new();
    for ((i, j, k), &val) in p0.indexed_iter() {
        if val.abs() > 1e-15 {
            let (x, y, z) = grid.indices_to_coordinates(i, j, k);
            sources.push(Box::new(PointSource::new(
                (x, y, z),
                Arc::clone(&signal),
            )));
        }
    }
    sources
}
