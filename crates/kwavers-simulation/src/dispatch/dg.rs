//! DG solver dispatch.

use std::sync::Arc;

use leto::Array3;

use crate::dispatch::shared::trim_initial_recorder_sample;
use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_core::error::KwaversResult;
use kwavers_receiver::recorder::pressure_statistics::SampledStatistics;
use kwavers_solver::forward::pstd::dg::{HybridSpectralDGConfig, HybridSpectralDGSolver};

/// Run a discontinuous Galerkin (hybrid spectral) simulation.
pub fn run(req: &SimulationRunRequest<'_>) -> KwaversResult<SimulationRunResult> {
    let sensor_mask = req
        .sensor_mask
        .clone()
        .unwrap_or_else(|| Array3::from_elem((req.grid.nx, req.grid.ny, req.grid.nz), false));
    let c = req.medium.max_sound_speed();
    let config = HybridSpectralDGConfig::default();
    let grid_arc = Arc::new(req.grid.clone());
    let mut solver = HybridSpectralDGSolver::new(config, grid_arc);

    let mut field = req
        .grid_source
        .p0
        .as_ref()
        .map(|a| a.to_contiguous())
        .unwrap_or_else(|| Array3::zeros((req.grid.nx, req.grid.ny, req.grid.nz)));
    let mut output = Array3::<f64>::zeros((req.grid.nx, req.grid.ny, req.grid.nz));

    let sensor_indices: Vec<(usize, usize, usize)> = sensor_mask
        .indexed_iter()
        .filter(|(_, &active)| active)
        .map(|([i, j, k], _)| (i, j, k))
        .collect();
    let n_sensors = sensor_indices.len().max(1);
    let mut sensor_data = leto::Array2::<f64>::zeros((n_sensors, req.time_steps + 1));

    for (idx, &(i, j, k)) in sensor_indices.iter().enumerate() {
        sensor_data[[idx, 0]] = field[[i, j, k]];
    }

    for step in 0..req.time_steps {
        solver.solve_step_into(&field, req.dt, c, &mut output)?;
        for (idx, &(i, j, k)) in sensor_indices.iter().enumerate() {
            sensor_data[[idx, step + 1]] = output[[i, j, k]];
        }
        std::mem::swap(&mut field, &mut output);
    }

    let sensor_data =
        trim_initial_recorder_sample(sensor_data, req.time_steps, req.record_start_index);
    let stats = if !sensor_indices.is_empty() {
        let n_cols = sensor_data.shape()[1];
        Some(SampledStatistics {
            p_max: leto::Array1::from_iter(
                sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
            ),
            p_min: leto::Array1::from_iter(
                sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| row.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
            ),
            p_rms: leto::Array1::from_iter(
                sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| {
                        let sq: f64 = row.iter().map(|v| v * v).sum();
                        (sq / n_cols as f64).sqrt()
                    }),
            ),
            p_final: leto::Array1::from_iter(
                sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| row[n_cols.saturating_sub(1)]),
            ),
        })
    } else {
        None
    };

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
        full_grid_stats: None,
        thermal_temperature: None,
        thermal_dose: None,
    })
}
