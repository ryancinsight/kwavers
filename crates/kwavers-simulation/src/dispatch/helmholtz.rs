//! Helmholtz FEM solver dispatch.

use kwavers_math::fft::Complex64;
use leto::Array3;
use std::f64::consts::TAU;

use crate::dispatch::shared::trim_initial_recorder_sample;
use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_core::error::KwaversResult;
use kwavers_receiver::recorder::pressure_statistics::SampledStatistics;
use kwavers_solver::forward::helmholtz::fem::{
    FemHelmholtzConfig, FemHelmholtzSolver, FemPreconditionerType,
};

/// Run a frequency-domain Helmholtz FEM simulation for the given request.
pub fn run(req: &SimulationRunRequest<'_>) -> KwaversResult<SimulationRunResult> {
    let c_max = req.medium.max_sound_speed();
    let wavenumber = if let Some(freq) = req.helmholtz.and_then(|h| h.frequency) {
        TAU * freq / c_max
    } else {
        TAU / (c_max * req.dt)
    };

    let config = FemHelmholtzConfig {
        wavenumber,
        tolerance: 5e-4,
        max_iterations: 5000,
        preconditioner: FemPreconditionerType::Diagonal,
        ..FemHelmholtzConfig::default()
    };

    let mut solver = FemHelmholtzSolver::from_grid(config, req.grid)?;
    solver.assemble_system(req.medium)?;

    // Source injection from p0
    if let Some(ref p0) = req.grid_source.p0 {
        let nx = req.grid.nx;
        let ny = req.grid.ny;
        for ([i, j, k], &val) in p0.indexed_iter() {
            if val.abs() > 0.0 {
                let node_idx = i + nx * (j + ny * k);
                solver.add_nodal_load(node_idx, Complex64::new(val, 0.0))?;
            }
        }
    }

    solver.solve_system()?;

    let sensor_mask = req
        .sensor_mask
        .clone()
        .unwrap_or_else(|| Array3::from_elem((req.grid.nx, req.grid.ny, req.grid.nz), false));
    let sensor_indices: Vec<(usize, usize, usize)> = sensor_mask
        .indexed_iter()
        .filter(|(_, &active)| active)
        .map(|([i, j, k], _)| (i, j, k))
        .collect();

    let n_sensors = sensor_indices.len().max(1);
    let mut sensor_data = leto::Array2::<f64>::zeros((n_sensors, 1));
    for (idx, &(i, j, k)) in sensor_indices.iter().enumerate() {
        let node_idx = i + req.grid.nx * (j + req.grid.ny * k);
        sensor_data[[idx, 0]] = solver.solution()[node_idx].norm();
    }

    let sensor_data = trim_initial_recorder_sample(sensor_data, 1, req.record_start_index);
    let stats = if !sensor_indices.is_empty() {
        Some(SampledStatistics {
            p_max: {
                let v: Vec<f64> = sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| row[0])
                    .collect();
                leto::Array1::from_vec(v.len(), v).expect("invariant: length matches row count")
            },
            p_min: {
                let v: Vec<f64> = sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| row[0])
                    .collect();
                leto::Array1::from_vec(v.len(), v).expect("invariant: length matches row count")
            },
            p_rms: {
                let v: Vec<f64> = sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| row[0])
                    .collect();
                leto::Array1::from_vec(v.len(), v).expect("invariant: length matches row count")
            },
            p_final: {
                let v: Vec<f64> = sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| row[0])
                    .collect();
                leto::Array1::from_vec(v.len(), v).expect("invariant: length matches row count")
            },
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
