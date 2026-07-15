//! BEM solver dispatch.

use kwavers_math::fft::Complex64;
use leto::{Array1, Array3};
use std::f64::consts::TAU;

use crate::dispatch::shared::trim_initial_recorder_sample;
use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_core::error::KwaversResult;
use kwavers_mesh::tetrahedral::TetrahedralMesh;
use kwavers_receiver::recorder::pressure_statistics::SampledStatistics;
use kwavers_solver::forward::bem::{BemConfig, BemSolution, BemSolver};

/// Run a boundary-element method simulation for the given request.
pub fn run(req: &SimulationRunRequest<'_>) -> KwaversResult<SimulationRunResult> {
    let c_max = req.medium.max_sound_speed();
    let (wavenumber, frequency) = if let Some(freq) = req.helmholtz.and_then(|h| h.frequency) {
        (TAU * freq / c_max, freq)
    } else {
        (TAU / (c_max * req.dt), 1.0 / req.dt)
    };

    let mesh = TetrahedralMesh::from_grid_vertices(req.grid)?;
    let config = BemConfig {
        wavenumber,
        sound_speed: c_max,
        frequency,
        tolerance: 5e-4,
        ..BemConfig::default()
    };

    let mut solver = BemSolver::from_mesh(config, &mesh)?;
    let n_boundary = solver.vertices.len();
    let nx = req.grid.nx;
    let ny = req.grid.ny;
    let nz = req.grid.nz;

    let mut dirichlet_nodes: Vec<usize> = Vec::new();
    if let Some(ref p0) = req.grid_source.p0 {
        for i in 0..nx {
            for j in 0..ny {
                for &k in &[0, nz - 1] {
                    let global_idx = i + nx * (j + ny * k);
                    if let Some(local_idx) = solver.local_index(global_idx) {
                        let val = p0[[i, j, k]];
                        if val.abs() > 0.0 && !dirichlet_nodes.contains(&local_idx) {
                            dirichlet_nodes.push(local_idx);
                            solver
                                .boundary_manager()
                                .add_dirichlet(vec![(local_idx, Complex64::new(val, 0.0))]);
                        }
                    }
                }
            }
            for k in 0..nz {
                for &j in &[0, ny - 1] {
                    let global_idx = i + nx * (j + ny * k);
                    if let Some(local_idx) = solver.local_index(global_idx) {
                        let val = p0[[i, j, k]];
                        if val.abs() > 0.0 && !dirichlet_nodes.contains(&local_idx) {
                            dirichlet_nodes.push(local_idx);
                            solver
                                .boundary_manager()
                                .add_dirichlet(vec![(local_idx, Complex64::new(val, 0.0))]);
                        }
                    }
                }
            }
        }
        for j in 0..ny {
            for k in 0..nz {
                for &i in &[0, nx - 1] {
                    let global_idx = i + nx * (j + ny * k);
                    if let Some(local_idx) = solver.local_index(global_idx) {
                        let val = p0[[i, j, k]];
                        if val.abs() > 0.0 && !dirichlet_nodes.contains(&local_idx) {
                            dirichlet_nodes.push(local_idx);
                            solver
                                .boundary_manager()
                                .add_dirichlet(vec![(local_idx, Complex64::new(val, 0.0))]);
                        }
                    }
                }
            }
        }
    }

    {
        let radiation_nodes: Vec<usize> = (0..n_boundary)
            .filter(|n| !dirichlet_nodes.contains(n))
            .collect();
        if !radiation_nodes.is_empty() {
            solver.boundary_manager().add_radiation(radiation_nodes);
        }
    }

    let bemsol: BemSolution = solver.solve(wavenumber, None)?;

    let sensor_mask = req
        .sensor_mask
        .clone()
        .unwrap_or_else(|| Array3::from_elem((nx, ny, nz), false));
    let sensor_indices: Vec<(usize, usize, usize)> = sensor_mask
        .indexed_iter()
        .filter(|(_, &active)| active)
        .map(|([i, j, k], _)| (i, j, k))
        .collect();

    let n_sensors = sensor_indices.len().max(1);
    let mut sensor_data = leto::Array2::<f64>::zeros((n_sensors, 1));

    let eval_points: Vec<[f64; 3]> = sensor_indices
        .iter()
        .map(|&(i, j, k)| {
            [
                req.grid.origin[0] + i as f64 * req.grid.dx,
                req.grid.origin[1] + j as f64 * req.grid.dy,
                req.grid.origin[2] + k as f64 * req.grid.dz,
            ]
        })
        .collect();

    if !eval_points.is_empty() {
        let points_arr = Array1::from_vec(eval_points.len(), eval_points)
            .expect("invariant: 1-D length matches");
        let scattered = solver.compute_scattered_field(&points_arr, &bemsol)?;
        for (idx, &val) in scattered.iter().enumerate() {
            sensor_data[[idx, 0]] = val.norm();
        }
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
                Array1::from_vec(v.len(), v).expect("invariant: length matches row count")
            }
            .into(),
            p_min: {
                let v: Vec<f64> = sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| row[0])
                    .collect();
                Array1::from_vec(v.len(), v).expect("invariant: length matches row count")
            }
            .into(),
            p_rms: {
                let v: Vec<f64> = sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| row[0])
                    .collect();
                Array1::from_vec(v.len(), v).expect("invariant: length matches row count")
            }
            .into(),
            p_final: {
                let v: Vec<f64> = sensor_data
                    .rows()
                    .expect("invariant: rank-2 rows")
                    .map(|row| row[0])
                    .collect();
                Array1::from_vec(v.len(), v).expect("invariant: length matches row count")
            }
            .into(),
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
