use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::mesh::tetrahedral::TetrahedralMesh;
use kwavers::domain::sensor::recorder::pressure_statistics::SampledStatistics;
use kwavers::domain::source::GridSource;
use kwavers::solver::forward::bem::{BemConfig, BemSolver, BemSolution};
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::TAU;

use crate::medium_py::MediumInner;
use crate::sensor_py::Sensor;
use crate::simulation_result_py::SimulationRunResult;
use crate::transducer_array_py::TransducerArray2D;

use super::super::Simulation;

impl Simulation {
    /// Run the Boundary Element Method (BEM) solver.
    ///
    /// The BEM solver discretises only the boundary surface, making it efficient
    /// for exterior-domain problems (scattering, radiation). A tetrahedral mesh
    /// is generated from the Cartesian grid via
    /// [`TetrahedralMesh::from_grid_vertices`], and the boundary surface is
    /// extracted automatically.
    ///
    /// # Boundary conditions
    ///
    /// - **Dirichlet** BCs are applied at boundary nodes where `grid_source.p0`
    ///   has non-zero values.
    /// - **Radiation** (Sommerfeld) BCs are applied on all remaining boundary
    ///   nodes, modelling an open domain with no reflections.
    ///
    /// # Frequency-domain semantics
    ///
    /// Like the Helmholtz solver, BEM produces a single steady-state snapshot.
    /// The wavenumber is derived from `helmholtz_frequency` when set, or falls
    /// back to the dt proxy: `k = 2π / (cₘₐₓ · dt)`.
    ///
    /// # Limitations
    ///
    /// - Only supports initial-condition (`p0`) sourcing mapped to boundary
    ///   Dirichlet BCs; time-varying sources and transducers are silently ignored.
    /// - CPML, nonlinear propagation, and power-law absorption are not applicable
    ///   to the BEM formulation.
    /// - Sound-soft (pressure-release, p=0) and rigid (∂p/∂n=0) scattering
    ///   require dedicated incident-field setup not yet exposed via `run()`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_bem_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        _time_steps: usize,
        dt: f64,
        helmholtz_frequency: Option<f64>,
        grid_source: GridSource,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        record_start_index: usize,
    ) -> KwaversResult<SimulationRunResult> {
        let c_max = medium.as_medium().max_sound_speed();

        // Derive wavenumber: prefer the user-set frequency when available;
        // otherwise fall back to the dt proxy: k = 2π / (cₘₐₓ · dt).
        let wavenumber = if let Some(freq) = helmholtz_frequency {
            TAU * freq / c_max
        } else {
            TAU / (c_max * dt)
        };

        // ── Generate tetrahedral mesh from Cartesian grid ──────────────────
        let mesh = TetrahedralMesh::from_grid_vertices(grid)?;

        // ── Build BEM solver from boundary surface ────────────────────────
        let config = BemConfig {
            wavenumber,
            sound_speed: c_max,
            frequency: if let Some(f) = helmholtz_frequency {
                f
            } else {
                1.0 / dt
            },
            tolerance: 5e-4, // 1e-8 is too tight for BiCGSTAB on these grid sizes
            ..BemConfig::default()
        };
        let mut solver = BemSolver::from_mesh(config, &mesh)?;
        let n_boundary = solver.vertices.len();

        // ── Boundary conditions ───────────────────────────────────────────
        // Map p0 → Dirichlet BCs on boundary nodes.
        // The global mesh node index matches the grid-vertex bijection:
        //    global_idx = i + nx * (j + ny * k)
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        let mut dirichlet_nodes: Vec<usize> = Vec::new();

        if let Some(ref p0) = grid_source.p0 {
            // Only iterate over the 6 faces of the bounding box (boundary nodes)
            // to avoid O(nx·ny·nz) scans. BEM only has boundary dofs.
            for i in 0..nx {
                for j in 0..ny {
                    for &k in &[0, nz - 1] {
                        let global_idx = i + nx * (j + ny * k);
                        if let Some(local_idx) = solver.local_index(global_idx) {
                            let val = p0[[i, j, k]];
                            if val.abs() > 0.0 && !dirichlet_nodes.contains(&local_idx) {
                                dirichlet_nodes.push(local_idx);
                                solver.boundary_manager().add_dirichlet(vec![(
                                    local_idx,
                                    Complex64::new(val, 0.0),
                                )]);
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
                                solver.boundary_manager().add_dirichlet(vec![(
                                    local_idx,
                                    Complex64::new(val, 0.0),
                                )]);
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
                                solver.boundary_manager().add_dirichlet(vec![(
                                    local_idx,
                                    Complex64::new(val, 0.0),
                                )]);
                            }
                        }
                    }
                }
            }
        }

        // Apply radiation (Sommerfeld) BCs on all non-Dirichlet boundary nodes
        {
            let radiation_nodes: Vec<usize> = (0..n_boundary)
                .filter(|n| !dirichlet_nodes.contains(n))
                .collect();
            if !radiation_nodes.is_empty() {
                solver.boundary_manager().add_radiation(radiation_nodes);
            }
        }

        if grid_source.p0.is_none()
            && (grid_source.p_mask.is_some() || grid_source.u_mask.is_some())
        {
            eprintln!(
                "[BEM] grid_source.p0 is None but time-varying sources are configured — \
                 the BEM solver only supports initial-condition (p0) boundary Dirichlet BCs."
            );
        }

        // ── Solve ─────────────────────────────────────────────────────────
        let bemsol: BemSolution = solver.solve(wavenumber, None)?;

        // ── Sensor extraction ────────────────────────────────────────────
        let sensor_mask = Self::create_sensor_mask(grid, sensor, transducer_sensor);
        let sensor_indices: Vec<(usize, usize, usize)> = sensor_mask
            .indexed_iter()
            .filter(|(_, &active)| active)
            .map(|((i, j, k), _)| (i, j, k))
            .collect();

        let n_sensors = sensor_indices.len().max(1);
        let n_cols = 1_usize; // single steady-state snapshot
        let mut sensor_data = ndarray::Array2::<f64>::zeros((n_sensors, n_cols));

        // Convert sensor grid indices to physical coordinates
        let eval_points: Vec<[f64; 3]> = sensor_indices
            .iter()
            .map(|&(i, j, k)| {
                [
                    grid.origin[0] + i as f64 * grid.dx,
                    grid.origin[1] + j as f64 * grid.dy,
                    grid.origin[2] + k as f64 * grid.dz,
                ]
            })
            .collect();

        if !eval_points.is_empty() {
            let points_arr = Array1::from_vec(eval_points);
            let scattered = solver.compute_scattered_field(&points_arr, &bemsol)?;
            for (idx, &val) in scattered.iter().enumerate() {
                sensor_data[[idx, 0]] = val.norm();
            }
        }

        let sensor_data =
            Self::trim_initial_recorder_sample(sensor_data, 1, record_start_index);

        // Compute per-sensor pressure statistics
        let stats = if !sensor_indices.is_empty() {
            Some(SampledStatistics {
                p_max: sensor_data
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
                    .collect::<Array1<f64>>(),
                p_min: sensor_data
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().fold(f64::INFINITY, |a, &b| a.min(b)))
                    .collect::<Array1<f64>>(),
                p_rms: sensor_data
                    .rows()
                    .into_iter()
                    .map(|row| (row[0] * row[0]).sqrt())
                    .collect::<Array1<f64>>(),
                p_final: sensor_data
                    .rows()
                    .into_iter()
                    .map(|row| row[0])
                    .collect::<Array1<f64>>(),
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
}
