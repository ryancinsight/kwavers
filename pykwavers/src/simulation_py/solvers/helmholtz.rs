use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::sensor::recorder::pressure_statistics::SampledStatistics;
use kwavers::domain::source::GridSource;
use kwavers::solver::forward::helmholtz::fem::{
    FemHelmholtzConfig, FemHelmholtzSolver, FemPreconditionerType,
};
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::TAU;

use crate::medium_py::MediumInner;
use crate::sensor_py::Sensor;
use crate::simulation_result_py::SimulationRunResult;
use crate::transducer_array_py::TransducerArray2D;

use super::super::Simulation;

/// Map a structured-grid index `(i, j, k)` to the 0‑based FEM node index
/// produced by [`TetrahedralMesh::from_grid_vertices`].
///
/// # Contract
///
/// `from_grid_vertices` visits nodes in the innermost-loop order `k → j → i`,
/// producing the flattened index `i + nx*(j + ny*k)`. This function is the
/// inverse bijection when the mesh and grid dimensions agree.
#[inline]
fn grid_vertex_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
    i + nx * (j + ny * k)
}

impl Simulation {
    /// Run the frequency-domain FEM Helmholtz solver.
    ///
    /// The Helmholtz equation `∇²u + k²u = −f` is solved on a tetrahedral mesh
    /// derived from the Cartesian grid via
    /// [`TetrahedralMesh::from_grid_vertices`]. The source term `f` is populated
    /// from `grid_source.p0` (initial pressure mapped to nodal loads), and the
    /// complex-valued steady-state field is evaluated at sensor positions.
    ///
    /// # Frequency-domain semantics
    ///
    /// Unlike time-domain solvers (FDTD/PSTD/DG), the Helmholtz solver produces
    /// a single steady-state snapshot. The `time_steps` parameter is ignored;
    /// a single column of sensor data is returned representing the complex
    /// pressure magnitude at each sensor location.
    ///
    /// The wavenumber `k` is derived from `helmholtz_frequency` when set:
    /// `k = 2π · f / cₘₐₓ`. When `helmholtz_frequency` is `None` (the
    /// default), the wavenumber falls back to the time-step proxy:
    /// `k = 2π / (cₘₐₓ · dt)`. Users should call
    /// [`Simulation::set_helmholtz_wavenumber`] to decouple the
    /// frequency-domain solve from the time step.
    ///
    /// # Limitations
    ///
    /// - Only supports initial-condition (`p0`) sourcing; time-varying sources
    ///   and transducers are silently ignored.
    /// - CPML and nonlinear propagation are not applicable to the Helmholtz
    ///   formulation.
    /// - ILU and AMG preconditioners return `FeatureNotAvailable`; only
    ///   `None` and `Diagonal` (Jacobi) are usable today.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_helmholtz_impl(
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

        let config = FemHelmholtzConfig {
            wavenumber,
            tolerance: 5e-4,
            max_iterations: 5000,
            preconditioner: FemPreconditionerType::Diagonal,
            ..FemHelmholtzConfig::default()
        };

        let mut solver = FemHelmholtzSolver::from_grid(config, grid)?;
        solver.assemble_system(medium.as_medium())?;

        // ── Source injection ──────────────────────────────────────────────
        if let Some(ref p0) = grid_source.p0 {
            let nx = grid.nx;
            let ny = grid.ny;
            for ((i, j, k), &val) in p0.indexed_iter() {
                if val.abs() > 0.0 {
                    let node_idx = grid_vertex_index(nx, ny, i, j, k);
                    solver.add_nodal_load(node_idx, Complex64::new(val, 0.0))?;
                }
            }
        } else if grid_source.p_mask.is_some() || grid_source.u_mask.is_some() {
            eprintln!(
                "[Helmholtz] grid_source.p0 is None but time-varying sources are configured — \
                 the Helmholtz solver only supports initial-condition (p0) snapshots."
            );
        }

        solver.solve_system()?;

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

        for (idx, &(i, j, k)) in sensor_indices.iter().enumerate() {
            let node_idx = grid_vertex_index(grid.nx, grid.ny, i, j, k);
            let val = solver.solution()[node_idx];
            sensor_data[[idx, 0]] = val.norm();
        }

        let sensor_data =
            Self::trim_initial_recorder_sample(sensor_data, 1, record_start_index);

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
