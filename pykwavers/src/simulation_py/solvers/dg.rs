use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::sensor::recorder::pressure_statistics::SampledStatistics;
use kwavers::domain::source::GridSource;
use kwavers::solver::forward::pstd::dg::{HybridSpectralDGConfig, HybridSpectralDGSolver};
use ndarray::Array1;
use std::sync::Arc;

use crate::medium_py::MediumInner;
use crate::sensor_py::Sensor;
use crate::simulation_result_py::SimulationRunResult;
use crate::transducer_array_py::TransducerArray2D;

use super::super::Simulation;

impl Simulation {
    /// Run the hybrid Spectral-DG solver.
    ///
    /// The Discontinuous Galerkin solver operates on a single scalar pressure field
    /// with automatic discontinuity detection and spectral/DG hybrid switching.
    /// This solver is suited for problems with sharp interfaces, shock-like features,
    /// or heterogeneous media where high-order spectral accuracy must coexist with
    /// discontinuity-resolving DG elements.
    ///
    /// Unlike FDTD/PSTD, the DG solver does not currently support:
    /// - CPML absorbing boundaries
    /// - Time-varying sources (only initial-condition `p0` snapshots)
    /// - Staggered velocity/intensity recording
    /// - Nonlinear propagation or power-law absorption
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_dg_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        grid_source: GridSource,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        record_start_index: usize,
    ) -> KwaversResult<SimulationRunResult> {
        let sensor_mask = Self::create_sensor_mask(grid, sensor, transducer_sensor);
        let c = medium.as_medium().max_sound_speed();

        if grid_source.p0.is_none()
            && (grid_source.p_mask.is_some() || grid_source.u_mask.is_some())
        {
            eprintln!(
                "[DG] grid_source.p0 is None but time-varying sources are configured — \
                 the DG solver only supports initial-condition (p0) snapshots. \
                 Field will be initialised from quiescent zeros."
            );
        }

        let config = HybridSpectralDGConfig::default();
        let grid_arc = Arc::new(grid.clone());
        let mut solver = HybridSpectralDGSolver::new(config, grid_arc);

        // Initialise pressure field from grid-source p0 or start from quiescent zeros
        let mut field = grid_source
            .p0
            .unwrap_or_else(|| ndarray::Array3::zeros((grid.nx, grid.ny, grid.nz)));
        let mut output = ndarray::Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

        // Collect sensor indices once for fast extraction inside the time loop
        let sensor_indices: Vec<(usize, usize, usize)> = sensor_mask
            .indexed_iter()
            .filter(|(_, &active)| active)
            .map(|((i, j, k), _)| (i, j, k))
            .collect();
        let n_sensors = sensor_indices.len().max(1);
        let mut sensor_data =
            ndarray::Array2::<f64>::zeros((n_sensors, time_steps + 1));

        // Record t=0 state
        for (idx, &(i, j, k)) in sensor_indices.iter().enumerate() {
            sensor_data[[idx, 0]] = field[[i, j, k]];
        }

        // Time-stepping loop — each step advances the scalar pressure field
        for step in 0..time_steps {
            solver.solve_step_into(&field, dt, c, &mut output)?;

            for (idx, &(i, j, k)) in sensor_indices.iter().enumerate() {
                sensor_data[[idx, step + 1]] = output[[i, j, k]];
            }

            // Swap buffers: output becomes next-step input
            std::mem::swap(&mut field, &mut output);
        }

        let sensor_data =
            Self::trim_initial_recorder_sample(sensor_data, time_steps, record_start_index);

        // Compute per-sensor pressure statistics from the recorded time series.
        // Only emit stats when actual sensors are present (not the phantom 1-row
        // fallback created when no sensor mask points are active).
        let stats = if !sensor_indices.is_empty() {
            let n_cols = sensor_data.ncols();
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
                    .map(|row| {
                        let sq_sum: f64 = row.iter().map(|v| v * v).sum();
                        (sq_sum / n_cols as f64).sqrt()
                    })
                    .collect::<Array1<f64>>(),
                p_final: sensor_data
                    .rows()
                    .into_iter()
                    .map(|row| row[n_cols.saturating_sub(1)])
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
