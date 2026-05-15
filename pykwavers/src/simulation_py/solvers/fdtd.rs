use kwavers::core::error::KwaversResult;
use kwavers::domain::boundary::cpml::CPMLConfig;
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::sensor::recorder::simple::SensorRecorder;
use kwavers::domain::source::{GridSource, Source as KwaversSource};
use kwavers::solver::forward::fdtd::config::{FdtdConfig, KSpaceCorrectionMode};
use kwavers::solver::forward::fdtd::solver::FdtdSolver;
use kwavers::solver::geometry::Geometry;
use kwavers::solver::interface::solver::Solver as SolverTrait;

use crate::medium_py::MediumInner;
use crate::sensor_py::Sensor;
use crate::simulation_result_py::{extract_full_grid_stats, SimulationRunResult};
use crate::transducer_array_py::TransducerArray2D;

use super::super::Simulation;

impl Simulation {
    /// Run FDTD simulation (internal).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_fdtd_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        grid_source: GridSource,
        sources: Vec<Box<dyn KwaversSource>>,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        pml_size: Option<usize>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        _pml_inside: bool,
        pml_alpha_xyz: Option<(f64, f64, f64)>,
        kspace_correction: KSpaceCorrectionMode,
        enable_nonlinear: bool,
        axisymmetric: bool,
        record_modes: &[String],
        record_start_index: usize,
    ) -> KwaversResult<SimulationRunResult> {
        

        let sensor_mask = Self::create_sensor_mask(grid, sensor, transducer_sensor);
        let transducer_ordered_indices = transducer_sensor
            .map(|trans| Self::create_transducer_ordered_indices(grid, &trans.inner));

        let geometry = if axisymmetric {
            Geometry::CylindricalAS
        } else {
            Geometry::Cartesian3D
        };

        let config = FdtdConfig {
            dt,
            nt: time_steps,
            spatial_order: 4,
            staggered_grid: true,
            cfl_factor: 0.3,
            subgridding: false,
            subgrid_factor: 2,
            enable_gpu_acceleration: false,
            enable_nonlinear,
            kspace_correction,
            sensor_mask: Some(sensor_mask.clone()),
            geometry,
        };

        let mut solver = FdtdSolver::new(config, grid, medium.as_medium(), grid_source)?;

        let modes = Self::recording_modes_from_strings(record_modes);
        if !modes.is_empty() {
            let shape = (grid.nx, grid.ny, grid.nz);
            solver.sensor_recorder =
                SensorRecorder::with_modes(Some(&sensor_mask), shape, time_steps + 1, &modes)?;
        } else if let Some(ordered) = transducer_ordered_indices {
            solver.sensor_recorder = SensorRecorder::from_ordered_indices(ordered, time_steps + 1)?;
        }

        let (default_thickness, max_allowed) =
            Self::cpml_thickness_limits(grid.nx, grid.ny, grid.nz);
        let thickness = pml_size.unwrap_or(default_thickness).min(max_allowed);

        if thickness > 0 && max_allowed > 0 {
            let mut cpml_config = if let Some((px, py, pz)) = pml_size_xyz {
                CPMLConfig::with_per_dimension_thickness(px, py, pz)
            } else {
                CPMLConfig::with_thickness(thickness)
            };
            if let Some((ax, ay, az)) = pml_alpha_xyz {
                cpml_config = cpml_config.with_alpha_xyz(ax, ay, az);
            }
            let max_c = medium.as_medium().max_sound_speed();
            solver.enable_cpml(cpml_config, dt, max_c)?;
        }

        for source in sources {
            SolverTrait::add_source(&mut solver, source)?;
        }

        solver.run_orchestrated(time_steps)?;

        let stats = solver.sensor_recorder.extract_all_stats();

        let full_data = solver.extract_recorded_sensor_data().ok_or_else(|| {
            kwavers::core::error::KwaversError::Io(std::io::Error::other("No sensor data recorded"))
        })?;
        let sensor_data =
            Self::trim_initial_recorder_sample(full_data, time_steps, record_start_index);

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
        })
    }
}
