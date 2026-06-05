use kwavers_boundary::cpml::CPMLConfig;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid as KwaversGrid;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers_receiver::recorder::simple::SensorRecorder;
use kwavers_solver::forward::pstd::config::{BoundaryConfig, CompatibilityMode, PSTDConfig};
use kwavers_solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers_solver::geometry::SolverGeometry;
use kwavers_solver::interface::solver::Solver as SolverTrait;
use kwavers_source::{GridSource, Source as KwaversSource};

use crate::medium_py::MediumInner;
use crate::sensor_py::Sensor;
use crate::simulation_result_py::{extract_full_grid_stats, SimulationRunResult};
use crate::transducer_array_py::TransducerArray2D;

use super::super::Simulation;

impl Simulation {
    /// Build and configure a PSTD solver without running it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_pstd_solver(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        compatibility_mode: CompatibilityMode,
        enable_nonlinear: bool,
        alpha_coeff_db: f64,
        alpha_power: f64,
        grid_source: GridSource,
        sources: Vec<Box<dyn KwaversSource>>,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        pml_size: Option<usize>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        pml_inside: bool,
        pml_alpha_xyz: Option<(f64, f64, f64)>,
        axisymmetric: bool,
        record_modes: &[String],
    ) -> KwaversResult<(PSTDSolver, KwaversGrid, ndarray::Array3<bool>)> {
        use kwavers_core::error::ValidationError;

        let sensor_mask = Self::create_sensor_mask(grid, sensor, transducer_sensor);
        let transducer_ordered_indices = transducer_sensor
            .map(|trans| Self::create_transducer_ordered_indices(grid, &trans.inner));

        let (default_thickness, max_allowed) =
            Self::cpml_thickness_limits(grid.nx, grid.ny, grid.nz);
        let thickness = pml_size.unwrap_or(default_thickness).min(max_allowed);

        let (sim_grid, grid_source, sensor_mask, effective_pml_inside) =
            if !pml_inside && thickness > 0 {
                if transducer_sensor.is_some() {
                    return Err(KwaversError::Validation(ValidationError::FieldValidation {
                        field: "pml_inside".to_string(),
                        value: "false".to_string(),
                        constraint: "pml_inside=false is not supported with transducer sensors"
                            .to_string(),
                    }));
                }
                let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
                let p = thickness;
                let pad_x = nx > 1;
                let pad_y = !axisymmetric && ny > 1;
                let pad_z_two_sided = !axisymmetric && nz > 1;
                let pad_z_one_sided = axisymmetric && nz > 1;

                let pnx = if pad_x { nx + 2 * p } else { nx };
                let pny = if pad_y { ny + 2 * p } else { ny };
                let pnz = if pad_z_two_sided {
                    nz + 2 * p
                } else if pad_z_one_sided {
                    nz + p
                } else {
                    nz
                };
                let padded_grid = KwaversGrid::new(pnx, pny, pnz, grid.dx, grid.dy, grid.dz)?;

                let px_embed = if pad_x { p } else { 0 };
                let py = if pad_y { p } else { 0 };
                let pz_embed = if pad_z_two_sided { p } else { 0 };
                let p = px_embed;

                let embed = |arr: ndarray::Array3<f64>| -> ndarray::Array3<f64> {
                    let mut out = ndarray::Array3::<f64>::zeros((pnx, pny, pnz));
                    out.slice_mut(ndarray::s![p..nx + p, py..ny + py, pz_embed..nz + pz_embed])
                        .assign(&arr);
                    out
                };

                let mut padded_mask = ndarray::Array3::<bool>::from_elem((pnx, pny, pnz), false);
                padded_mask
                    .slice_mut(ndarray::s![p..nx + p, py..ny + py, pz_embed..nz + pz_embed])
                    .assign(&sensor_mask);

                let padded_source = GridSource {
                    p0: grid_source.p0.map(&embed),
                    u0: grid_source
                        .u0
                        .map(|(ux, uy, uz)| (embed(ux), embed(uy), embed(uz))),
                    p_mask: grid_source.p_mask.map(&embed),
                    p_signal: grid_source.p_signal,
                    p_mode: grid_source.p_mode,
                    u_mask: grid_source.u_mask.map(embed),
                    u_signal: grid_source.u_signal,
                    u_mode: grid_source.u_mode,
                };

                (padded_grid, padded_source, padded_mask, true)
            } else {
                (grid.clone(), grid_source, sensor_mask, pml_inside)
            };

        let alpha_is_zero = pml_alpha_xyz
            .map(|(ax, ay, az)| ax == 0.0 && ay == 0.0 && az == 0.0)
            .unwrap_or(false);
        let boundary = if thickness > 0 && max_allowed > 0 && !alpha_is_zero {
            let mut cpml_config = if let Some((px, py, pz)) = pml_size_xyz {
                CPMLConfig::with_per_dimension_thickness(px, py, pz)
            } else {
                CPMLConfig::with_thickness(thickness)
            };
            if let Some((ax, ay, az)) = pml_alpha_xyz {
                cpml_config = cpml_config.with_alpha_xyz(ax, ay, az);
            }
            if axisymmetric && !pml_inside {
                cpml_config = cpml_config.with_radial_inner_z_transparent();
            }
            BoundaryConfig::CPML(cpml_config)
        } else {
            BoundaryConfig::None
        };

        let effective_alpha_db = if alpha_coeff_db > 0.0 {
            alpha_coeff_db
        } else {
            medium.as_medium().alpha_coefficient(0.0, 0.0, 0.0, grid)
        };

        let effective_alpha_power = {
            let y_medium = medium.as_medium().alpha_power(0.0, 0.0, 0.0, grid);
            if alpha_coeff_db <= 0.0 && y_medium > 0.0 && (y_medium - 1.0).abs() > 1e-12 {
                y_medium
            } else {
                alpha_power
            }
        };

        let absorption_mode = if effective_alpha_db > 0.0 {
            AbsorptionMode::PowerLaw {
                alpha_coeff: effective_alpha_db,
                alpha_power: effective_alpha_power,
            }
        } else {
            AbsorptionMode::Lossless
        };

        let geometry = if axisymmetric {
            SolverGeometry::CylindricalAS
        } else {
            SolverGeometry::Cartesian3D
        };

        let config = PSTDConfig {
            dt,
            nt: time_steps,
            compatibility_mode,
            sensor_mask: Some(sensor_mask.clone()),
            boundary,
            pml_inside: effective_pml_inside,
            absorption_mode,
            nonlinearity: enable_nonlinear,
            geometry,
            ..Default::default()
        };

        let mut solver =
            PSTDSolver::new(config, sim_grid.clone(), medium.as_medium(), grid_source)?;

        let spec = Self::record_modes_to_spec(record_modes);
        let shape = (sim_grid.nx, sim_grid.ny, sim_grid.nz);
        if let Some(ordered) = transducer_ordered_indices {
            solver.sensor_recorder = SensorRecorder::from_ordered_indices(ordered, time_steps + 1)?;
        } else {
            solver.sensor_recorder =
                SensorRecorder::with_spec(Some(&sensor_mask), shape, time_steps + 1, spec)?;
        }

        for source in sources {
            SolverTrait::add_source(&mut solver, source)?;
        }

        Ok((solver, sim_grid, sensor_mask))
    }

    /// Run PSTD for `checkpoint_steps` steps and save state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_pstd_to_checkpoint(
        grid: &KwaversGrid,
        medium: &MediumInner,
        total_steps: usize,
        checkpoint_steps: usize,
        dt: f64,
        compatibility_mode: CompatibilityMode,
        enable_nonlinear: bool,
        alpha_coeff_db: f64,
        alpha_power: f64,
        grid_source: GridSource,
        sources: Vec<Box<dyn KwaversSource>>,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        pml_size: Option<usize>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        pml_inside: bool,
        pml_alpha_xyz: Option<(f64, f64, f64)>,
        checkpoint_path: &std::path::Path,
    ) -> KwaversResult<()> {
        let (mut solver, _sim_grid, _sensor_mask) = Self::prepare_pstd_solver(
            grid,
            medium,
            total_steps,
            dt,
            compatibility_mode,
            enable_nonlinear,
            alpha_coeff_db,
            alpha_power,
            grid_source,
            sources,
            sensor,
            transducer_sensor,
            pml_size,
            pml_size_xyz,
            pml_inside,
            pml_alpha_xyz,
            false,
            &[],
        )?;
        solver.run_to_checkpoint(checkpoint_steps, checkpoint_path)
    }

    /// Resume a checkpointed PSTD simulation from a preloaded checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_pstd_from_checkpoint_loaded(
        grid: &KwaversGrid,
        medium: &MediumInner,
        total_steps: usize,
        dt: f64,
        compatibility_mode: CompatibilityMode,
        enable_nonlinear: bool,
        alpha_coeff_db: f64,
        alpha_power: f64,
        grid_source: GridSource,
        sources: Vec<Box<dyn KwaversSource>>,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        pml_size: Option<usize>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        pml_inside: bool,
        pml_alpha_xyz: Option<(f64, f64, f64)>,
        checkpoint: kwavers_solver::forward::pstd::checkpoint::PSTDCheckpoint,
        remaining_steps: usize,
        checkpoint_path: &std::path::Path,
        record_modes: &[String],
    ) -> KwaversResult<SimulationRunResult> {
        let (mut solver, _sim_grid, _sensor_mask) = Self::prepare_pstd_solver(
            grid,
            medium,
            total_steps,
            dt,
            compatibility_mode,
            enable_nonlinear,
            alpha_coeff_db,
            alpha_power,
            grid_source,
            sources,
            sensor,
            transducer_sensor,
            pml_size,
            pml_size_xyz,
            pml_inside,
            pml_alpha_xyz,
            false,
            record_modes,
        )?;

        checkpoint.validate_restore_contract(grid.nx, grid.ny, grid.nz, total_steps, dt)?;

        let full_data = solver
            .run_from_checkpoint_loaded(checkpoint, checkpoint_path, remaining_steps)?
            .ok_or_else(|| KwaversError::Io(std::io::Error::other("No sensor data recorded")))?;

        let stats = solver.sensor_recorder.extract_all_stats();
        let sensor_data = Self::trim_initial_recorder_sample(full_data, total_steps, 1);

        let ux_data = solver
            .sensor_recorder
            .recorded_ux_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let uy_data = solver
            .sensor_recorder
            .recorded_uy_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let uz_data = solver
            .sensor_recorder
            .recorded_uz_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let ix_data = solver
            .sensor_recorder
            .recorded_ix_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let iy_data = solver
            .sensor_recorder
            .recorded_iy_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let iz_data = solver
            .sensor_recorder
            .recorded_iz_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let i_avg_x = solver.sensor_recorder.extract_i_avg_x();
        let i_avg_y = solver.sensor_recorder.extract_i_avg_y();
        let i_avg_z = solver.sensor_recorder.extract_i_avg_z();
        let velocity_stats = solver.sensor_recorder.extract_sampled_velocity_stats();
        let full_grid_stats = extract_full_grid_stats(&solver.sensor_recorder);

        Ok(SimulationRunResult {
            sensor_data,
            stats,
            ux_data,
            uy_data,
            uz_data,
            ix_data,
            iy_data,
            iz_data,
            i_avg_x,
            i_avg_y,
            i_avg_z,
            velocity_stats,
            full_grid_stats,
            thermal_temperature: None,
            thermal_dose: None,
        })
    }
}
