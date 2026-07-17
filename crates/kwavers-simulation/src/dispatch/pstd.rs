//! PSTD solver dispatch.
//!
//! Handles PSTD, PSTD+Thermal, and GPU PSTD dispatch paths.

use leto::{Array3, SliceArg};

use crate::configs::ThermalConfig;
use crate::dispatch::shared::{record_modes_to_spec, trim_initial_recorder_view};
use crate::types::extract_full_grid_stats;
use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_boundary::cpml::CPMLConfig;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers_receiver::recorder::simple::SensorRecorder;
use kwavers_solver::forward::pstd::config::{BoundaryConfig, PSTDConfig};
use kwavers_solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers_solver::geometry::SolverGeometry;
use kwavers_solver::interface::solver::Solver as SolverTrait;
use kwavers_source::{GridSource, Source as KwaversSource};

fn embed_leto3(
    arr: &leto::Array3<f64>,
    padded_shape: [usize; 3],
    offset: [usize; 3],
) -> leto::Array3<f64> {
    let mut out = leto::Array3::<f64>::zeros(padded_shape);
    for ([i, j, k], &value) in arr.indexed_iter() {
        out[[i + offset[0], j + offset[1], k + offset[2]]] = value;
    }
    out
}

// ── PSTD (acoustic-only) ──────────────────────────────────────────────────────

/// Run a PSTD (acoustic-only) simulation.
pub fn run(
    req: &SimulationRunRequest<'_>,
    sources: Vec<Box<dyn KwaversSource>>,
) -> KwaversResult<SimulationRunResult> {
    let (mut solver, _sim_grid) = prepare_solver(req, sources)?;
    solver.run_orchestrated(req.time_steps)?;
    extract_result(&solver, req.time_steps, req.record_start_index)
}

// ── PSTD + Thermal ────────────────────────────────────────────────────────────

/// Run a PSTD simulation with thermal coupling.
pub fn run_with_thermal(
    req: &SimulationRunRequest<'_>,
    sources: Vec<Box<dyn KwaversSource>>,
    thermal_cfg: &ThermalConfig,
) -> KwaversResult<SimulationRunResult> {
    use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
    use kwavers_core::constants::medical::THERMAL_DOSE_REFERENCE_TEMP_C;
    use kwavers_core::constants::thermodynamic::KELVIN_OFFSET_C;
    use kwavers_medium::HomogeneousMedium;
    use kwavers_physics::thermal::diffusion::ThermalDiffusionConfig;
    use kwavers_solver::forward::thermal_diffusion::ThermalDiffusionSolver;

    let (mut solver, sim_grid) = prepare_solver(req, sources)?;

    let mut medium =
        HomogeneousMedium::new(thermal_cfg.density, SOUND_SPEED_TISSUE, 0.0, 0.0, &sim_grid);
    medium
        .set_thermal_properties(thermal_cfg.thermal_conductivity, thermal_cfg.specific_heat)
        .map_err(|e| KwaversError::InternalError(e.to_string()))?;

    let arterial_temp_k = thermal_cfg.arterial_temperature_c + KELVIN_OFFSET_C;
    let initial_temp_k = thermal_cfg.initial_temperature_c + KELVIN_OFFSET_C;

    let config = ThermalDiffusionConfig {
        enable_bioheat: thermal_cfg.enable_bioheat,
        perfusion_rate: thermal_cfg.perfusion_rate,
        blood_density: thermal_cfg.blood_density,
        blood_specific_heat: thermal_cfg.blood_specific_heat,
        arterial_temperature: arterial_temp_k,
        enable_hyperbolic: false,
        relaxation_time: 20.0,
        track_thermal_dose: thermal_cfg.track_thermal_dose,
        dose_reference_temperature: THERMAL_DOSE_REFERENCE_TEMP_C,
        spatial_order: 2,
    };

    let mut thermal_solver = ThermalDiffusionSolver::new(config, &sim_grid);
    thermal_solver.set_temperature(Array3::from_elem(
        (sim_grid.nx, sim_grid.ny, sim_grid.nz),
        initial_temp_k,
    ));

    let omega_c = std::f64::consts::TAU * thermal_cfg.center_frequency_hz;
    let dt_thermal = thermal_cfg
        .dt_thermal
        .unwrap_or(req.dt * thermal_cfg.n_acoustic_per_thermal as f64);
    let rho_cp = thermal_cfg.density * thermal_cfg.specific_heat;
    let background_heat_ks = thermal_cfg.metabolic_heat / rho_cp;

    solver.run_orchestrated_with_thermal(
        kwavers_solver::forward::pstd::implementation::core::orchestrator::thermal::ThermalOrchestrationInput {
            acoustic_steps: req.time_steps,
            thermal: &mut thermal_solver,
            thermal_medium: &medium,
            omega_c,
            dt_thermal,
            n_acoustic_per_thermal: thermal_cfg.n_acoustic_per_thermal,
            rho_cp,
            background_heat_ks,
        },
    )?;

    let mut result = extract_result(&solver, req.time_steps, req.record_start_index)?;
    result.thermal_temperature = Some(thermal_solver.temperature().to_owned());
    result.thermal_dose = thermal_solver.thermal_dose().map(|d| d.to_owned());
    Ok(result)
}

// ── GPU PSTD ──────────────────────────────────────────────────────────────────

/// Try GPU PSTD, fall back to CPU.
#[cfg(feature = "gpu")]
pub fn run_gpu_or_fallback(
    req: &SimulationRunRequest<'_>,
    sources: Vec<Box<dyn KwaversSource>>,
) -> KwaversResult<SimulationRunResult> {
    // TODO: integrate GPU PSTD dispatch
    run(req, sources)
}

// ── Solver preparation ────────────────────────────────────────────────────────

/// Build and configure a PSTD solver (without running it).
pub(crate) fn prepare_solver(
    req: &SimulationRunRequest<'_>,
    sources: Vec<Box<dyn KwaversSource>>,
) -> KwaversResult<(PSTDSolver, Grid)> {
    let sensor_mask = req
        .sensor_mask
        .clone()
        .unwrap_or_else(|| Array3::from_elem((req.grid.nx, req.grid.ny, req.grid.nz), false));

    let pml = req.pml.cloned().unwrap_or_default();
    let (default_thickness, max_allowed) =
        pml.effective_thickness(req.grid.nx, req.grid.ny, req.grid.nz);
    let thickness = pml.size.unwrap_or(default_thickness).min(max_allowed);

    let (sim_grid, grid_source, sensor_mask, effective_pml_inside) = if !pml.inside && thickness > 0
    {
        if req.transducer_ordered_indices.is_some() {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "pml_inside".to_string(),
                value: "false".to_string(),
                constraint: "pml_inside=false is not supported with transducer sensors".into(),
            }));
        }
        let (nx, ny, nz) = (req.grid.nx, req.grid.ny, req.grid.nz);
        let p = thickness;
        let pad_x = nx > 1;
        let pad_y = !req.axisymmetric && ny > 1;
        let pad_z_two_sided = !req.axisymmetric && nz > 1;
        let pad_z_one_sided = req.axisymmetric && nz > 1;

        let pnx = if pad_x { nx + 2 * p } else { nx };
        let pny = if pad_y { ny + 2 * p } else { ny };
        let pnz = if pad_z_two_sided {
            nz + 2 * p
        } else if pad_z_one_sided {
            nz + p
        } else {
            nz
        };
        let padded_grid = Grid::new(pnx, pny, pnz, req.grid.dx, req.grid.dy, req.grid.dz)?;

        let px_embed = if pad_x { p } else { 0 };
        let py = if pad_y { p } else { 0 };
        let pz_embed = if pad_z_two_sided { p } else { 0 };
        let offset = [px_embed, py, pz_embed];
        let padded_shape = [pnx, pny, pnz];

        let mut padded_mask = Array3::<bool>::from_elem((pnx, pny, pnz), false);
        padded_mask
            .slice_with_mut::<3>(&[
                SliceArg::Range {
                    start: Some(px_embed as isize),
                    end: Some((nx + px_embed) as isize),
                    step: 1,
                },
                SliceArg::Range {
                    start: Some(py as isize),
                    end: Some((ny + py) as isize),
                    step: 1,
                },
                SliceArg::Range {
                    start: Some(pz_embed as isize),
                    end: Some((nz + pz_embed) as isize),
                    step: 1,
                },
            ])
            .expect("invariant: padded-mask embed slice bounds")
            .assign(&sensor_mask);

        let padded_source = GridSource {
            p0: req
                .grid_source
                .p0
                .as_ref()
                .map(|a| embed_leto3(a, padded_shape, offset)),
            u0: req.grid_source.u0.as_ref().map(|(ux, uy, uz)| {
                (
                    embed_leto3(ux, padded_shape, offset),
                    embed_leto3(uy, padded_shape, offset),
                    embed_leto3(uz, padded_shape, offset),
                )
            }),
            p_mask: req
                .grid_source
                .p_mask
                .as_ref()
                .map(|a| embed_leto3(a, padded_shape, offset)),
            p_signal: req.grid_source.p_signal.clone(),
            p_mode: req.grid_source.p_mode,
            u_mask: req
                .grid_source
                .u_mask
                .as_ref()
                .map(|a| embed_leto3(a, padded_shape, offset)),
            u_signal: req.grid_source.u_signal.clone(),
            u_mode: req.grid_source.u_mode,
        };

        (padded_grid, padded_source, padded_mask, true)
    } else {
        let source = req.grid_source.clone();
        (req.grid.clone(), source, sensor_mask, pml.inside)
    };

    let boundary = if thickness > 0 && max_allowed > 0 && !pml.alpha_is_zero() {
        let mut cpml_config = if let Some((px, py, pz)) = pml.size_xyz {
            CPMLConfig::with_per_dimension_thickness(px, py, pz)
        } else {
            CPMLConfig::with_thickness(thickness)
        };
        if let Some((ax, ay, az)) = pml.alpha_xyz {
            cpml_config = cpml_config.with_alpha_xyz(ax, ay, az);
        }
        if req.axisymmetric && !pml.inside {
            cpml_config = cpml_config.with_radial_inner_z_transparent();
        }
        BoundaryConfig::CPML(cpml_config)
    } else {
        BoundaryConfig::None
    };

    let nl = req.nonlinear.cloned().unwrap_or_default();
    let effective_alpha_db = if nl.alpha_coeff > 0.0 {
        nl.alpha_coeff
    } else {
        req.medium.alpha_coefficient(0.0, 0.0, 0.0, req.grid)
    };

    let effective_alpha_power = {
        let y_medium = req.medium.alpha_power(0.0, 0.0, 0.0, req.grid);
        if nl.alpha_coeff <= 0.0 && y_medium > 0.0 && (y_medium - 1.0).abs() > 1e-12 {
            y_medium
        } else {
            nl.alpha_power
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

    let geometry = if req.axisymmetric {
        SolverGeometry::CylindricalAS
    } else {
        SolverGeometry::Cartesian3D
    };

    let config = PSTDConfig {
        dt: req.dt,
        nt: req.time_steps,
        compatibility_mode: req.compatibility_mode,
        sensor_mask: Some(sensor_mask.clone()),
        boundary,
        pml_inside: effective_pml_inside,
        absorption_mode,
        nonlinearity: nl.enabled,
        geometry,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(config, sim_grid.clone(), req.medium, grid_source)?;

    let spec = record_modes_to_spec(&req.record_modes);
    let shape = (sim_grid.nx, sim_grid.ny, sim_grid.nz);
    if let Some(ref ordered) = req.transducer_ordered_indices {
        solver.sensor_recorder =
            SensorRecorder::from_ordered_indices(ordered.clone(), req.time_steps + 1)?;
    } else {
        solver.sensor_recorder =
            SensorRecorder::with_spec(Some(&sensor_mask), shape, req.time_steps + 1, spec)?;
    }

    for source in sources {
        SolverTrait::add_source(&mut solver, source)?;
    }

    Ok((solver, sim_grid))
}

// ── Result extraction ─────────────────────────────────────────────────────────

/// Extract pressure/intensity/velocity fields from a PSTD solver.
pub(crate) fn extract_result(
    solver: &PSTDSolver,
    time_steps: usize,
    record_start_index: usize,
) -> KwaversResult<SimulationRunResult> {
    let stats = solver.sensor_recorder.extract_all_stats();
    let full_data = solver
        .sensor_recorder
        .recorded_pressure_view()
        .ok_or_else(|| KwaversError::Io(std::io::Error::other("No sensor data recorded")))?;
    let sensor_data = trim_initial_recorder_view(full_data, time_steps, record_start_index);

    let ux_data = solver
        .sensor_recorder
        .recorded_ux_view()
        .map(|d| trim_initial_recorder_view(d, time_steps, record_start_index));
    let uy_data = solver
        .sensor_recorder
        .recorded_uy_view()
        .map(|d| trim_initial_recorder_view(d, time_steps, record_start_index));
    let uz_data = solver
        .sensor_recorder
        .recorded_uz_view()
        .map(|d| trim_initial_recorder_view(d, time_steps, record_start_index));
    let ix_data = solver
        .sensor_recorder
        .recorded_ix_view()
        .map(|d| trim_initial_recorder_view(d, time_steps, record_start_index));
    let iy_data = solver
        .sensor_recorder
        .recorded_iy_view()
        .map(|d| trim_initial_recorder_view(d, time_steps, record_start_index));
    let iz_data = solver
        .sensor_recorder
        .recorded_iz_view()
        .map(|d| trim_initial_recorder_view(d, time_steps, record_start_index));
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
