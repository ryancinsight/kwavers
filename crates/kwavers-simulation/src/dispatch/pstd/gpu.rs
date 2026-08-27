//! Hephaestus-backed PSTD request orchestration.

use crate::dispatch::shared::trim_initial_recorder_sample;
use crate::types::{SimulationRunRequest, SimulationRunResult};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_gpu::pstd_gpu::{run_gpu_pstd, GpuPstdRunConfig};
use kwavers_solver::forward::pstd::implementation::core::source_injection;
use kwavers_source::{GridSource, Source as KwaversSource, SourceField, SourceInjectionMode};
use leto::Array3;

/// Run Cartesian PSTD through the selected Hephaestus GPU provider.
///
/// Dynamic pressure sources are sampled once into the source-major batch
/// representation consumed by the GPU solver. Contracts not yet represented
/// by that provider are rejected before device acquisition.
///
/// # Errors
///
/// Returns an explicit error for unsupported coupling, geometry, source, PML,
/// or recording contracts and propagates provider acquisition and execution
/// failures.
#[cfg(feature = "gpu")]
pub fn run_gpu(
    req: &SimulationRunRequest<'_>,
    sources: Vec<Box<dyn KwaversSource>>,
) -> KwaversResult<SimulationRunResult> {
    validate_gpu_request(req)?;
    let source = sample_gpu_pressure_sources(req, sources)?;
    let sensor_mask = req
        .sensor_mask
        .clone()
        .unwrap_or_else(|| Array3::from_elem((req.grid.nx, req.grid.ny, req.grid.nz), false));
    let pml = req.pml.cloned().unwrap_or_default();
    let nonlinear = req.nonlinear.cloned().unwrap_or_default();

    let sensor_data = run_gpu_pstd(
        req.grid,
        req.medium,
        &source,
        &sensor_mask,
        GpuPstdRunConfig {
            time_steps: req.time_steps,
            dt: req.dt,
            nonlinear: nonlinear.enabled,
            alpha_coeff_db: nonlinear.alpha_coeff,
            alpha_power: nonlinear.alpha_power,
            pml_size: pml.size,
            pml_size_xyz: pml.size_xyz,
            pml_inside: pml.inside,
            pml_alpha_xyz: pml.alpha_xyz,
        },
    )?;
    let sensor_data =
        trim_initial_recorder_sample(sensor_data, req.time_steps, req.record_start_index);

    Ok(SimulationRunResult {
        sensor_data,
        stats: None,
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

#[cfg(feature = "gpu")]
fn validate_gpu_request(req: &SimulationRunRequest<'_>) -> KwaversResult<()> {
    if let Some(sensor_mask) = &req.sensor_mask {
        if sensor_mask.shape() != [req.grid.nx, req.grid.ny, req.grid.nz] {
            return Err(KwaversError::DimensionMismatch(format!(
                "sensor_mask shape {:?}; expected [{}, {}, {}]",
                sensor_mask.shape(),
                req.grid.nx,
                req.grid.ny,
                req.grid.nz
            )));
        }
    }
    if req.thermal.is_some() {
        return Err(KwaversError::FeatureNotAvailable(
            "Hephaestus PSTD does not support coupled thermal propagation".to_owned(),
        ));
    }
    if req.axisymmetric {
        return Err(KwaversError::FeatureNotAvailable(
            "Hephaestus PSTD currently implements Cartesian geometry only".to_owned(),
        ));
    }
    if req.transducer_ordered_indices.is_some() {
        return Err(KwaversError::FeatureNotAvailable(
            "Hephaestus PSTD does not yet preserve explicit transducer sensor ordering".to_owned(),
        ));
    }
    if req.pml.is_some_and(|pml| {
        !pml.inside
            && pml
                .effective_thickness(req.grid.nx, req.grid.ny, req.grid.nz)
                .0
                > 0
    }) {
        return Err(KwaversError::FeatureNotAvailable(
            "Hephaestus PSTD does not yet implement external PML grid padding".to_owned(),
        ));
    }
    if req.grid_source.p0.is_some() || req.grid_source.u0.is_some() {
        return Err(KwaversError::FeatureNotAvailable(
            "Hephaestus PSTD does not yet upload pressure or velocity initial conditions"
                .to_owned(),
        ));
    }
    if req.grid_source.u_mask.is_some() || req.grid_source.u_signal.is_some() {
        return Err(KwaversError::FeatureNotAvailable(
            "Hephaestus SimulationRunner does not yet map grid velocity sources".to_owned(),
        ));
    }
    if req.compatibility_mode != kwavers_solver::forward::pstd::config::CompatibilityMode::Optimal {
        return Err(KwaversError::FeatureNotAvailable(
            "Hephaestus PSTD does not implement reference compatibility mode".to_owned(),
        ));
    }
    if let Some(mode) = req.record_modes.iter().find(|mode| mode.as_str() != "p") {
        return Err(KwaversError::FeatureNotAvailable(format!(
            "Hephaestus PSTD does not yet implement recording mode {mode}"
        )));
    }
    Ok(())
}

#[cfg(feature = "gpu")]
#[derive(Clone, Copy)]
enum PressureContribution {
    Grid { signal_row: usize, weight: f64 },
    Dynamic { source_index: usize, weight: f64 },
}

#[cfg(feature = "gpu")]
fn sample_gpu_pressure_sources(
    req: &SimulationRunRequest<'_>,
    sources: Vec<Box<dyn KwaversSource>>,
) -> KwaversResult<GridSource> {
    if sources.is_empty() {
        return Ok(req.grid_source.clone());
    }

    let shape = [req.grid.nx, req.grid.ny, req.grid.nz];
    let total = req
        .grid
        .nx
        .checked_mul(req.grid.ny)
        .and_then(|xy| xy.checked_mul(req.grid.nz))
        .ok_or_else(|| {
            KwaversError::InvalidInput(format!(
                "GPU PSTD grid shape overflows usize: {}x{}x{}",
                req.grid.nx, req.grid.ny, req.grid.nz
            ))
        })?;
    let mut contributions = Vec::new();
    append_grid_pressure_contributions(&req.grid_source, shape, &mut contributions)?;

    let mut dynamic_waveforms = Vec::with_capacity(sources.len());
    for (source_index, source) in sources.iter().enumerate() {
        if source.source_type() != SourceField::Pressure {
            return Err(KwaversError::FeatureNotAvailable(format!(
                "Hephaestus SimulationRunner cannot sample dynamic {:?} sources",
                source.source_type()
            )));
        }
        let mask = source.create_mask(req.grid);
        if mask.shape() != shape {
            return Err(KwaversError::DimensionMismatch(format!(
                "dynamic source mask shape {:?} does not match GPU PSTD grid {shape:?}",
                mask.shape()
            )));
        }
        let scale = match source_injection::determine_injection_mode(&mask) {
            SourceInjectionMode::Additive { scale } => scale,
            SourceInjectionMode::Boundary => 1.0,
        };
        for (flat, weight) in mask.iter().copied().enumerate() {
            if !weight.is_finite() {
                return Err(KwaversError::InvalidInput(format!(
                    "dynamic source {source_index} mask has non-finite value at flat index {flat}"
                )));
            }
            if weight.abs() > 1e-12 {
                contributions.push((
                    flat,
                    PressureContribution::Dynamic {
                        source_index,
                        weight: weight * scale,
                    },
                ));
            }
        }
        let mut waveform = Vec::new();
        waveform.try_reserve_exact(req.time_steps).map_err(|_| {
            KwaversError::InvalidInput(format!(
                "cannot allocate {0} samples for dynamic source {source_index}",
                req.time_steps
            ))
        })?;
        for step in 0..req.time_steps {
            let amplitude = source.amplitude(step as f64 * req.dt);
            if !amplitude.is_finite() {
                return Err(KwaversError::InvalidInput(format!(
                    "dynamic source {source_index} has non-finite amplitude at step {step}"
                )));
            }
            waveform.push(amplitude);
        }
        dynamic_waveforms.push(waveform);
    }

    contributions.sort_unstable_by_key(|(flat, _)| *flat);
    let source_count = contributions
        .iter()
        .map(|(flat, _)| *flat)
        .fold((None, 0usize), |(previous, count), flat| {
            (Some(flat), count + usize::from(previous != Some(flat)))
        })
        .1;
    let signal_len = source_count.checked_mul(req.time_steps).ok_or_else(|| {
        KwaversError::InvalidInput(format!(
            "GPU PSTD sampled source shape overflows usize: {source_count} sources x {} steps",
            req.time_steps
        ))
    })?;
    let mut signals = Vec::new();
    signals.try_reserve_exact(signal_len).map_err(|_| {
        KwaversError::InvalidInput(format!(
            "cannot allocate GPU PSTD sampled source storage for {source_count} sources x {} steps",
            req.time_steps
        ))
    })?;
    signals.resize(signal_len, 0.0);
    let mut mask_values = vec![0.0; total];
    let grid_signal = req.grid_source.p_signal.as_ref();
    let mut output_row = usize::MAX;
    let mut previous_flat = None;
    for (flat, contribution) in contributions {
        if previous_flat != Some(flat) {
            output_row = output_row.wrapping_add(1);
            previous_flat = Some(flat);
            mask_values[flat] = 1.0;
        }
        let output = &mut signals[output_row * req.time_steps..(output_row + 1) * req.time_steps];
        match contribution {
            PressureContribution::Grid { signal_row, weight } => {
                let signal = grid_signal.expect("invariant: grid contribution requires signal");
                for (step, value) in output.iter_mut().enumerate().take(signal.shape()[1]) {
                    *value += signal[[signal_row, step]] * weight;
                }
            }
            PressureContribution::Dynamic {
                source_index,
                weight,
            } => {
                for (value, &amplitude) in output.iter_mut().zip(&dynamic_waveforms[source_index]) {
                    *value += amplitude * weight;
                }
            }
        }
    }

    let p_mask = Array3::from_shape_vec(shape, mask_values).map_err(|error| {
        KwaversError::InvalidInput(format!("GPU PSTD sampled mask shape mismatch: {error}"))
    })?;
    let p_signal =
        leto::Array2::from_shape_vec([source_count, req.time_steps], signals).map_err(|error| {
            KwaversError::InvalidInput(format!("GPU PSTD sampled signal shape mismatch: {error}"))
        })?;
    Ok(GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        ..req.grid_source.clone()
    })
}

#[cfg(feature = "gpu")]
fn append_grid_pressure_contributions(
    source: &GridSource,
    shape: [usize; 3],
    contributions: &mut Vec<(usize, PressureContribution)>,
) -> KwaversResult<()> {
    let (mask, signal) = match (&source.p_mask, &source.p_signal) {
        (None, None) => return Ok(()),
        (Some(mask), Some(signal)) => (mask, signal),
        (Some(_), None) => {
            return Err(KwaversError::InvalidInput(
                "GPU PSTD pressure mask requires a pressure signal".to_owned(),
            ));
        }
        (None, Some(_)) => {
            return Err(KwaversError::InvalidInput(
                "GPU PSTD pressure signal requires a pressure mask".to_owned(),
            ));
        }
    };
    if mask.shape() != shape {
        return Err(KwaversError::DimensionMismatch(format!(
            "pressure source mask shape {:?} does not match GPU PSTD grid {shape:?}",
            mask.shape()
        )));
    }
    let active_count = mask.iter().filter(|&&weight| weight != 0.0).count();
    if active_count > 0 && signal.shape()[0] != 1 && signal.shape()[0] != active_count {
        return Err(KwaversError::DimensionMismatch(format!(
            "pressure signal has {} rows for {active_count} active mask cells; expected 1 or {active_count}",
            signal.shape()[0]
        )));
    }
    let mut active_row = 0usize;
    for (flat, weight) in mask.iter().copied().enumerate() {
        if !weight.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "pressure source mask has non-finite value at flat index {flat}"
            )));
        }
        if weight != 0.0 {
            contributions.push((
                flat,
                PressureContribution::Grid {
                    signal_row: if signal.shape()[0] == 1 {
                        0
                    } else {
                        active_row
                    },
                    weight,
                },
            ));
            active_row += 1;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_source_tests {
    use super::sample_gpu_pressure_sources;
    use crate::types::SimulationRunRequest;
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;
    use kwavers_solver::config::{FftBackend, SolverType};
    use kwavers_solver::forward::fdtd::config::KSpaceCorrectionMode;
    use kwavers_solver::forward::pstd::config::CompatibilityMode;
    use kwavers_source::{GridSource, TimeVaryingSource};
    use leto::{Array2, Array3};

    #[test]
    fn sampled_pressure_sources_preserve_weights_rows_and_superposition() {
        let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1_000.0, 1_500.0, &grid);
        let mut mask = Array3::zeros([3, 3, 3]);
        mask[[0, 0, 0]] = 2.0;
        let request = SimulationRunRequest {
            grid: &grid,
            medium: &medium,
            time_steps: 3,
            dt: 1.0,
            solver_type: SolverType::PSTD,
            fft_backend: FftBackend::Hephaestus,
            pml: None,
            helmholtz: None,
            nonlinear: None,
            thermal: None,
            poroelastic: None,
            compatibility_mode: CompatibilityMode::Optimal,
            kspace_correction: KSpaceCorrectionMode::None,
            axisymmetric: false,
            grid_source: GridSource {
                p_mask: Some(mask),
                p_signal: Some(
                    Array2::from_shape_vec([1, 3], vec![3.0, 5.0, 7.0])
                        .expect("signal shape matches samples"),
                ),
                ..GridSource::new_empty()
            },
            sensor_mask: None,
            transducer_ordered_indices: None,
            record_modes: Vec::new(),
            record_start_index: 1,
            transducers_for_rs: &[],
            elastic_velocity_source: None,
            elastic_ivp_axis: None,
        };
        let sources = vec![
            Box::new(TimeVaryingSource::new(
                (1, 1, 1),
                vec![11.0, 13.0, 17.0],
                1.0,
            )) as Box<dyn kwavers_source::Source>,
            Box::new(TimeVaryingSource::new(
                (1, 1, 1),
                vec![19.0, 23.0, 29.0],
                1.0,
            )) as Box<dyn kwavers_source::Source>,
        ];

        let sampled = sample_gpu_pressure_sources(&request, sources)
            .expect("pressure sources are representable as a GPU batch");
        let sampled_mask = sampled.p_mask.expect("sampled source has a mask");
        let sampled_signal = sampled.p_signal.expect("sampled source has a signal");

        assert_eq!(sampled_mask[[0, 0, 0]], 1.0);
        assert_eq!(sampled_mask[[1, 1, 1]], 1.0);
        assert_eq!(
            sampled_mask.iter().filter(|&&value| value != 0.0).count(),
            2
        );
        assert_eq!(sampled_signal.shape(), [2, 3]);
        assert_eq!(sampled_signal[[0, 0]], 6.0);
        assert_eq!(sampled_signal[[0, 1]], 10.0);
        assert_eq!(sampled_signal[[0, 2]], 14.0);
        assert_eq!(sampled_signal[[1, 0]], 30.0);
        assert_eq!(sampled_signal[[1, 1]], 36.0);
        assert_eq!(sampled_signal[[1, 2]], 46.0);
    }
}
