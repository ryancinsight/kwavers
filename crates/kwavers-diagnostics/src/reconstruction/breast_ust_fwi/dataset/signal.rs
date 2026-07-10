//! Signal generation, frequency-bin extraction, and grid-mapping helpers
//! for the breast UST PSTD acquisition pipeline.

use super::BreastUstPstdDatasetConfig;
use kwavers_boundary::CPMLConfig;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::forward::pstd::config::BoundaryConfig;
use kwavers_transducer::transducers::ElementPosition;
use leto::{
    Array2,
    ArrayView1,
    SliceArg,
};
use kwavers_math::fft::Complex64;

pub(super) fn pstd_boundary(cpml_thickness_cells: usize) -> BoundaryConfig {
    if cpml_thickness_cells == 0 {
        BoundaryConfig::None
    } else {
        BoundaryConfig::CPML(CPMLConfig::with_thickness(cpml_thickness_cells))
    }
}

pub(super) fn tone_signal(
    frequency_hz: f64,
    steps: usize,
    config: BreastUstPstdDatasetConfig,
) -> Array2<f64> {
    Array2::from_shape_fn((1, steps), |[_, n]| {
        let phase = TWO_PI * frequency_hz * n as f64 * config.time_step_s;
        config.source_amplitude_pa * phase.sin()
    })
}

pub(super) fn frequency_bin(
    samples: ArrayView1<'_, f64>,
    frequency_hz: f64,
    dt: f64,
    start_sample: usize,
) -> Complex64 {
    let _window = samples
        .slice_with::<1>(&[SliceArg::Range {
            start: Some(start_sample as isize),
            end: None,
            step: 1,
        }])
        .expect("start_sample within trace");
    let scale = 2.0 / _window.size() as f64;
    samples.iter().skip(start_sample).enumerate().fold(
        Complex64::new(0.0, 0.0),
        |acc, (n, &sample)| {
            let phase = -TWO_PI * frequency_hz * (start_sample + n) as f64 * dt;
            acc + Complex64::new(phase.cos(), phase.sin()) * sample
        },
    ) * scale
}

pub(super) fn map_ring_points_to_grid(
    dims: (usize, usize, usize),
    config: BreastUstPstdDatasetConfig,
    points: &[ElementPosition],
) -> KwaversResult<Vec<(usize, usize, usize)>> {
    points
        .iter()
        .map(|point| map_ring_point_to_grid(dims, config.spacing_m, *point))
        .collect()
}

pub(super) fn map_ring_point_to_grid(
    (nx, ny, nz): (usize, usize, usize),
    spacing_m: f64,
    point: ElementPosition,
) -> KwaversResult<(usize, usize, usize)> {
    let center = [
        0.5 * (nx - 1) as f64 * spacing_m,
        0.5 * (ny - 1) as f64 * spacing_m,
        0.5 * (nz - 1) as f64 * spacing_m,
    ];
    let coord = [
        center[0] + point.x_m,
        center[1] + point.y_m,
        center[2] + point.z_m,
    ];
    let max = [
        (nx - 1) as f64 * spacing_m,
        (ny - 1) as f64 * spacing_m,
        (nz - 1) as f64 * spacing_m,
    ];
    for axis in 0..3 {
        if coord[axis] < 0.0 || coord[axis] > max[axis] {
            return Err(KwaversError::InvalidInput(format!(
                "ring point {:?} maps outside centered PSTD grid bounds {:?}",
                point, max
            )));
        }
    }
    Ok((
        (coord[0] / spacing_m).round() as usize,
        (coord[1] / spacing_m).round() as usize,
        (coord[2] / spacing_m).round() as usize,
    ))
}

pub(super) fn grid_index_to_ring_point(
    (nx, ny, nz): (usize, usize, usize),
    spacing_m: f64,
    (ix, iy, iz): (usize, usize, usize),
) -> ElementPosition {
    let center = [
        0.5 * (nx - 1) as f64,
        0.5 * (ny - 1) as f64,
        0.5 * (nz - 1) as f64,
    ];
    ElementPosition {
        x_m: (ix as f64 - center[0]) * spacing_m,
        y_m: (iy as f64 - center[1]) * spacing_m,
        z_m: (iz as f64 - center[2]) * spacing_m,
    }
}

pub(super) fn time_steps_for_frequency(
    frequency_hz: f64,
    config: BreastUstPstdDatasetConfig,
) -> KwaversResult<usize> {
    time_steps_for_cycles(
        frequency_hz,
        config.time_step_s,
        config.cycles_per_frequency,
    )
}

pub(super) fn frequency_bin_start_step(
    frequency_hz: f64,
    config: BreastUstPstdDatasetConfig,
    total_steps: usize,
) -> KwaversResult<usize> {
    let bin_steps = time_steps_for_cycles(
        frequency_hz,
        config.time_step_s,
        config.frequency_bin_cycles,
    )?;
    Ok(total_steps.saturating_sub(bin_steps))
}

pub(super) fn time_steps_for_cycles(
    frequency_hz: f64,
    dt: f64,
    cycles: usize,
) -> KwaversResult<usize> {
    let raw = (cycles as f64 / (frequency_hz * dt)).ceil();
    if !raw.is_finite() || raw > usize::MAX as f64 {
        return Err(KwaversError::InvalidInput(format!(
            "time-step count is not representable for frequency {frequency_hz}"
        )));
    }
    Ok((raw as usize).max(2))
}
