//! Pressure-source schedule assembly shared by GPU PSTD entry points.
//!
//! Both the direct runner and the simulation-trait adapter upload this one
//! schedule. Keeping the conversion here prevents their finite-aperture mask,
//! local-medium, and source-mode contracts from drifting.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_source::{GridSource, SourceMode};

/// Pressure-source payload ready for one GPU PSTD batch.
///
/// `indices` are row-major grid positions. `signals` is source-major with
/// exactly `time_steps` values for each index, already scaled for the GPU
/// mass-source update. `uses_kspace_correction` selects the matching shader
/// source term.
#[derive(Debug)]
pub struct PstdPressureSourceSchedule {
    /// Row-major pressure-source positions in the simulation grid.
    pub indices: Vec<u32>,
    /// Source-major, GPU-representable pressure drive samples.
    pub signals: Vec<f32>,
    /// Whether the additive source uses the k-space correction factor.
    pub uses_kspace_correction: bool,
}

/// Assemble the GPU pressure-source schedule from a grid source.
///
/// Each nonzero pressure-mask value is retained as a finite-aperture BLI
/// weight. For source cell `s`, the uploaded drive is
/// `p_signal[s, t] * p_mask[s] * 2 dt / (N_dim c0[s] dx)`, matching the
/// Treeby--Cox mass-source scaling used by the GPU pressure update. The
/// local sound speed is read from the same single-precision field uploaded to
/// the provider, so host schedule assembly cannot use a different medium than
/// the GPU solver.
///
/// # Errors
///
/// Returns [`KwaversError::InvalidInput`] when the source mask or signal does
/// not match the grid, a source value is non-finite or not representable by
/// the GPU payload, a local sound speed is invalid, or the source mode is
/// unsupported by the GPU update.
pub fn prepare_pstd_pressure_source(
    grid: &Grid,
    source: &GridSource,
    sound_speed: &[f32],
    dt: f64,
    time_steps: usize,
) -> KwaversResult<PstdPressureSourceSchedule> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(KwaversError::InvalidInput(format!(
            "GPU PSTD pressure-source schedule requires finite positive dt; got {dt}"
        )));
    }

    let total = grid
        .nx
        .checked_mul(grid.ny)
        .and_then(|xy| xy.checked_mul(grid.nz))
        .ok_or_else(|| {
            KwaversError::InvalidInput(format!(
                "GPU PSTD grid shape overflows usize: {}×{}×{}",
                grid.nx, grid.ny, grid.nz
            ))
        })?;
    if sound_speed.len() != total {
        return Err(KwaversError::InvalidInput(format!(
            "GPU PSTD sound-speed field has {} cells but the grid has {total}",
            sound_speed.len()
        )));
    }

    let (p_mask, p_signal) = match (&source.p_mask, &source.p_signal) {
        (None, None) => {
            return Ok(PstdPressureSourceSchedule {
                indices: Vec::new(),
                signals: Vec::new(),
                uses_kspace_correction: false,
            });
        }
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
        (Some(mask), Some(signal)) => (mask, signal),
    };

    let mask = p_mask.as_slice_memory_order().ok_or_else(|| {
        KwaversError::InvalidInput("p_mask must be dense row-major array".to_owned())
    })?;
    if mask.len() != total {
        return Err(KwaversError::InvalidInput(format!(
            "p_mask has {} cells but grid has {total}",
            mask.len()
        )));
    }

    let mut weighted_indices = Vec::new();
    for (index, &weight) in mask.iter().enumerate() {
        if !weight.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "p_mask has non-finite BLI weight {weight} at flat index {index}"
            )));
        }
        if weight != 0.0 {
            let index = u32::try_from(index).map_err(|_| {
                KwaversError::InvalidInput(format!(
                    "GPU PSTD pressure-source index {index} exceeds u32 storage"
                ))
            })?;
            weighted_indices.push((index, weight));
        }
    }

    let source_count = weighted_indices.len();
    let signal_rows = p_signal.shape()[0];
    if source_count > 0 && signal_rows != 1 && signal_rows != source_count {
        return Err(KwaversError::InvalidInput(format!(
            "p_signal has {signal_rows} rows for {source_count} pressure-source cells; expected 1 or {source_count}"
        )));
    }

    let signal_len = source_count.checked_mul(time_steps).ok_or_else(|| {
        KwaversError::InvalidInput(format!(
            "GPU PSTD pressure-source shape overflows usize: {source_count} sources × {time_steps} steps"
        ))
    })?;
    let mut signals = Vec::new();
    signals.try_reserve_exact(signal_len).map_err(|_| {
        KwaversError::InvalidInput(format!(
            "cannot allocate GPU PSTD pressure-source schedule: {source_count} sources × {time_steps} steps"
        ))
    })?;
    signals.resize(signal_len, 0.0);

    let active_dimensions = [grid.nx > 1, grid.ny > 1, grid.nz > 1]
        .iter()
        .filter(|&&active| active)
        .count()
        .max(1);
    let signal_steps = p_signal.shape()[1].min(time_steps);
    for (source_index, &(flat_index, weight)) in weighted_indices.iter().enumerate() {
        let flat_index = usize::try_from(flat_index).map_err(|_| {
            KwaversError::InvalidInput(
                "GPU PSTD source index cannot address host memory".to_owned(),
            )
        })?;
        let local_sound_speed = sound_speed[flat_index];
        if !(local_sound_speed.is_finite() && local_sound_speed > 0.0) {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD sound speed must be finite and positive at flat index {flat_index}; got {local_sound_speed}"
            )));
        }
        let source_scale =
            2.0 * dt / (active_dimensions as f64 * f64::from(local_sound_speed) * grid.dx);
        let signal_row = if signal_rows == 1 { 0 } else { source_index };
        for step in 0..signal_steps {
            let scaled = p_signal[[signal_row, step]] * weight * source_scale;
            if !(scaled.is_finite() && scaled.abs() <= f64::from(f32::MAX)) {
                return Err(KwaversError::InvalidInput(format!(
                    "GPU PSTD pressure source {source_index} step {step} is not representable as finite f32"
                )));
            }
            signals[source_index * time_steps + step] = scaled as f32;
        }
    }

    Ok(PstdPressureSourceSchedule {
        indices: weighted_indices
            .into_iter()
            .map(|(index, _)| index)
            .collect(),
        signals,
        uses_kspace_correction: source_mode_uses_kspace_correction(source.p_mode, "pressure")?,
    })
}

pub(super) fn source_mode_uses_kspace_correction(
    mode: SourceMode,
    source_kind: &str,
) -> KwaversResult<bool> {
    match mode {
        SourceMode::Additive => Ok(true),
        SourceMode::AdditiveNoCorrection => Ok(false),
        SourceMode::Dirichlet => Err(KwaversError::InvalidInput(format!(
            "GPU PSTD does not support Dirichlet {source_kind} sources"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{prepare_pstd_pressure_source, source_mode_uses_kspace_correction};
    use kwavers_grid::Grid;
    use kwavers_source::{GridSource, SourceMode};
    use leto::{Array2, Array3};

    #[test]
    fn schedule_preserves_weighted_local_medium_scaling() {
        let grid =
            Grid::new(2, 1, 1, 1.0, 1.0, 1.0).expect("two-cell one-dimensional grid is valid");
        let source = GridSource {
            p_mask: Some(
                Array3::from_shape_vec([2, 1, 1], vec![2.0, 0.5])
                    .expect("mask storage matches the grid"),
            ),
            p_signal: Some(
                Array2::from_shape_vec([2, 2], vec![3.0, 5.0, 7.0, 11.0])
                    .expect("signal rows match source cells"),
            ),
            p_mode: SourceMode::AdditiveNoCorrection,
            ..GridSource::new_empty()
        };

        let schedule = prepare_pstd_pressure_source(&grid, &source, &[1_000.0, 2_000.0], 1.0, 2)
            .expect("finite weighted source schedule is valid");

        assert_eq!(schedule.indices, vec![0, 1]);
        assert_eq!(schedule.signals, vec![0.012, 0.02, 0.0035, 0.0055]);
        assert!(!schedule.uses_kspace_correction);
    }

    #[test]
    fn schedule_rejects_signal_without_mask() {
        let grid =
            Grid::new(2, 1, 1, 1.0, 1.0, 1.0).expect("two-cell one-dimensional grid is valid");
        let source = GridSource {
            p_signal: Some(
                Array2::from_shape_vec([1, 1], vec![1.0])
                    .expect("single signal sample has a matching shape"),
            ),
            ..GridSource::new_empty()
        };

        let error = prepare_pstd_pressure_source(&grid, &source, &[1_000.0, 2_000.0], 1.0, 1)
            .expect_err("a pressure waveform without a spatial mask is invalid");

        assert_eq!(
            error.to_string(),
            "Invalid input: GPU PSTD pressure signal requires a pressure mask"
        );
    }

    #[test]
    fn schedule_rejects_mask_without_signal() {
        let grid =
            Grid::new(2, 1, 1, 1.0, 1.0, 1.0).expect("two-cell one-dimensional grid is valid");
        let source = GridSource {
            p_mask: Some(
                Array3::from_shape_vec([2, 1, 1], vec![1.0, 0.0])
                    .expect("mask storage matches the grid"),
            ),
            ..GridSource::new_empty()
        };

        let error = prepare_pstd_pressure_source(&grid, &source, &[1_000.0, 2_000.0], 1.0, 1)
            .expect_err("a pressure mask without a temporal waveform is invalid");

        assert_eq!(
            error.to_string(),
            "Invalid input: GPU PSTD pressure mask requires a pressure signal"
        );
    }

    #[test]
    fn schedule_rejects_nonpositive_or_nonfinite_timestep() {
        let grid = Grid::new(1, 1, 1, 1.0, 1.0, 1.0).expect("single-cell grid is valid");
        let source = GridSource {
            p_mask: Some(
                Array3::from_shape_vec([1, 1, 1], vec![1.0])
                    .expect("mask storage matches the grid"),
            ),
            p_signal: Some(
                Array2::from_shape_vec([1, 1], vec![1.0])
                    .expect("single signal sample has a matching shape"),
            ),
            ..GridSource::new_empty()
        };

        for dt in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let error = prepare_pstd_pressure_source(&grid, &source, &[1_000.0], dt, 1)
                .expect_err("invalid timestep must not produce a GPU source schedule");
            assert_eq!(
                error.to_string(),
                format!(
                    "Invalid input: GPU PSTD pressure-source schedule requires finite positive dt; got {dt}"
                )
            );
        }
    }

    #[test]
    fn source_mode_selects_the_physical_correction_and_rejects_dirichlet() {
        assert!(
            source_mode_uses_kspace_correction(SourceMode::Additive, "pressure")
                .expect("additive pressure source is supported")
        );
        assert!(
            !source_mode_uses_kspace_correction(SourceMode::AdditiveNoCorrection, "velocity")
                .expect("uncorrected velocity source is supported")
        );

        let error = source_mode_uses_kspace_correction(SourceMode::Dirichlet, "pressure")
            .expect_err("Dirichlet pressure source must be rejected");

        assert_eq!(
            error.to_string(),
            "Invalid input: GPU PSTD does not support Dirichlet pressure sources"
        );
    }
}
