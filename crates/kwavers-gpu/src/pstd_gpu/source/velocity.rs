//! X-velocity-source schedule assembly shared by GPU PSTD entry points.
//!
//! The schedule scales each signal row by the one shared `2·c₀·Δt/Δx`
//! quotient — the same quotient the CPU solver applies, so the providers
//! agree without normalisation.

use super::source_mode_uses_kspace_correction;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_source::{velocity_source_scale, GridSource};
use std::borrow::Cow;

/// Velocity-source payload ready for one GPU PSTD batch.
///
/// `indices` are row-major grid positions. `signals` is source-major with
/// exactly `time_steps` values for each index, already scaled for the GPU
/// velocity update. `uses_kspace_correction` selects the matching shader
/// source term.
#[derive(Debug)]
pub struct PstdVelocitySourceSchedule {
    /// Row-major x-velocity-source positions in the simulation grid.
    pub indices: Vec<u32>,
    /// Source-major, GPU-representable velocity drive samples.
    pub signals: Vec<f32>,
    /// Whether the additive source uses the k-space correction factor.
    pub uses_kspace_correction: bool,
}

/// Assemble the GPU x-velocity-source schedule from a grid source.
///
/// Each nonzero velocity-mask cell contributes its signal row, scaled by the
/// shared `2·c₀·Δt/Δx` helper — the same quotient the CPU solver applies, so
/// the providers agree without normalisation. The local sound speed is read
/// from the same single-precision field uploaded to the provider, so host
/// schedule assembly cannot use a different medium than the GPU solver.
///
/// # Errors
///
/// Returns `KwaversError::InvalidInput` when the source mask or signal does
/// not match the grid, a source value is non-finite or not representable by
/// the GPU payload, a local sound speed is invalid, or the source mode is
/// unsupported by the GPU update.
pub fn prepare_pstd_velocity_source(
    grid: &Grid,
    source: &GridSource,
    sound_speed: &[f32],
    dt: f64,
    time_steps: usize,
) -> KwaversResult<PstdVelocitySourceSchedule> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(KwaversError::InvalidInput(format!(
            "GPU PSTD velocity-source schedule requires finite positive dt; got {dt}"
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

    let (u_mask, u_signal) = match (&source.u_mask, &source.u_signal) {
        (None, None) => {
            return Ok(PstdVelocitySourceSchedule {
                indices: Vec::new(),
                signals: Vec::new(),
                uses_kspace_correction: false,
            });
        }
        (Some(_), None) => {
            return Err(KwaversError::InvalidInput(
                "GPU PSTD velocity mask requires a velocity signal".to_owned(),
            ));
        }
        (None, Some(_)) => {
            return Err(KwaversError::InvalidInput(
                "GPU PSTD velocity signal requires a velocity mask".to_owned(),
            ));
        }
        (Some(mask), Some(signal)) => (mask, signal),
    };

    // Row-major order: borrowed when `u_mask` is C-dense, otherwise
    // collected from its logical iterator.
    let mask_flat: Cow<'_, [_]> = u_mask.as_slice().map_or_else(
        || Cow::Owned(u_mask.iter().copied().collect()),
        Cow::Borrowed,
    );
    if mask_flat.len() != total {
        return Err(KwaversError::InvalidInput(format!(
            "u_mask has {} cells but grid has {total}",
            mask_flat.len()
        )));
    }
    let mut indices = Vec::new();
    for (index, &value) in mask_flat.iter().enumerate() {
        if value != 0.0 {
            let index = u32::try_from(index).map_err(|_| {
                KwaversError::InvalidInput(format!(
                    "GPU PSTD velocity-source index {index} exceeds u32 storage"
                ))
            })?;
            indices.push(index);
        }
    }

    let source_count = indices.len();
    let signal_rows = u_signal.shape()[1];
    if source_count > 0 && signal_rows != 1 && signal_rows != source_count {
        return Err(KwaversError::InvalidInput(format!(
            "u_signal has {signal_rows} source rows for {source_count} velocity-source cells; expected 1 or {source_count}"
        )));
    }

    let signal_len = source_count.checked_mul(time_steps).ok_or_else(|| {
        KwaversError::InvalidInput(format!(
            "GPU PSTD velocity-source shape overflows usize: {source_count} sources × {time_steps} steps"
        ))
    })?;
    let mut signals = Vec::new();
    signals.try_reserve_exact(signal_len).map_err(|_| {
        KwaversError::InvalidInput(format!(
            "cannot allocate GPU PSTD velocity-source schedule: {source_count} sources × {time_steps} steps"
        ))
    })?;
    signals.resize(signal_len, 0.0);

    let signal_steps = u_signal.shape()[2].min(time_steps);
    for (source_index, &flat) in indices.iter().enumerate() {
        let flat = usize::try_from(flat).map_err(|_| {
            KwaversError::InvalidInput(
                "GPU PSTD source index cannot address host memory".to_owned(),
            )
        })?;
        let local_sound_speed = sound_speed[flat];
        if !(local_sound_speed.is_finite() && local_sound_speed > 0.0) {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD sound speed must be finite and positive at flat index {flat}; got {local_sound_speed}"
            )));
        }
        // The one shared quotient: 2·c₀·Δt/Δx, or 1.0 for Dirichlet (which
        // the correction gate below already rejects on the GPU path).
        let source_scale =
            velocity_source_scale(f64::from(local_sound_speed), dt, grid.dx, source.u_mode);
        let signal_row = if signal_rows == 1 { 0 } else { source_index };
        for step in 0..signal_steps {
            let scaled = u_signal[[0, signal_row, step]] * source_scale;
            if !(scaled.is_finite() && scaled.abs() <= f64::from(f32::MAX)) {
                return Err(KwaversError::InvalidInput(format!(
                    "GPU PSTD velocity source {source_index} step {step} is not representable as finite f32"
                )));
            }
            signals[source_index * time_steps + step] = scaled as f32;
        }
    }

    Ok(PstdVelocitySourceSchedule {
        indices,
        signals,
        uses_kspace_correction: source_mode_uses_kspace_correction(source.u_mode, "velocity")?,
    })
}

#[cfg(test)]
mod tests {
    use super::prepare_pstd_velocity_source;
    use kwavers_grid::Grid;
    use kwavers_source::{GridSource, SourceMode};
    use leto::Array3;

    fn velocity_source(mask: Vec<f64>, signal_rows: Vec<f64>, mode: SourceMode) -> GridSource {
        let rows = signal_rows.len() / 2;
        GridSource {
            u_mask: Some(
                Array3::from_shape_vec([4, 1, 1], mask).expect("mask storage matches the grid"),
            ),
            u_signal: Some(
                Array3::from_shape_vec([1, rows, 2], signal_rows)
                    .expect("signal rows match source cells"),
            ),
            u_mode: mode,
            ..GridSource::new_empty()
        }
    }

    #[test]
    fn velocity_schedule_applies_the_shared_two_c_dt_dx_scale() {
        let grid = Grid::new(4, 1, 1, 2.0e-3, 1.0e-3, 1.0e-3).expect("four-cell grid is valid");
        let source = velocity_source(
            vec![0.0, 1.0, 0.0, 1.0],
            vec![3.0, 5.0, 7.0, 11.0],
            SourceMode::AdditiveNoCorrection,
        );
        let schedule = prepare_pstd_velocity_source(&grid, &source, &[1500.0f32; 4], 1e-7, 2)
            .expect("valid schedule");
        assert_eq!(schedule.indices, vec![1, 3]);
        assert!(!schedule.uses_kspace_correction);
        // 2·1500·1e-7/2e-3 = 0.15 exactly.
        let expected = [3.0 * 0.15, 5.0 * 0.15, 7.0 * 0.15, 11.0 * 0.15];
        for (index, &value) in schedule.signals.iter().enumerate() {
            assert_eq!(value, expected[index] as f32, "sample {index}");
        }
    }

    #[test]
    fn velocity_schedule_rejects_dirichlet_like_pressure() {
        let grid = Grid::new(4, 1, 1, 1e-3, 1e-3, 1e-3).expect("four-cell grid is valid");
        let source = velocity_source(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 2.0],
            SourceMode::Dirichlet,
        );
        let error = prepare_pstd_velocity_source(&grid, &source, &[1500.0f32; 4], 1e-7, 2)
            .expect_err("Dirichlet velocity is unsupported");
        assert!(error.to_string().contains("Dirichlet"));
    }

    #[test]
    fn empty_velocity_source_is_a_lossless_schedule() {
        let grid = Grid::new(4, 1, 1, 1e-3, 1e-3, 1e-3).expect("four-cell grid is valid");
        let schedule =
            prepare_pstd_velocity_source(&grid, &GridSource::new_empty(), &[1500.0f32; 4], 1e-7, 2)
                .expect("empty schedule");
        assert!(schedule.indices.is_empty());
        assert!(schedule.signals.is_empty());
        assert!(!schedule.uses_kspace_correction);
    }
}
