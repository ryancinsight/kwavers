//! Propagation-path and sensitivity row assembly.

mod finite_frequency;
mod geometric;
pub(super) mod path;
#[cfg(test)]
mod tests;

use super::types::{ShiftSensitivity, SoundSpeedShiftConfig, SoundSpeedShiftSample};

pub(super) fn row_entries(
    sample: &SoundSpeedShiftSample,
    active_lookup: &[Option<usize>],
    shape: (usize, usize),
    config: SoundSpeedShiftConfig,
) -> Vec<(usize, f64)> {
    let path = path::build_path(sample, config.propagation);
    let mut entries = match config.sensitivity {
        ShiftSensitivity::GeometricRay => {
            geometric::entries(&path, active_lookup, shape, config.spacing_m)
        }
        ShiftSensitivity::FiniteFrequency {
            wavelength_m,
            support_radius_m,
        } => finite_frequency::entries(
            &path,
            active_lookup,
            shape,
            config.spacing_m,
            wavelength_m,
            support_radius_m,
        ),
    };
    merge_by_column(&mut entries);
    entries
}

fn merge_by_column(entries: &mut Vec<(usize, f64)>) {
    entries.sort_by_key(|(column, _)| *column);
    let mut write = 0;
    for read in 0..entries.len() {
        if write > 0 && entries[write - 1].0 == entries[read].0 {
            entries[write - 1].1 += entries[read].1;
        } else {
            let entry = entries[read];
            entries[write] = entry;
            write += 1;
        }
    }
    entries.truncate(write);
}

fn active_column(
    active_lookup: &[Option<usize>],
    ix: usize,
    iy: usize,
    ny: usize,
    weight: f64,
) -> Option<(usize, f64)> {
    active_lookup[ix * ny + iy].map(|column| (column, weight))
}
