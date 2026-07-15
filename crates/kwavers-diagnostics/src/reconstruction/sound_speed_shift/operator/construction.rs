//! Construction of the speed-shift operator from acquisition geometry.

use leto::Array2;

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::inverse::same_aperture::active_grid;

use super::super::propagation::row_entries;
use super::super::types::{SoundSpeedShiftConfig, SoundSpeedShiftSample};
use super::graph::{active_lookup, neighbor_indices};
use super::row_storage::RayRowStorage;
use super::validation::validate_inputs;
use super::SoundSpeedShiftOperator;

impl SoundSpeedShiftOperator {
    pub(in crate::reconstruction::sound_speed_shift) fn new(
        samples: &[SoundSpeedShiftSample],
        active_mask: &Array2<bool>,
        config: SoundSpeedShiftConfig,
    ) -> KwaversResult<Self> {
        validate_inputs(samples, active_mask, config)?;
        let active = active_grid(active_mask, config.spacing_m);
        let shape = {
            let [rows, cols] = active_mask.shape();
            (rows, cols)
        };
        let active_lookup = active_lookup(&active, shape);
        let mut rows = RayRowStorage::new();

        for (sample_index, sample) in samples.iter().enumerate() {
            if !config.sampling.accepts(sample_index) {
                continue;
            }
            let entries = row_entries(sample, &active_lookup, shape, config);
            if !rows.push_nonempty_row(sample_index, entries) {
                return Err(KwaversError::InvalidInput(format!(
                    "Selected speed-shift sample {sample_index} does not intersect the active mask"
                )));
            }
        }

        if rows.row_count() == 0 {
            return Err(KwaversError::InvalidInput(
                "Speed-shift sampling selected zero measurement rows".to_owned(),
            ));
        }

        Ok(Self {
            neighbor_indices: neighbor_indices(&active, shape, &active_lookup),
            active,
            rows,
            shape,
        })
    }
}
