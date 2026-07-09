//! Matrix-free straight-ray operator for speed-of-sound shift imaging.

mod algebra;
mod construction;
mod graph;
mod row_storage;
mod validation;

use leto::Array2;

use kwavers_solver::inverse::same_aperture::ActiveGrid;

use row_storage::RayRowStorage;

use super::types::SoundSpeedShiftSample;

#[derive(Clone, Debug)]
pub(super) struct SoundSpeedShiftOperator {
    active: ActiveGrid,
    neighbor_indices: Vec<[Option<usize>; 4]>,
    rows: RayRowStorage,
    shape: (usize, usize),
}

impl SoundSpeedShiftOperator {
    #[must_use]
    pub(super) fn image_from_model(&self, model: &[f64]) -> Array2<f64> {
        let mut image = Array2::<f64>::zeros(self.shape);
        self.image_from_model_into(model, &mut image);
        image
    }

    pub(super) fn image_from_model_into(&self, model: &[f64], image: &mut Array2<f64>) {
        debug_assert_eq!(image.dim(), self.shape);
        debug_assert_eq!(model.len(), self.active.len());
        image.fill(0.0);
        for ((ix, iy), value) in self.active.indices.iter().zip(model.iter()) {
            image[[*ix, *iy]] = *value;
        }
    }

    #[must_use]
    pub(super) fn rhs_from_sample_time_shifts(
        &self,
        samples: &[SoundSpeedShiftSample],
        reference_sound_speed_m_s: f64,
    ) -> Vec<f64> {
        let c0_sq = reference_sound_speed_m_s * reference_sound_speed_m_s;
        (0..self.rows.row_count())
            .map(|row| -samples[self.rows.sample_index(row)].time_shift_s * c0_sq)
            .collect()
    }

    #[must_use]
    pub(super) fn rhs_from_time_shift_values(
        &self,
        time_shifts_s: &[f64],
        reference_sound_speed_m_s: f64,
    ) -> Vec<f64> {
        let mut out = vec![0.0; self.rows.row_count()];
        self.rhs_from_time_shift_values_into(time_shifts_s, reference_sound_speed_m_s, &mut out);
        out
    }

    pub(super) fn rhs_from_time_shift_values_into(
        &self,
        time_shifts_s: &[f64],
        reference_sound_speed_m_s: f64,
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), self.rows.row_count());
        let c0_sq = reference_sound_speed_m_s * reference_sound_speed_m_s;
        for (row, value) in out.iter_mut().enumerate() {
            *value = -time_shifts_s[self.rows.sample_index(row)] * c0_sq;
        }
    }

    #[must_use]
    pub(super) fn model_from_image(&self, image: &Array2<f64>) -> Vec<f64> {
        self.active
            .indices
            .iter()
            .map(|(ix, iy)| image[[*ix, *iy]])
            .collect()
    }

    #[must_use]
    pub(super) fn rows(&self) -> usize {
        self.rows.row_count()
    }

    #[must_use]
    pub(super) fn cols(&self) -> usize {
        self.active.len()
    }

    #[cfg(test)]
    #[must_use]
    pub(super) fn stored_segment_count(&self) -> usize {
        self.rows.nonzero_count()
    }

    #[must_use]
    pub(super) fn stored_weight_count(&self) -> usize {
        self.rows.nonzero_count()
    }
}
