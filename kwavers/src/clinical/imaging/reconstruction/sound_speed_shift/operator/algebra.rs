//! Matrix-vector algebra over the speed-shift row storage.

use super::SoundSpeedShiftOperator;

impl SoundSpeedShiftOperator {
    pub(in crate::clinical::imaging::reconstruction::sound_speed_shift) fn matvec(
        &self,
        x: &[f64],
        out: &mut [f64],
    ) {
        debug_assert_eq!(x.len(), self.cols());
        debug_assert_eq!(out.len(), self.rows());
        for (row, dst) in out.iter_mut().enumerate() {
            *dst = self
                .rows
                .row_entries(row)
                .map(|(col, length)| length * x[col])
                .sum();
        }
    }

    pub(in crate::clinical::imaging::reconstruction::sound_speed_shift) fn t_matvec(
        &self,
        y: &[f64],
        out: &mut [f64],
    ) {
        debug_assert_eq!(y.len(), self.rows());
        debug_assert_eq!(out.len(), self.cols());
        out.fill(0.0);
        for (row, value) in y.iter().enumerate() {
            for (col, length) in self.rows.row_entries(row) {
                out[col] += length * *value;
            }
        }
    }

    pub(in crate::clinical::imaging::reconstruction::sound_speed_shift) fn normal_diag_into(
        &self,
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), self.cols());
        out.fill(0.0);
        for row in 0..self.rows() {
            for (col, length) in self.rows.row_entries(row) {
                out[col] += length * length;
            }
        }
    }
}
