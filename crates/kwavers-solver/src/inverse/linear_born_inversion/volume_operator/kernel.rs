//! Born sensitivity kernel evaluation for [`super::VolumeOperator`].

use super::{RowContext, VolumeOperator};

impl<'a> VolumeOperator<'a> {
    /// Evaluate Born sensitivity kernel A\[row, col\] using precomputed distances.
    ///
    /// ## Spreading denominator
    ///
    /// `sqrt(ds·dr) = sqrt(ds) · sqrt(dr)`.  Both factors come from the
    /// precomputed `elem_sqrt_dist` table; no `sqrt` is computed in the hot path.
    ///
    /// ## Attenuation model
    ///
    /// When `row_context.attenuation_model` is set, exponential amplitude decay
    /// `exp(−α·f·path)` is applied where `α` is the voxel's specific attenuation
    /// coefficient \[Np/(m·MHz)\] and `f` is the channel frequency \[MHz\].
    ///
    /// ## Harmonic scaling
    ///
    /// A non-zero `harmonic_path_scale` activates the second-harmonic linear
    /// growth term `scale · path_m` (from the shock-distance expansion of Westervelt
    /// nonlinearity); fundamental rows use `harmonic_path_scale = 0` and evaluate to 1.
    #[inline]
    pub(super) fn row_value_for_col(&self, row_context: &RowContext, col: usize) -> f64 {
        let base = row_context.source_idx * self.n_active + col;
        let base_r = row_context.receiver_idx * self.n_active + col;
        let ds = self.elem_dist[base];
        let dr = self.elem_dist[base_r];
        let sqrt_ds = self.elem_sqrt_dist[base];
        let sqrt_dr = self.elem_sqrt_dist[base_r];
        let path_m = ds + dr;
        let spreading = (sqrt_ds * sqrt_dr).max(1.0e-6);
        let voxel = &self.active[col];
        let attenuation = if row_context.attenuation_model {
            (-(voxel.attenuation_np_per_m_mhz * path_m) * row_context.frequency_mhz).exp()
        } else {
            1.0
        };
        let harmonic = if row_context.harmonic_path_scale == 0.0 {
            1.0
        } else {
            row_context.harmonic_path_scale * path_m
        };
        self.voxel_volume_m3 * attenuation * harmonic * (row_context.k * path_m).cos() / spreading
    }
}
