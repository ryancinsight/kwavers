use super::PSTDSolver;
use kwavers_core::error::KwaversResult;

impl PSTDSolver {
    /// Apply boundary.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn apply_boundary(&mut self, time_index: usize) -> KwaversResult<()> {
        let Some(boundary) = &mut self.boundary else {
            return Ok(());
        };
        let [nx, ny, nz] = self.fields.p.shape();
        let mut pressure =
            ndarray::Array3::from_shape_vec((nx, ny, nz), self.fields.p.iter().copied().collect())
                .expect("leto pressure field shape must map to ndarray");
        boundary.apply_acoustic(pressure.view_mut().into(), &self.grid, time_index)?;
        for (dst_value, src_value) in self
            .fields
            .p
            .as_slice_mut()
            .expect("leto PSTD pressure field must be contiguous")
            .iter_mut()
            .zip(pressure.iter())
        {
            *dst_value = *src_value;
        }
        Ok(())
    }
}
