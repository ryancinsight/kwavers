use super::PSTDSolver;
use kwavers_core::error::KwaversResult;

impl PSTDSolver {
    /// Apply boundary in-place on the solver's pressure field.
    ///
    /// The pressure field is already `leto::Array3<f64>`, and
    /// `Boundary::apply_acoustic` takes `leto::ArrayViewMut3<f64>`, so the
    /// field is passed directly — no copy-out/copy-back as would be needed if
    /// the solver and boundary used incompatible array types.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub(crate) fn apply_boundary(&mut self, time_index: usize) -> KwaversResult<()> {
        let Some(boundary) = &mut self.boundary else {
            return Ok(());
        };
        boundary.apply_acoustic(self.fields.p.view_mut(), &self.grid, time_index)
    }
}
