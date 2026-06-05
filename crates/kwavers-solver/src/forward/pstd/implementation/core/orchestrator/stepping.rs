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
        boundary.apply_acoustic(self.fields.p.view_mut(), &self.grid, time_index)?;
        Ok(())
    }
}
