use super::PSTDSolver;
use crate::core::error::KwaversResult;

impl PSTDSolver {
    pub(crate) fn apply_boundary(&mut self, time_index: usize) -> KwaversResult<()> {
        let Some(boundary) = &mut self.boundary else {
            return Ok(());
        };
        boundary.apply_acoustic(self.fields.p.view_mut(), &self.grid, time_index)?;
        Ok(())
    }
}
