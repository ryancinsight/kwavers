//! PSTD workspace contract.
//!
//! The PSTD solver is FFT-heavy and relies on stable scratch ownership to keep
//! per-step allocation at zero in steady-state execution. This wrapper exposes
//! that expectation explicitly as part of the public first-wave scientific API.

use crate::domain::grid::Grid;
use crate::solver::validation::MemoryBudget;
use crate::solver::workspace::{SolverWorkspace, WorkspaceProfile};

/// Reusable scratch storage for the acoustic PSTD solver family.
#[derive(Debug)]
pub struct PstdWorkspace {
    storage: SolverWorkspace,
}

impl PstdWorkspace {
    /// Allocate all steady-state scratch buffers for a PSTD grid.
    #[must_use]
    pub fn new(grid: &Grid) -> Self {
        Self {
            storage: SolverWorkspace::new(grid),
        }
    }

    /// Borrow the shared scratch storage.
    #[must_use]
    pub fn storage(&self) -> &SolverWorkspace {
        &self.storage
    }

    /// Borrow the shared scratch storage mutably.
    pub fn storage_mut(&mut self) -> &mut SolverWorkspace {
        &mut self.storage
    }

    /// Clear reusable buffers without reallocating.
    pub fn clear(&mut self) {
        self.storage.clear();
    }

    /// Scientific memory budget for steady-state PSTD stepping.
    #[must_use]
    pub fn memory_budget(&self) -> MemoryBudget {
        self.storage.memory_budget()
    }

    /// Workspace profile used by validation and benchmarking harnesses.
    #[must_use]
    pub fn profile(&self) -> WorkspaceProfile {
        self.storage.profile()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn pstd_workspace_reports_zero_transient_budget() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0).unwrap();
        let workspace = PstdWorkspace::new(&grid);
        let budget = workspace.memory_budget();
        assert!(budget.workspace_bytes > 0);
        assert_eq!(budget.max_transient_bytes, 0);
        assert_eq!(budget.current_transient_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(budget.peak_transient_bytes.load(Ordering::Relaxed), 0);
    }
}
