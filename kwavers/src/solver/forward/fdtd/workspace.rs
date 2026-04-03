//! FDTD workspace contract.
//!
//! This module exposes the steady-state memory surface required by the
//! scientific acceptance contract for the acoustic FDTD path. The workspace is
//! intentionally thin and delegates storage ownership to the shared
//! [`SolverWorkspace`](crate::solver::workspace::SolverWorkspace) so the hot
//! path has a single source of truth for scratch allocation behavior.

use crate::domain::grid::Grid;
use crate::solver::validation::MemoryBudget;
use crate::solver::workspace::{SolverWorkspace, WorkspaceProfile};

/// Reusable scratch storage for the acoustic FDTD solver family.
#[derive(Debug)]
pub struct FdtdWorkspace {
    storage: SolverWorkspace,
}

impl FdtdWorkspace {
    /// Allocate all steady-state scratch buffers for an FDTD grid.
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

    /// Scientific memory budget for steady-state FDTD stepping.
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
    fn fdtd_workspace_reports_zero_transient_budget() {
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0).unwrap();
        let workspace = FdtdWorkspace::new(&grid);
        let budget = workspace.memory_budget();
        assert!(budget.workspace_bytes > 0);
        assert_eq!(budget.max_transient_bytes, 0);
        assert_eq!(budget.current_transient_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(budget.peak_transient_bytes.load(Ordering::Relaxed), 0);
    }
}
