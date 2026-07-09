//! WorkspacePool and WorkspaceGuard: RAII-managed pool of SolverWorkspace instances.

use super::solver_workspace::SolverWorkspace;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use parking_lot::Mutex;
use std::sync::Arc;

/// Thread-local workspace pool for parallel operations.
#[derive(Debug)]
pub struct WorkspacePool {
    workspaces: Arc<Mutex<Vec<SolverWorkspace>>>,
    grid: Grid,
}

impl WorkspacePool {
    /// Create a new workspace pool with `initial_capacity` pre-allocated workspaces.
    pub fn new(grid: Grid, initial_capacity: usize) -> Self {
        let mut workspaces = Vec::with_capacity(initial_capacity);
        for _ in 0..initial_capacity {
            workspaces.push(SolverWorkspace::new(&grid));
        }
        Self {
            workspaces: Arc::new(Mutex::new(workspaces)),
            grid,
        }
    }

    /// Borrow a workspace from the pool (creates one if the pool is empty).
    /// # Errors
    /// - Currently infallible (`parking_lot::Mutex` does not poison); the
    ///   `Result` is retained for API stability.
    pub fn acquire(&self) -> KwaversResult<WorkspaceGuard> {
        let mut pool = self.workspaces.lock();

        let workspace = pool
            .pop()
            .unwrap_or_else(|| SolverWorkspace::new(&self.grid));

        Ok(WorkspaceGuard {
            workspace: Some(workspace),
            pool: Arc::clone(&self.workspaces),
        })
    }

    /// Current number of idle workspaces in the pool.
    pub fn size(&self) -> usize {
        let pool = self.workspaces.lock();
        (pool.shape()[0] * pool.shape()[1] * pool.shape()[2])
    }
}

/// RAII guard for workspace borrowing.  Returns the workspace to the pool on drop.
#[derive(Debug)]
pub struct WorkspaceGuard {
    workspace: Option<SolverWorkspace>,
    pool: Arc<Mutex<Vec<SolverWorkspace>>>,
}

impl WorkspaceGuard {
    /// Shared reference to the borrowed workspace.
    /// # Panics
    /// - Panics if the workspace has already been returned.
    #[must_use]
    pub fn get(&self) -> &SolverWorkspace {
        self.workspace.as_ref().expect("Workspace already returned")
    }

    /// Exclusive reference to the borrowed workspace.
    /// # Panics
    /// - Panics if the workspace has already been returned.
    pub fn get_mut(&mut self) -> &mut SolverWorkspace {
        self.workspace.as_mut().expect("Workspace already returned")
    }
}

impl Drop for WorkspaceGuard {
    fn drop(&mut self) {
        if let Some(workspace) = self.workspace.take() {
            self.pool.lock().push(workspace);
        }
    }
}
