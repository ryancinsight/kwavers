//! WorkspacePool and WorkspaceGuard: RAII-managed pool of SolverWorkspace instances.

use super::solver_workspace::SolverWorkspace;
use kwavers_core::error::KwaversResult;
#[cfg(not(feature = "parallel"))]
use kwavers_core::error::{KwaversError, SystemError};
use kwavers_domain::grid::Grid;
#[cfg(feature = "parallel")]
use parking_lot::Mutex;
use std::sync::Arc;
#[cfg(not(feature = "parallel"))]
use std::sync::Mutex;

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
    /// - Returns [`KwaversError::System`] if the pool lock cannot be acquired (non-parallel builds).
    ///
    pub fn acquire(&self) -> KwaversResult<WorkspaceGuard> {
        #[cfg(feature = "parallel")]
        let mut pool = self.workspaces.lock();
        #[cfg(not(feature = "parallel"))]
        let mut pool = match self.workspaces.lock() {
            Ok(p) => p,
            Err(e) => {
                return Err(KwaversError::System(SystemError::ResourceExhausted {
                    resource: "workspace pool".to_owned(),
                    reason: format!("Failed to acquire lock: {e}"),
                }))
            }
        };

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
        #[cfg(feature = "parallel")]
        let pool = self.workspaces.lock();
        #[cfg(not(feature = "parallel"))]
        let pool = self
            .workspaces
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pool.len()
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
            #[cfg(feature = "parallel")]
            self.pool.lock().push(workspace);
            #[cfg(not(feature = "parallel"))]
            if let Ok(mut pool) = self.pool.lock() {
                pool.push(workspace);
            }
        }
    }
}
