//! Memory-efficient workspace for solver operations
//!
//! This module provides pre-allocated workspace arrays to minimize allocations
//! during simulation, and the [`ScratchArena`] trait that unifies the shared
//! contract across all solver workspace types.
//!
//! # Design principles
//! - **SSOT**: One canonical trait for scratch-buffer memory management.
//! - **DRY**: `ScratchArena` eliminates per-solver boilerplate.
//! - **Performance**: Zero-allocation hot paths via pre-allocated buffers.
//!
//! # Invariant (Memory Monotonicity)
//! For any `T: ScratchArena`, `T::memory_bytes()` is constant after construction.
//! `T::clear()` sets all elements to zero without reallocation, so
//! `memory_bytes()` before and after `clear()` are equal.

use crate::core::error::KwaversResult;
#[cfg(not(feature = "parallel"))]
use crate::core::error::{KwaversError, SystemError};
use crate::domain::grid::Grid;

// ─────────────────────────────────────────────────────────────────────────────
// ScratchArena — shared contract for pre-allocated solver scratch buffers
// ─────────────────────────────────────────────────────────────────────────────

/// Shared contract for pre-allocated solver scratch buffers.
///
/// Every solver workspace that allocates arrays at construction time and reuses
/// them across time-loop iterations must implement `ScratchArena` to expose a
/// uniform interface for memory reporting and buffer invalidation.
///
/// # Invariants
///
/// 1. **Memory stability** — `memory_bytes()` returns the same value throughout
///    the lifetime of the arena (construction does not change the footprint).
/// 2. **Zero after clear** — after `clear()` returns, every element of every
///    scratch buffer is exactly zero (or `Complex::zero()` for complex buffers).
/// 3. **No reallocation** — `clear()` must not allocate; it only fills existing
///    backing storage.
///
/// # Mathematical basis
///
/// Let W be a workspace allocated for an N-element grid. Define:
/// ```text
/// StaticFootprint(W) := Σ_i (|buf_i| × sizeof(elem_i))
/// ```
/// `memory_bytes()` returns `StaticFootprint(W)` in O(1).
/// `clear()` executes O(StaticFootprint(W)/cache_line) cache-line writes.
pub trait ScratchArena {
    /// Total statically pre-allocated memory in bytes.
    ///
    /// Counts only the persistent buffers allocated at construction time.
    /// Transient allocations inside solver methods are excluded.
    /// This operation is O(1).
    fn memory_bytes(&self) -> usize;

    /// Zero all scratch buffers in-place without reallocating.
    ///
    /// After this call every element of every pre-allocated buffer is `0.0`
    /// (or `Complex { re: 0.0, im: 0.0 }` for complex buffers).  The arena
    /// may be immediately reused for a subsequent simulation step.
    fn clear(&mut self);
}
use ndarray::Array3;
use num_complex::Complex;
#[cfg(feature = "parallel")]
use parking_lot::Mutex;
use std::sync::Arc;
#[cfg(not(feature = "parallel"))]
use std::sync::Mutex;

/// Pre-allocated workspace for solver operations
#[derive(Debug)]
pub struct SolverWorkspace {
    /// FFT workspace for complex operations
    pub fft_buffer: Array3<Complex<f64>>,

    /// Real-valued workspace for intermediate calculations
    pub real_buffer: Array3<f64>,

    /// K-space workspace for spectral operations
    pub k_space_buffer: Array3<f64>,

    /// Additional real workspace for in-place operations
    pub temp_buffer: Array3<f64>,

    /// Grid dimensions for validation
    grid_shape: (usize, usize, usize),
}

impl SolverWorkspace {
    /// Create a new workspace for the given grid
    pub fn new(grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);

        Self {
            fft_buffer: Array3::zeros(shape),
            real_buffer: Array3::zeros(shape),
            k_space_buffer: Array3::zeros(shape),
            temp_buffer: Array3::zeros(shape),
            grid_shape: shape,
        }
    }

    /// Validate that the workspace matches the expected grid dimensions
    pub fn validate_shape(&self, grid: &Grid) -> bool {
        self.grid_shape == (grid.nx, grid.ny, grid.nz)
    }

    /// Clear all workspace arrays (useful for debugging)
    pub fn clear(&mut self) {
        self.fft_buffer.fill(Complex::new(0.0, 0.0));
        self.real_buffer.fill(0.0);
        self.k_space_buffer.fill(0.0);
        self.temp_buffer.fill(0.0);
    }

    /// Get the statically pre-allocated memory in bytes.
    ///
    /// Counts one `Array3<Complex<f64>>` (`fft_buffer`) and three `Array3<f64>`
    /// buffers (`real_buffer`, `k_space_buffer`, `temp_buffer`).
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let complex_size = std::mem::size_of::<Complex<f64>>(); // 16 bytes
        let real_size = std::mem::size_of::<f64>();              //  8 bytes
        let n = self.grid_shape.0 * self.grid_shape.1 * self.grid_shape.2;
        // 1 × fft_buffer (Complex<f64>) + 3 × real buffers (f64)
        complex_size * n + 3 * real_size * n
    }
}

impl ScratchArena for SolverWorkspace {
    #[inline]
    fn memory_bytes(&self) -> usize {
        self.memory_usage()
    }

    fn clear(&mut self) {
        self.fft_buffer.fill(Complex::new(0.0, 0.0));
        self.real_buffer.fill(0.0);
        self.k_space_buffer.fill(0.0);
        self.temp_buffer.fill(0.0);
    }
}

/// Thread-local workspace pool for parallel operations
#[derive(Debug)]
pub struct WorkspacePool {
    /// Available workspaces
    workspaces: Arc<Mutex<Vec<SolverWorkspace>>>,

    /// Grid dimensions for workspace creation
    grid: Grid,
}

impl WorkspacePool {
    /// Create a new workspace pool
    pub fn new(grid: Grid, initial_capacity: usize) -> Self {
        let mut workspaces = Vec::with_capacity(initial_capacity);

        // Pre-allocate workspaces
        for _ in 0..initial_capacity {
            workspaces.push(SolverWorkspace::new(&grid));
        }

        Self {
            workspaces: Arc::new(Mutex::new(workspaces)),
            grid,
        }
    }

    /// Borrow a workspace from the pool
    pub fn acquire(&self) -> KwaversResult<WorkspaceGuard> {
        #[cfg(feature = "parallel")]
        let mut pool = self.workspaces.lock();
        #[cfg(not(feature = "parallel"))]
        let mut pool = match self.workspaces.lock() {
            Ok(p) => p,
            Err(e) => {
                return Err(KwaversError::System(SystemError::ResourceExhausted {
                    resource: "workspace pool".to_string(),
                    reason: format!("Failed to acquire lock: {e}"),
                }))
            }
        };

        let workspace = match pool.pop() {
            Some(ws) => ws,
            _ => {
                // Create new workspace if pool is empty
                SolverWorkspace::new(&self.grid)
            }
        };

        Ok(WorkspaceGuard {
            workspace: Some(workspace),
            pool: Arc::clone(&self.workspaces),
        })
    }

    /// Get the current pool size
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

/// RAII guard for workspace borrowing
#[derive(Debug)]
pub struct WorkspaceGuard {
    workspace: Option<SolverWorkspace>,
    pool: Arc<Mutex<Vec<SolverWorkspace>>>,
}

impl WorkspaceGuard {
    /// Get a reference to the workspace
    #[must_use]
    pub fn get(&self) -> &SolverWorkspace {
        self.workspace.as_ref().expect("Workspace already returned")
    }

    /// Get a mutable reference to the workspace
    pub fn get_mut(&mut self) -> &mut SolverWorkspace {
        self.workspace.as_mut().expect("Workspace already returned")
    }
}

impl Drop for WorkspaceGuard {
    fn drop(&mut self) {
        if let Some(workspace) = self.workspace.take() {
            // Return workspace to pool
            #[cfg(feature = "parallel")]
            self.pool.lock().push(workspace);
            #[cfg(not(feature = "parallel"))]
            if let Ok(mut pool) = self.pool.lock() {
                pool.push(workspace);
            }
        }
    }
}

/// In-place operations for memory efficiency
pub mod inplace_ops {
    use ndarray::{Array3, Zip};

    /// Add two arrays in-place: a += b
    #[inline]
    pub fn add_inplace(a: &mut Array3<f64>, b: &Array3<f64>) {
        Zip::from(a).and(b).for_each(|a, &b| *a += b);
    }

    /// Subtract two arrays in-place: a -= b
    #[inline]
    pub fn sub_inplace(a: &mut Array3<f64>, b: &Array3<f64>) {
        Zip::from(a).and(b).for_each(|a, &b| *a -= b);
    }

    /// Multiply array by scalar in-place: a *= scalar
    #[inline]
    pub fn scale_inplace(a: &mut Array3<f64>, scalar: f64) {
        a.mapv_inplace(|x| x * scalar);
    }

    /// Compute a = a * b + c in-place (fused multiply-add)
    #[inline]
    pub fn fma_inplace(a: &mut Array3<f64>, b: &Array3<f64>, c: &Array3<f64>) {
        Zip::from(a)
            .and(b)
            .and(c)
            .for_each(|a, &b, &c| *a = *a * b + c);
    }

    /// Apply a function to each element in-place
    #[inline]
    pub fn apply_inplace<F>(a: &mut Array3<f64>, f: F)
    where
        F: Fn(f64) -> f64,
    {
        a.mapv_inplace(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;

    #[test]
    fn test_workspace_creation() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
        let workspace = SolverWorkspace::new(&grid);

        assert_eq!(workspace.fft_buffer.shape(), &[64, 64, 64]);
        assert_eq!(workspace.real_buffer.shape(), &[64, 64, 64]);
        assert!(workspace.validate_shape(&grid));
    }

    #[test]
    fn test_workspace_pool() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let pool = WorkspacePool::new(grid, 2);

        assert_eq!(pool.size(), 2);

        // Acquire workspace
        let mut guard = pool.acquire().unwrap();
        assert_eq!(pool.size(), 1);

        // Use workspace
        guard.get_mut().real_buffer.fill(1.0);

        // Drop guard returns workspace to pool
        drop(guard);
        assert_eq!(pool.size(), 2);
    }

    #[test]
    fn test_inplace_operations() {
        use inplace_ops::*;

        let mut a = Array3::from_elem((10, 10, 10), 1.0);
        let b = Array3::from_elem((10, 10, 10), 2.0);
        let c = Array3::from_elem((10, 10, 10), 3.0);

        // Test add_inplace
        add_inplace(&mut a, &b);
        assert_eq!(a[[0, 0, 0]], 3.0);

        // Test scale_inplace
        scale_inplace(&mut a, 2.0);
        assert_eq!(a[[0, 0, 0]], 6.0);

        // Test fma_inplace: a = a * b + c = 6 * 2 + 3 = 15
        fma_inplace(&mut a, &b, &c);
        assert_eq!(a[[0, 0, 0]], 15.0);
    }

    // ─── ScratchArena contract tests for SolverWorkspace ───────────────────

    #[test]
    fn scratch_arena_memory_bytes_matches_memory_usage() {
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
        let ws = SolverWorkspace::new(&grid);
        // ScratchArena::memory_bytes() delegates to memory_usage(); both must agree.
        assert_eq!(ws.memory_bytes(), ws.memory_usage());
        // Quantitative: 1 complex (16 B) + 3 real (8 B) buffers × 512 elements.
        let n = 8 * 8 * 8;
        let expected = n * 16 + 3 * n * 8; // fft_buffer + real/k_space/temp
        assert_eq!(ws.memory_bytes(), expected);
    }

    #[test]
    fn scratch_arena_clear_zeros_solver_workspace() {
        let grid = Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).unwrap();
        let mut ws = SolverWorkspace::new(&grid);
        ws.fft_buffer.fill(Complex::new(1.0, 2.0));
        ws.real_buffer.fill(3.0);
        ws.k_space_buffer.fill(4.0);
        ws.temp_buffer.fill(5.0);

        ScratchArena::clear(&mut ws);

        assert!(ws.fft_buffer.iter().all(|c| c.re == 0.0 && c.im == 0.0),
            "fft_buffer not zeroed");
        assert!(ws.real_buffer.iter().all(|&v| v == 0.0),     "real_buffer not zeroed");
        assert!(ws.k_space_buffer.iter().all(|&v| v == 0.0),  "k_space_buffer not zeroed");
        assert!(ws.temp_buffer.iter().all(|&v| v == 0.0),     "temp_buffer not zeroed");
    }

    #[test]
    fn scratch_arena_memory_bytes_stable_after_clear() {
        let grid = Grid::new(6, 6, 6, 1e-3, 1e-3, 1e-3).unwrap();
        let mut ws = SolverWorkspace::new(&grid);
        let before = ws.memory_bytes();
        ws.real_buffer.fill(99.0);
        ScratchArena::clear(&mut ws);
        assert_eq!(ws.memory_bytes(), before);
    }
}
