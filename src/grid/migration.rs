//! Migration helpers for transitioning to the new Grid API
//!
//! This module provides compatibility shims to help migrate from the old
//! public-field Grid to the new encapsulated version.

use super::structure_proper::Grid as ProperGrid;
use crate::error::KwaversResult;
use uom::si::f64::Length;
use uom::si::length::meter;

/// Compatibility wrapper that provides the old API while using the new implementation
///
/// This allows gradual migration by providing field-like access through methods.
#[derive(Debug, Clone)]
pub struct GridCompat {
    inner: ProperGrid,
}

impl GridCompat {
    /// Create from the new Grid type
    pub fn from_proper(grid: ProperGrid) -> Self {
        Self { inner: grid }
    }
    
    /// Get the inner proper Grid
    pub fn into_proper(self) -> ProperGrid {
        self.inner
    }
    
    /// Create with old API (f64 meters)
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        // This mimics the old panicking behavior for compatibility
        let grid = ProperGrid::from_meters(nx, ny, nz, dx, dy, dz)
            .expect("Invalid grid parameters");
        Self { inner: grid }
    }
    
    /// Create with validation (old API)
    pub fn create(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<Self> {
        Ok(Self {
            inner: ProperGrid::from_meters(nx, ny, nz, dx, dy, dz)?,
        })
    }
    
    // Provide field-like access through methods
    // These allow code like `grid.nx` to be replaced with `grid.nx()`
    // with minimal changes
    
    #[inline]
    pub fn nx(&self) -> usize {
        self.inner.nx()
    }
    
    #[inline]
    pub fn ny(&self) -> usize {
        self.inner.ny()
    }
    
    #[inline]
    pub fn nz(&self) -> usize {
        self.inner.nz()
    }
    
    #[inline]
    pub fn dx(&self) -> f64 {
        self.inner.dx_meters()
    }
    
    #[inline]
    pub fn dy(&self) -> f64 {
        self.inner.dy_meters()
    }
    
    #[inline]
    pub fn dz(&self) -> f64 {
        self.inner.dz_meters()
    }
    
    // Additional compatibility methods
    
    #[inline]
    pub fn dim(&self) -> (usize, usize, usize) {
        self.inner.dim()
    }
    
    #[inline]
    pub fn total_points(&self) -> usize {
        self.inner.total_points()
    }
    
    #[inline]
    pub fn is_uniform(&self) -> bool {
        self.inner.is_uniform()
    }
}

/// Macro to help migrate field access to method calls
///
/// Usage: `migrate_grid_access!(grid.nx)` expands to `grid.nx()`
#[macro_export]
macro_rules! migrate_grid_access {
    ($grid:expr . nx) => { $grid.nx() };
    ($grid:expr . ny) => { $grid.ny() };
    ($grid:expr . nz) => { $grid.nz() };
    ($grid:expr . dx) => { $grid.dx() };
    ($grid:expr . dy) => { $grid.dy() };
    ($grid:expr . dz) => { $grid.dz() };
}

/// Helper trait for migrating solver caches
///
/// Solvers that previously relied on Grid's k_squared_cache should implement this
pub trait SolverCache {
    /// Pre-compute k-squared array for the solver
    fn precompute_k_squared(grid: &ProperGrid) -> ndarray::Array3<f64> {
        use std::f64::consts::PI;
        
        let nx = grid.nx();
        let ny = grid.ny();
        let nz = grid.nz();
        let dx = grid.dx_meters();
        let dy = grid.dy_meters();
        let dz = grid.dz_meters();
        
        let mut k_squared = ndarray::Array3::zeros((nx, ny, nz));
        
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let kx = if i <= nx / 2 {
                        2.0 * PI * i as f64 / (nx as f64 * dx)
                    } else {
                        2.0 * PI * (i as i32 - nx as i32) as f64 / (nx as f64 * dx)
                    };
                    
                    let ky = if j <= ny / 2 {
                        2.0 * PI * j as f64 / (ny as f64 * dy)
                    } else {
                        2.0 * PI * (j as i32 - ny as i32) as f64 / (ny as f64 * dy)
                    };
                    
                    let kz = if k <= nz / 2 {
                        2.0 * PI * k as f64 / (nz as f64 * dz)
                    } else {
                        2.0 * PI * (k as i32 - nz as i32) as f64 / (nz as f64 * dz)
                    };
                    
                    k_squared[[i, j, k]] = kx * kx + ky * ky + kz * kz;
                }
            }
        }
        
        k_squared
    }
}

/// Migration guide documentation
pub mod migration_guide {
    //! # Grid Migration Guide
    //!
    //! ## Changes
    //! 
    //! 1. **Private Fields**: Grid fields are now private. Use getter methods:
    //!    - `grid.nx` → `grid.nx()`
    //!    - `grid.dx` → `grid.dx()` (returns `Length`) or `grid.dx_meters()` (returns `f64`)
    //!
    //! 2. **Unit Safety**: Grid spacing now uses `uom::Length` for compile-time unit checking:
    //!    ```rust
    //!    use uom::si::f64::Length;
    //!    use uom::si::length::millimeter;
    //!    
    //!    let grid = Grid::new(
    //!        100, 100, 100,
    //!        Length::new::<millimeter>(1.0),
    //!        Length::new::<millimeter>(1.0),
    //!        Length::new::<millimeter>(1.0),
    //!    )?;
    //!    ```
    //!
    //! 3. **No Panics**: `Grid::new` now returns `Result` instead of panicking:
    //!    ```rust
    //!    // Old (panics on error)
    //!    let grid = Grid::new(0, 0, 0, 1.0, 1.0, 1.0); // Panic!
    //!    
    //!    // New (returns Result)
    //!    let grid = Grid::new(...)? ; // Proper error handling
    //!    ```
    //!
    //! 4. **K-Space Cache Removed**: Solvers should manage their own caches:
    //!    ```rust
    //!    struct MyPstdSolver {
    //!        grid: Grid,
    //!        k_squared: Array3<f64>, // Solver owns its cache
    //!    }
    //!    
    //!    impl MyPstdSolver {
    //!        fn new(grid: Grid) -> Self {
    //!            let k_squared = SolverCache::precompute_k_squared(&grid);
    //!            Self { grid, k_squared }
    //!        }
    //!    }
    //!    ```
    //!
    //! ## Migration Steps
    //!
    //! 1. Replace direct field access with getter methods
    //! 2. Update constructors to handle `Result`
    //! 3. Move k-space caches to solver implementations
    //! 4. Consider using unit-safe APIs for new code
    //!
    //! ## Compatibility
    //!
    //! Use `GridCompat` for gradual migration:
    //! ```rust
    //! use crate::grid::migration::GridCompat;
    //! 
    //! let grid = GridCompat::new(100, 100, 100, 0.001, 0.001, 0.001);
    //! let nx = grid.nx(); // Method call, not field access
    //! ```
}