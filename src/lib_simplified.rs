//! # Kwavers: Acoustic Simulation Library (Simplified)
//!
//! Core acoustic wave simulation functionality without unnecessary complexity.

// Essential modules only
pub mod boundary;
pub mod error;
pub mod grid;
pub mod medium;
pub mod physics;
pub mod solver;
pub mod source;

// Core re-exports for ease of use
pub use error::{KwaversError, KwaversResult};
pub use grid::Grid;
pub use medium::{Medium, HomogeneousMedium};
pub use source::Source;

// Solver exports - direct access, no factory needed
pub use solver::fdtd::{FdtdConfig, FdtdSolver};
pub use solver::pstd::{PstdConfig, PstdSolver};

// Boundary conditions
pub use boundary::{Boundary, PMLBoundary, PMLConfig};

// Simple example of intended usage:
// ```rust
// let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
// let medium = HomogeneousMedium::water(&grid);
// let solver = FdtdSolver::new(grid, medium, FdtdConfig::default());
// solver.run(100)?;
// ```

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize logging (optional)
pub fn init_logging() {
    env_logger::init();
}