//! Acoustic Wave Solver Module for Clinical Therapy Applications
//!
//! This module provides production-ready acoustic field computation for therapeutic
//! ultrasound applications including HIFU, lithotripsy, and sonoporation.
//!
//! # Module Organization
//!
//! The `AcousticWaveSolver` is the main public API for clinical applications.
//! It composes acoustic solver backends provided by the simulation layer.
//!
//! # Architecture
//!
//! ```text
//! Clinical Layer (this module)
//!     AcousticWaveSolver
//!         ↓ uses
//! Simulation Layer
//!     simulation::backends::AcousticSolverBackend (trait)
//!         ↑ implemented by
//! simulation::backends::acoustic::FdtdBackend
//!         ↓ wraps
//! Solver Layer
//!     solver::forward::fdtd::FdtdSolver
//! ```
//!
//! # Design Rationale
//!
//! By using simulation layer backends:
//! - Clinical code never depends on solver layer (only simulation)
//! - Clear layering: Clinical → Simulation → Solver
//! - Solver changes don't propagate to clinical code
//! - Simulation layer orchestrates solver access
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::therapy::therapy_integration::acoustic::AcousticWaveSolver;
//!
//! let solver = AcousticWaveSolver::new(&grid, &medium)?;
//! solver.step()?;
//! let pressure = solver.pressure_field();
//! ```

use kwavers_simulation::backends::acoustic::AcousticSolverBackend;
use leto::Array3;

mod constructors;
mod fields;
mod stepping;
#[cfg(test)]
mod tests;

/// Acoustic wave solver for therapy applications
///
/// Production-ready solver providing acoustic field simulation for therapeutic
/// ultrasound applications. Automatically selects appropriate numerical backend
/// (FDTD, PSTD) based on problem characteristics.
///
/// # Backend Selection Criteria
///
/// The solver analyzes the problem and selects the backend as follows:
///
/// | Characteristic | Threshold | Selected Backend |
/// |----------------|-----------|------------------|
/// | Points per wavelength | < 4 | PSTD (spectral accuracy) |
/// | Heterogeneity | > 30% | FDTD (handles discontinuities) |
/// | Default | - | FDTD (robust, general-purpose) |
///
/// # Thread Safety
///
/// Not thread-safe (single-threaded simulation loop). Internal operations may
/// use SIMD or GPU parallelism.
#[derive(Debug)]
pub struct AcousticWaveSolver {
    /// Solver backend (FDTD, PSTD, etc.)
    backend: Box<dyn AcousticSolverBackend>,
    /// Computational grid
    grid: kwavers_grid::Grid,
    /// Accumulated squared pressure for temporal averaging (Pa²)
    accumulated_p_squared: Array3<f64>,
}
