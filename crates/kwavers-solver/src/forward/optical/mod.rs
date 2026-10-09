//! Optical fluence solver for photoacoustic tomography.
//!
//! Implements the steady-state diffusion approximation to the Radiative
//! Transfer Equation, yielding fluence Phi(r) and absorbed energy
//! H = mu_a * Phi as the photoacoustic source term.

pub mod diffusion;
pub(crate) mod plugin;
pub(crate) mod solver;

pub use diffusion::{DiffusionSolver, DiffusionSolverConfig};
pub use plugin::OpticalDiffusionPlugin;
pub use solver::OpticalDiffusionSolver;