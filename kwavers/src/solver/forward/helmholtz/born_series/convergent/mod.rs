//! Convergent Born Series (CBS) solver for the Helmholtz equation.
//!
//! ## Mathematical Foundation
//!
//! CBS iteration: `ψ_{n+1} = ψ_n - G * (k²V ψ_n)`
//!
//! Converges when `‖k²V G‖ < 1`, which is less restrictive than the
//! standard Born convergence criterion `‖k²V G‖ < ½`.
//!
//! ## References
//!
//! 1. Stanziola, A., et al. (2025). "Iterative Born Solver for the Acoustic
//!    Helmholtz Equation with Heterogeneous Sound Speed and Density"
//! 2. de Hoop, M. V. (1995). "Convergent Born series for acoustic and elastic
//!    wave equations"

pub mod green;
pub mod iteration;
pub mod solver;
pub mod stats;
#[cfg(test)]
mod tests;

pub use solver::ConvergentBornSolver;
pub use stats::ConvergentBornStats;
