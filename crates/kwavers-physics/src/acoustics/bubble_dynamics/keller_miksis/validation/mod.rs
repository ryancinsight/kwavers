//! Keller-Miksis validation tests.
//!
//! Value-semantic regression tests for the compressible Keller-Miksis ODE,
//! its thermodynamic auxiliary updates, and the Plesset-Prosperetti shape-
//! stability coupling. Tests are partitioned by physical responsibility:
//!
//! - [`dynamics`]: K-M wall-motion ODE (compression, expansion, acoustic
//!   forcing, Mach limiting, radiation damping, physical bounds).
//! - [`thermodynamics`]: heat capacity, vapor mass transfer, adiabatic and
//!   conductive temperature updates, Van der Waals bubble pressure.
//! - [`shape_stability`]: shape-mode seeding, Plesset breakup detection,
//!   collapse-driven mode growth, capillary boundedness at rest.

mod dynamics;
mod shape_stability;
mod thermodynamics;
