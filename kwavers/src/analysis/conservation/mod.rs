//! Conservation Law Verification Framework
//!
//! Verifies that GPU multiphysics simulations conserve mass, energy, and momentum
//! within numerical precision limits.
//!
//! ## Theory
//!
//! Conservation laws are fundamental constraints in physics:
//! - **Mass Conservation:** ∫ρ dV = constant (no sources/sinks)
//! - **Momentum Conservation:** ∫ρu dV = constant (no external forces)
//! - **Energy Conservation:** ∫(KE + PE + TE) dV = constant (adiabatic)
//!
//! For discrete simulations with conservative schemes, violations should be O(ε_machine).
//!
//! ## References
//!
//! - k-Wave: Energy conservation in acoustic FDTD
//! - fullwave25: Nonlinear acoustic energy balance
//! - BabelBrain: Thermal energy conservation in HIFU
//! - mSOUND: Mass conservation in multiphysics coupling

pub mod checkers;
pub mod detectors;
pub mod reports;

pub use checkers::{ConservationChecker, ConservationLaw};
pub use detectors::{ConservationViolation, ConservationViolationDetector};
pub use reports::{ConservationReport, ConservationViolationAnalysis};
