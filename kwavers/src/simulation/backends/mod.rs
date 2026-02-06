//! Solver backend adapters for simulation layer
//!
//! This module provides concrete backend implementations that adapt low-level
//! solvers to simulation-facing interfaces. Backends are instantiated and managed
//! by the simulation layer, not directly by the clinical layer.
//!
//! # Architecture
//!
//! ```text
//! Clinical Layer
//!     ↓ uses
//! Simulation Layer (provides high-level APIs)
//!     ↓ contains & instantiates
//! simulation::backends (concrete solver adapters)
//!     ↓ wraps
//! Solver Layer (low-level numerical methods)
//! ```
//!
//! # Design Rationale
//!
//! By placing backends in the simulation layer:
//! - Clinical code never depends on solver layer implementation details
//! - Solver changes don't propagate up to clinical code
//! - Backends can be composed and orchestrated by simulation
//! - Clear dependency direction: Clinical → Simulation → Solver

pub mod acoustic;

pub use acoustic::AcousticSolverBackend;
