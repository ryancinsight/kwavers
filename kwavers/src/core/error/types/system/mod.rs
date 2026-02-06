//! System-level error types
//!
//! Infrastructure, I/O, and system resource error hierarchies

pub mod grid;
pub mod io;
pub mod numerical;

// Explicit re-exports of system error types
pub use grid::GridErrorType;
pub use io::IoErrorType;
pub use numerical::NumericalErrorType;
