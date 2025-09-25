//! System-level error types
//!
//! Infrastructure, I/O, and system resource error hierarchies

pub mod io;
pub mod grid;
pub mod numerical;

// Re-export system errors
pub use io::*;
pub use grid::*;
pub use numerical::*;