//! Transducer design components
//!
//! Modular transducer design following Single Responsibility Principle

pub mod geometry;

pub use geometry::ElementGeometry;

// Re-export commonly used types
pub use crate::error::KwaversResult;