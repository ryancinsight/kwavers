//! Focused transducer sources module
//!
//! This module provides focused transducer geometries compatible with k-Wave toolbox,
//! including bowl transducers, arc sources, and multi-element arrays.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for transducer geometry generation
//! - **DRY**: Reusable geometric primitives and calculations
//! - **Zero-Copy**: Uses iterators and in-place operations
//!
//! # Literature References
//! - O'Neil (1949): "Theory of focusing radiators"
//! - Cobbold (2007): "Foundations of Biomedical Ultrasound"
//! - Szabo (2014): "Diagnostic Ultrasound Imaging"

pub mod bowl;
pub mod geometry;
pub mod pressure_field;
pub mod validation;

pub use bowl::{BowlConfig, BowlTransducer};
pub use geometry::{FocusedGeometry, GeometryCalculator};
pub use pressure_field::PressureFieldGenerator;
pub use validation::TransducerValidator;