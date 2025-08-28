//! Focused Transducer Sources Module
//!
//! This module provides focused transducer geometries compatible with k-Wave toolbox,
//! including bowl transducers, arc sources, and multi-element arrays.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for transducer geometry generation
//! - **DRY**: Reusable geometric primitives and calculations
//! - **Zero-Copy**: Uses iterators and in-place operations
//! - **KISS**: Clear interfaces for complex geometries
//!
//! # Literature References
//! - O'Neil (1949): "Theory of focusing radiators"
//! - Cobbold (2007): "Foundations of Biomedical Ultrasound"
//! - Szabo (2014): "Diagnostic Ultrasound Imaging"

pub mod arc;
pub mod bowl;
pub mod multi_bowl;
pub mod utils;

pub use arc::{ArcConfig, ArcSource};
pub use bowl::{BowlConfig, BowlTransducer};
pub use multi_bowl::MultiBowlArray;
pub use utils::{make_annular_array, make_bowl, ApodizationType};