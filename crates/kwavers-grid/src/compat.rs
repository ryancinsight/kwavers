//! Compatibility surfaces for legacy third-party array APIs.
//!
//! This module centralizes transitional ndarray imports so call sites can
//! migrate incrementally without depending on `ndarray` directly.

pub mod ndarray {
    pub use leto::{Array1, Array2, Array3, ArrayView3, ArrayViewMut3};
}
