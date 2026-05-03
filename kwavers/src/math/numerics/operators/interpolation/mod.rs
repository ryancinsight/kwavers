//! Interpolation Operators
//!
//! Unified spatial interpolation operators for heterogeneous media,
//! sensor data extraction, and grid refinement.
//!
//! ## References
//!
//! - Press et al. (2007). *Numerical Recipes*. Chapter 3.
//! - de Boor, C. (2001). *A Practical Guide to Splines*.

pub mod linear;
#[cfg(test)]
mod tests;
pub mod traits;
pub mod trilinear;

pub use linear::LinearInterpolator;
pub use traits::Interpolator;
pub use trilinear::TrilinearInterpolator;
