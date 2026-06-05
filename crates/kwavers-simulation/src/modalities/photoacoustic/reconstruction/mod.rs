//! Image Reconstruction Algorithms for Photoacoustic Imaging
//!
//! Implements reconstruction algorithms for photoacoustic image formation
//! from time-resolved pressure measurements at detector arrays.
//!
//! ## Mathematical Foundation
//!
//! ### Universal Back-Projection (UBP)
//!
//! ```text
//! p₀(r) = Σᵢ (1/|r - rᵢ|) · pᵢ(t = |r - rᵢ|/c)
//! ```
//!
//! ## References
//!
//! - Xu & Wang (2005): "Universal back-projection algorithm for photoacoustic computed tomography"
//!   *Physical Review E* 71(1), 016706.
//! - Treeby et al. (2010): "k-Wave: MATLAB toolbox for simulation and reconstruction"
//!   *Journal of Biomedical Optics* 15(2), 021314.

pub mod core;
#[cfg(test)]
mod tests;

pub use core::{
    compute_detector_positions, interpolate_detector_signal, time_reversal_reconstruction,
};
