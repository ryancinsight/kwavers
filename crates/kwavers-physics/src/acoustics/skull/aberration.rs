//! Aberration correction for transcranial ultrasound.
//!
//! # Theorem: time-reversal reciprocity
//!
//! The lossless linear acoustic wave equation is invariant under `t -> -t`.
//! If `p(x,t)` is a solution, then `p(x,T-t)` is also a solution. Recording the
//! distorted field at aperture elements and retransmitting its time reverse
//! therefore refocuses at the original source under the same medium.
//!
//! # Phase-screen theorem
//!
//! For a planar array and a thin skull phase screen, the accumulated phase along
//! a z-propagating ray is
//!
//! ```text
//! Phi(x,y,z) = sum_0^z [2 pi f / c(x,y,z') - 2 pi f / c_water] dz.
//! ```
//!
//! Setting the drive correction to `-Phi(x_i,y_i,z_max)` cancels the skull phase
//! term in the Born single-scattering approximation, leaving only geometric
//! focusing phase at the target.
//!
//! # References
//!
//! - Fink M. (1992). Time reversal of ultrasonic fields. IEEE TUFFC 39(5), 555-566.
//! - Clement G. T., Hynynen K. (2002). Phys. Med. Biol. 47(8), 1219-1236.
//! - Aubry J-F. et al. (2003). J. Acoust. Soc. Am. 113(1), 84-93.
//! - Pinton G. et al. (2012). Med. Phys. 39(1), 299-307.

mod aperture;
mod constants;
mod elements;
mod model;
mod phase;

#[cfg(test)]
mod tests;

pub use model::AberrationCorrection;
