//! Plane Wave Compounding for Real-Time Ultrasound Imaging
//!
//! Multi-angle plane wave insonification with coherent compounding achieves
//! 10× frame rate improvement over focused beam imaging.
//!
//! | Submodule  | Contents                                                   |
//! |------------|------------------------------------------------------------|
//! | `config`   | `PlaneWaveCompoundingConfig` — imaging parameters and defaults       |
//! | `compound` | `PlaneWaveCompound` — field generation, beamforming, DAS  |
//!
//! ## References
//! - Montaldo et al. (2009): "Coherent plane-wave compounding for very high frame rate."
//!   *IEEE UFFC*, 56(3), 489–506.
//! - Jensen et al. (2016): "Plane wave imaging." *Ultrasound Imaging*.

mod compound;
mod config;
#[cfg(test)]
mod tests;

pub use compound::PlaneWaveCompound;
pub use config::PlaneWaveCompoundingConfig;
