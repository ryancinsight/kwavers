//! Diverging wave imaging and synthetic transmit aperture (STA).
//!
//! This module implements virtual-source delay calculations for diverging wave
//! ultrasound imaging. Configuration lives in `config`, processor kernels live
//! in `processor`, and value-semantic specifications live in `tests`.
//!
//! # Mathematical Foundation
//!
//! ## Virtual Source Model
//!
//! A diverging wave is produced by a virtual point source at depth `F` behind
//! the transducer face (`z = -F`, `F > 0`).
//!
//! ```text
//! tau_tx(x,z,i) = (sqrt((x - x_i)^2 + (z + F)^2) - F) / c
//! tau_rx(x,z,j) = sqrt((x - x_j)^2 + z^2) / c
//! tau_sta(x,z,i,j) = tau_tx(x,z,i) + tau_rx(x,z,j)
//! ```
//!
//! ## PRF Limit
//!
//! For unambiguous imaging to depth `z_max`, `PRF_max = c / (2 z_max)` by the
//! round-trip travel-time bound.
//!
//! # References
//!
//! - Jensen, J.A., et al. (2006). Synthetic aperture ultrasound imaging.
//! - Papadacci, C., et al. (2014). High-contrast ultrafast imaging of the heart.
//! - Tanter, M., & Fink, M. (2014). Ultrafast imaging in biomedical ultrasound.

mod config;
mod processor;

pub use config::DivergingWaveConfig;
pub use processor::DivergingWave;

#[cfg(test)]
mod tests;
