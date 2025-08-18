//! Seismic Imaging Reconstruction Algorithms
//!
//! This module implements seismic imaging methods including Full Waveform Inversion (FWI)
//! and Reverse Time Migration (RTM), following established literature methods for accurate
//! subsurface velocity model reconstruction and structural imaging.
//!
//! ## Literature References
//!
//! 1. **Virieux & Operto (2009)**: "An overview of full-waveform inversion in
//!    exploration geophysics", Geophysics, 74(6), WCC1-WCC26
//! 2. **Tarantola (1984)**: "Inversion of seismic reflection data in the acoustic
//!    approximation", Geophysics, 49(8), 1259-1266
//! 3. **Plessix (2006)**: "A review of the adjoint-state method for computing the
//!    gradient of a functional with geophysical applications", Geophys. J. Int.
//! 4. **Pratt et al. (1998)**: "Gauss-Newton and full Newton methods in
//!    frequency-space seismic waveform inversion", Geophysical Journal International
//! 5. **Baysal et al. (1983)**: "Reverse time migration", Geophysics, 48(11), 1514-1524

pub mod constants;
pub mod config;
pub mod fwi;
pub mod rtm;
pub mod wavelet;
pub mod misfit;

// Re-export main types
pub use config::{SeismicImagingConfig, RtmImagingCondition, AnisotropyParameters};
pub use fwi::FullWaveformInversion;
pub use rtm::ReverseTimeMigration;
pub use wavelet::RickerWavelet;
pub use misfit::{MisfitFunction, MisfitType};