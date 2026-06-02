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
//!
//! ## Not yet implemented
//!
//! - **Trust-region Newton FWI**: Global convergence guarantee for nonlinear inversion
//!   (Tarantola 1984; Virieux & Operto 2009, Geophysics).
//! - **Multi-scale frequency continuation**: Sequential low-to-high frequency inversion
//!   to avoid local minima (Fichtner 2011, Full Seismic Waveform Modelling).
//! - **Uncertainty quantification**: Ensemble or Monte Carlo sampling of the posterior.
//!
//! 3. **Plessix (2006)**: "A review of the adjoint-state method for computing the
//!    gradient of a functional with geophysical applications", Geophys. J. Int.
//! 4. **Pratt et al. (1998)**: "Gauss-Newton and full Newton methods in
//!    frequency-space seismic waveform inversion", Geophysical Journal International
//! 5. **Baysal et al. (1983)**: "Reverse time migration", Geophysics, 48(11), 1514-1524

pub mod config;
pub mod constants;
pub mod fd_coeffs;
pub mod misfit;
pub mod rtm;
pub mod wavelet;

// Re-export main types. Time-domain FWI is consolidated under
// `solver::inverse::fwi::time_domain` (factory-dispatched after T15 lands);
// the parallel reconstruction/seismic/fwi stack (custom inline FDTD stencil
// with zero external consumers) was removed 2026-05-20 — see backlog T16.
pub use config::{AnisotropyParameters, RtmImagingCondition, SeismicImagingConfig};
pub use misfit::{MisfitFunction, MisfitType};
pub use rtm::ReverseTimeMigration;
pub use wavelet::SeismicRickerWavelet;
