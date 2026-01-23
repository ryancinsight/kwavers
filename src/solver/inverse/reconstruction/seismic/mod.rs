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
//! TODO_AUDIT: P2 - Full Waveform Inversion - Implement complete FWI with nonlinear optimization, multi-scale inversion, and uncertainty quantification
//! DEPENDS ON: solver/inverse/reconstruction/seismic/fwi/nonlinear.rs, solver/inverse/reconstruction/seismic/fwi/multi_scale.rs, solver/inverse/reconstruction/seismic/fwi/uncertainty.rs
//! MISSING: Trust-region Newton method for global convergence in nonlinear FWI
//! MISSING: Multi-scale frequency continuation from low to high frequencies
//! MISSING: Hessian-based preconditioning for acceleration and regularization
//! MISSING: Uncertainty quantification using ensemble methods and Monte Carlo sampling
//! MISSING: Source encoding and simultaneous source inversion for efficiency
//! SEVERITY: HIGH (enables quantitative seismic imaging)
//! THEOREM: Born approximation: δu ≈ G δm u⁰ for small perturbations, where G is Green's function
//! THEOREM: Gauss-Newton: ∇²L ≈ J^T J where J is Jacobian of forward operator
//! REFERENCES: Tarantola (1984) Inverse Problem Theory; Virieux & Operto (2009) Geophysics; Fichtner (2011) Full Seismic Waveform Modelling and Inversion
//! 3. **Plessix (2006)**: "A review of the adjoint-state method for computing the
//!    gradient of a functional with geophysical applications", Geophys. J. Int.
//! 4. **Pratt et al. (1998)**: "Gauss-Newton and full Newton methods in
//!    frequency-space seismic waveform inversion", Geophysical Journal International
//! 5. **Baysal et al. (1983)**: "Reverse time migration", Geophysics, 48(11), 1514-1524

pub mod config;
pub mod constants;
pub mod fd_coeffs;
pub mod fwi;
pub mod misfit;
pub mod rtm;
pub mod wavelet;

// Re-export main types
pub use config::{AnisotropyParameters, RtmImagingCondition, SeismicImagingConfig};
pub use fwi::FullWaveformInversion;
pub use misfit::{MisfitFunction, MisfitType};
pub use rtm::ReverseTimeMigration;
pub use wavelet::RickerWavelet;
