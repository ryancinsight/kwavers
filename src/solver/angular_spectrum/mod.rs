//! Hybrid Angular Spectrum Method for Efficient Wave Propagation
//!
//! This module implements the Hybrid Angular Spectrum (HAS) method for ultrasound
//! wave propagation, combining the efficiency of angular spectrum methods with
//! the accuracy of local corrections for inhomogeneous media.
//!
//! ## Overview
//!
//! The Hybrid Angular Spectrum method efficiently propagates acoustic waves by:
//! 1. Using angular spectrum decomposition for smooth regions
//! 2. Applying local corrections near inhomogeneities
//! 3. Maintaining phase accuracy through proper boundary matching
//!
//! ## Mathematical Foundation
//!
//! The angular spectrum representation of a field is:
//!
//! p(x,y,z) = (1/(2π)) ∫∫ P(kx,ky,z) exp(j(kx x + ky y)) dkx dky
//!
//! Where P(kx,ky,z) is the angular spectrum at depth z.
//!
//! ## Hybrid Approach
//!
//! For homogeneous regions: Direct angular spectrum propagation
//! For inhomogeneous regions: Split-step corrections with local operators
//!
//! ## References
//!
//! - Zeng, X., & McGough, R. J. (2008). "Evaluation of the angular spectrum approach
//!   for simulations of near-field pressures." *JASA*, 123(1), 68-76.
//! - Christopher, D. A., & Parker, K. J. (1991). "New approaches to the linear propagation
//!   of acoustic fields." *JASA*, 90(1), 507-521.
//! - Tabei, M., et al. (2002). "A hybrid angular spectrum method." *JASA*, 112(6), 2887-2893.

pub mod hybrid_solver;
pub mod angular_spectrum;
pub mod local_corrections;

pub use hybrid_solver::{HybridAngularSpectrumSolver, HASConfig, PropagationMode};
pub use angular_spectrum::{AngularSpectrum, SpectrumConfig};
pub use local_corrections::{LocalCorrection, CorrectionType, CorrectionConfig};








