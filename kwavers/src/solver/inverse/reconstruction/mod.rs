//! Reconstruction algorithms for acoustic field recovery
//!
//! This module implements various reconstruction algorithms, including plane and
//! line array reconstruction methods.
//!
//! ## Literature References
//!
//! 1. **Treeby & Cox (2010)**: "MATLAB toolbox for the simulation and
//!    reconstruction of photoacoustic wave fields", J. Biomed. Opt.
pub mod photoacoustic;
pub mod plane_recon;
pub mod seismic;
pub mod unified_sirt;

mod back_projection;
mod config;
mod filters;
mod interpolation;

pub use back_projection::{UniversalBackProjection, WeightFunction};
pub use config::{
    FilterType, InterpolationMethod, ReconstructionAlgorithm, ReconstructionConfig, Reconstructor,
};
pub use filters::apply_reconstruction_filter;
pub use interpolation::interpolate_3d;
pub use unified_sirt::{SirtAlgorithm, SirtConfig, SirtReconstructor, SirtResult};
