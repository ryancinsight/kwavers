//! Inverse methods
//!
//! This module contains solvers for inverse problems that estimate causes
//! from observed effects, including reconstruction, time reversal, and
//! physics-informed neural networks (PINNs).
//!
//! # Module Organization
//!
//! - `reconstruction`: Image reconstruction from measured data
//! - `seismic`: Seismic inversion methods
//! - `time_reversal`: Time-reversal focusing and imaging
//! - `pinn`: Physics-Informed Neural Networks for PDE solving and parameter estimation
//!
//! # Architecture
//!
//! All inverse solvers share the same physics specifications from `domain::physics`
//! but differ in their solution approach:
//! - Traditional methods: Optimization, adjoint methods, iterative reconstruction
//! - PINN methods: Neural network approximation with physics-informed loss

pub mod elastography;
pub mod fwi;
pub mod linear_born_inversion;
pub mod marchenko;
pub mod pinn;
pub mod reconstruction;
pub mod rytov;
pub mod same_aperture;
pub mod seismic;
pub mod time_reversal;

pub use pinn::{
    elastic_2d, AdaptiveRefinement, CollocationSampler, CollocationSamplingStrategy,
    MultiRegionDomain, PinnGeometryInterfaceCondition,
};
pub use reconstruction::{
    ReconstructionAlgorithm, ReconstructionConfig, ReconstructionFilterType,
    ReconstructionInterpolationMethod, Reconstructor, UniversalBackProjection, WeightFunction,
};
pub use time_reversal::{TimeReversalConfig, TimeReversalReconstructor};
