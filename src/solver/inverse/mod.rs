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

pub mod pinn;
pub mod reconstruction;
pub mod seismic;
pub mod time_reversal;

pub use pinn::{
    elastic_2d, AdaptiveRefinement, CollocationSampler, InterfaceCondition, MultiRegionDomain,
    SamplingStrategy,
};
pub use reconstruction::{
    FilterType, InterpolationMethod, ReconstructionAlgorithm, ReconstructionConfig, Reconstructor,
    UniversalBackProjection, WeightFunction,
};
pub use time_reversal::{TimeReversalConfig, TimeReversalReconstructor};
