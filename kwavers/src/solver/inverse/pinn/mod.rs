//! Physics-Informed Neural Networks (PINNs) for Inverse Problems
//!
//! This module contains PINN implementations for solving inverse problems in
//! wave propagation. PINNs use neural networks as universal approximators for
//! PDE solutions, trained to satisfy:
//! 1. PDE residuals in the domain interior
//! 2. Boundary conditions on domain boundaries
//! 3. Initial conditions (for time-dependent problems)
//! 4. Observed data (for inverse parameter estimation)
//!
//! # Architectural Rationale
//!
//! **Why PINNs are in `solver/inverse/` instead of `analysis/ml/`:**
//!
//! PINNs are **solvers**, not analysis tools. They solve PDEs (forward problem)
//! and estimate parameters (inverse problem) using neural networks as the
//! representation. This is fundamentally different from:
//! - **Analysis tools** (`analysis/ml/`): Post-processing trained models,
//!   feature extraction, uncertainty quantification on existing solutions
//! - **Forward solvers** (`solver/forward/`): Numerical discretization methods
//!   (finite difference, finite element, spectral) that integrate PDEs
//!
//! PINNs occupy a unique position:
//! - They SOLVE the same wave equations as forward solvers (same physics traits)
//! - They use neural networks + autodiff instead of grid-based discretization
//! - They can handle inverse problems (parameter estimation) naturally via
//!   gradient descent on both network weights and physical parameters
//!
//! # Design Philosophy
//!
//! **Shared Abstractions, Different Implementations**
//!
//! PINNs and forward solvers both implement the same `domain::physics` traits:
//! - `WaveEquation` - abstract wave equation interface
//! - `AcousticWaveEquation` - acoustic wave specialization
//! - `ElasticWaveEquation` - elastic wave specialization
//!
//! The key difference:
//! - **Forward solvers**: Discretize space → compute derivatives numerically →
//!   integrate in time using explicit/implicit schemes
//! - **PINNs**: Neural network u_θ(x,t) → compute derivatives via autodiff →
//!   optimize θ to minimize PDE residual
//!
//! This allows:
//! - Validation logic to be shared (both solve same equations)
//! - Material properties and geometry to be reused
//! - Hybrid solvers (PINN for some regions, FD for others)
//!
//! # Mathematical Foundation
//!
//! ## Forward Problem (PDE solving)
//!
//! Given a PDE:
//! ```text
//! L[u] = f   in Ω
//! B[u] = g   on ∂Ω
//! ```
//!
//! where L is a differential operator, B is a boundary operator.
//!
//! PINN approximates u ≈ u_θ(x) where θ are neural network weights, trained
//! to minimize the physics-informed loss:
//!
//! ```text
//! L_total = λ_pde · L_pde + λ_bc · L_bc
//!
//! L_pde = (1/N_i) Σ ||L[u_θ](x_i) - f(x_i)||²   (PDE residual)
//! L_bc  = (1/N_b) Σ ||B[u_θ](x_b) - g(x_b)||²   (boundary condition)
//! ```
//!
//! where {x_i} are interior collocation points and {x_b} are boundary points.
//!
//! ## Inverse Problem (parameter estimation)
//!
//! Given observations u_obs at points {x_obs} and unknown parameters p:
//!
//! ```text
//! L[u; p] = f   (PDE depends on parameters p)
//! ```
//!
//! PINN jointly optimizes network weights θ AND parameters p to minimize:
//!
//! ```text
//! L_total = λ_pde · L_pde + λ_bc · L_bc + λ_data · L_data
//!
//! L_data = (1/N_obs) Σ ||u_θ(x_obs) - u_obs||²   (data fitting)
//! ```
//!
//! This enables estimation of:
//! - Sound speed fields c(x) from time-of-flight data
//! - Elastic moduli λ(x), μ(x) from strain measurements
//! - Source locations and strengths from recorded waveforms
//!
//! # Framework Integration
//!
//! ## Burn Backend Strategy
//!
//! PINNs use the Burn deep learning framework for:
//! - Automatic differentiation (compute ∂u/∂x, ∂²u/∂x², etc.)
//! - GPU acceleration (training on large collocation point sets)
//! - Model serialization and deployment
//!
//! Backend options:
//! - **NdArray** (default): CPU-only, zero-copy interop with forward solvers
//! - **WGPU**: Cross-platform GPU (Vulkan/Metal/DX12)
//! - **CUDA**: NVIDIA GPU (best performance for large-scale training)
//!
//! ## Tensor Interoperability
//!
//! The `domain::tensor` module provides conversion between ndarray (used by
//! forward solvers) and Burn tensors (used by PINNs):
//!
//! ```rust,ignore
//! // Forward solver output (ndarray)
//! let pressure_field: ArrayD<f64> = acoustic_solver.pressure();
//!
//! // Convert to Burn tensor for PINN training
//! let burn_tensor = Tensor::<B, 3>::from_data(pressure_field.into());
//!
//! // Train PINN to match forward solver (verification)
//! pinn.train(burn_tensor, collocation_points);
//! ```
//!
//! # Module Organization
//!
//! ```text
//! solver/inverse/pinn/
//!     mod.rs              ← This file (PINN framework)
//!     geometry.rs         ← Collocation sampling, interface conditions
//!     elastic_2d/         ← 2D elastic wave PINN
//!         config.rs       ← Training hyperparameters
//!         model.rs        ← Neural network architecture
//!         loss.rs         ← Physics-informed loss functions
//!         training.rs     ← Training loop and optimizer
//!         inference.rs    ← Trained model deployment
//!     elastic_3d/         ← 3D elastic wave PINN (future)
//!     acoustic_2d/        ← 2D acoustic wave PINN (future)
//!     acoustic_3d/        ← 3D acoustic wave PINN (future)
//!     coupled/            ← Multi-physics PINNs (future)
//! ```
//!
//! # References
//!
//! - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
//!   "Physics-informed neural networks: A deep learning framework for solving
//!   forward and inverse problems involving nonlinear partial differential
//!   equations." Journal of Computational Physics, 378, 686-707.
//!   DOI: 10.1016/j.jcp.2018.10.045
//!
//! - Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S.,
//!   & Yang, L. (2021). "Physics-informed machine learning." Nature Reviews
//!   Physics, 3(6), 422-440. DOI: 10.1038/s42254-021-00314-5
//!
//! - Jagtap, A. D., Kawaguchi, K., & Karniadakis, G. E. (2020).
//!   "Adaptive activation functions accelerate convergence in deep and
//!   physics-informed neural networks." Journal of Computational Physics, 404, 109136.
//!   DOI: 10.1016/j.jcp.2019.109136
//!
//! - Wang, S., Teng, Y., & Perdikaris, P. (2021).
//!   "Understanding and mitigating gradient flow pathologies in physics-informed
//!   neural networks." SIAM Journal on Scientific Computing, 43(5), A3055-A3081.
//!   DOI: 10.1137/20M1318043

pub mod elastic_2d;
pub mod geometry;
pub mod ml;

#[cfg(feature = "pinn")]
pub mod beamforming;

// Re-export key types for convenience
pub use geometry::{
    AdaptiveRefinement, CollocationSampler, InterfaceCondition, MultiRegionDomain, SamplingStrategy,
};

#[cfg(feature = "pinn")]
pub use beamforming::{create_burn_beamforming_provider, BurnPinnBeamformingAdapter};

// Modules planned but not yet implemented:
// - elastic_3d: 3D elastic wave PINN solver
// - acoustic_2d: 2D acoustic wave PINN solver
// - acoustic_3d: 3D acoustic wave PINN solver
// - electromagnetic_2d: 2D electromagnetic wave PINN solver
// - electromagnetic_3d: 3D electromagnetic wave PINN solver
// - coupled: Multi-physics coupling PINNs (acoustic-elastic, thermo-acoustic, etc.)
