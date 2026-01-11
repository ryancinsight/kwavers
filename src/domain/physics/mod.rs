//! Physics Specification Layer
//!
//! This module defines abstract trait interfaces for physical laws and equations
//! that govern wave propagation phenomena. These specifications are independent
//! of the numerical method used to solve them.
//!
//! # Design Philosophy
//!
//! **Separate Specification from Implementation**
//!
//! Physics specifications (this module) define WHAT the equations are:
//! - Mathematical structure of PDEs
//! - Boundary and initial conditions
//! - Conservation laws and invariants
//! - Material property interfaces
//!
//! Solver implementations (in `solver/`) define HOW to solve them:
//! - Forward solvers: Numerical discretization (FD, FEM, spectral)
//! - Inverse solvers: Neural networks (PINNs), optimization
//! - Analytical solvers: Closed-form solutions (Green's functions, etc.)
//!
//! This separation enables:
//! 1. **Reusable validation**: Test different solvers against same specifications
//! 2. **Shared abstractions**: Material properties, boundary conditions, domains
//! 3. **Type safety**: Compile-time verification of physics constraints
//! 4. **Composability**: Combine multiple physics via trait bounds
//!
//! # Mathematical Foundation
//!
//! All wave equations are second-order hyperbolic PDEs:
//!
//! ```text
//! ∂²u/∂t² = L[u] + f
//! ```
//!
//! where:
//! - u: field variable (scalar for acoustic, vector for elastic, tensor for general)
//! - L: spatial differential operator (depends on wave type and material properties)
//! - f: source term (external forcing)
//!
//! Different wave types are characterized by:
//! - **Acoustic**: Scalar pressure field, L = c²∇²
//! - **Elastic**: Vector displacement field, L = (λ+μ)∇(∇·u) + μ∇²u
//! - **Electromagnetic**: Vector E and B fields, coupled first-order system
//!
//! # Architecture
//!
//! ```text
//! domain/physics/              ← Physics specifications (this layer)
//!     wave_equation.rs          ← Abstract wave equation traits
//!     acoustic.rs               ← Acoustic wave specialization (future)
//!     elastic.rs                ← Elastic wave specialization (future)
//!     electromagnetic.rs        ← EM wave specialization (future)
//!
//! domain/medium/                ← Material property traits (existing)
//!     acoustic.rs               ← AcousticProperties trait
//!     elastic.rs                ← ElasticProperties trait
//!
//! solver/forward/               ← Numerical implementations
//!     acoustic/                 ← FD/FEM/spectral acoustic solvers
//!     elastic/                  ← FD/FEM/spectral elastic solvers
//!
//! solver/inverse/               ← Inverse problem implementations
//!     pinn/                     ← Physics-informed neural networks
//!         acoustic.rs           ← PINN for acoustic equations
//!         elastic.rs            ← PINN for elastic equations
//!     optimization/             ← Parameter estimation, tomography
//! ```
//!
//! # Usage Pattern
//!
//! ## Implementing a Forward Solver
//!
//! ```rust,ignore
//! use kwavers::domain::physics::{WaveEquation, AcousticWaveEquation};
//! use kwavers::domain::medium::AcousticProperties;
//!
//! struct AcousticFDSolver<M: AcousticProperties> {
//!     medium: M,
//!     grid: Grid,
//!     // ... discretization parameters
//! }
//!
//! impl<M: AcousticProperties> WaveEquation for AcousticFDSolver<M> {
//!     fn domain(&self) -> &Domain { &self.grid.domain }
//!     fn spatial_operator(&self, field: &ArrayD<f64>) -> ArrayD<f64> {
//!         // Finite difference Laplacian
//!     }
//!     // ... other trait methods
//! }
//!
//! impl<M: AcousticProperties> AcousticWaveEquation for AcousticFDSolver<M> {
//!     fn sound_speed(&self) -> ArrayD<f64> { self.medium.sound_speed_array() }
//!     // ... other acoustic-specific methods
//! }
//! ```
//!
//! ## Implementing an Inverse Solver (PINN)
//!
//! ```rust,ignore
//! use kwavers::domain::physics::{WaveEquation, AcousticWaveEquation};
//! use burn::prelude::*;
//!
//! struct AcousticPINN<B: Backend> {
//!     network: NeuralNet<B>,
//!     domain: Domain,
//!     sound_speed: ArrayD<f64>,
//! }
//!
//! impl<B: Backend> WaveEquation for AcousticPINN<B> {
//!     fn domain(&self) -> &Domain { &self.domain }
//!     fn spatial_operator(&self, field: &ArrayD<f64>) -> ArrayD<f64> {
//!         // Autodiff through neural network
//!     }
//!     // ... other trait methods
//! }
//!
//! impl<B: Backend> AcousticWaveEquation for AcousticPINN<B> {
//!     fn sound_speed(&self) -> ArrayD<f64> { self.sound_speed.clone() }
//!     // ... other acoustic-specific methods
//! }
//! ```
//!
//! # References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks: A deep learning
//!   framework for solving forward and inverse problems involving nonlinear
//!   partial differential equations" - JCP 378:686-707
//! - Morse & Ingard (1968): "Theoretical Acoustics" - Princeton University Press
//! - Achenbach (1973): "Wave Propagation in Elastic Solids" - North-Holland
//! - Treeby & Cox (2010): "k-Wave: MATLAB toolbox for the simulation and
//!   reconstruction of photoacoustic wave fields" - JBO 15(2):021314

pub mod wave_equation;

// Re-export core types for convenience
pub use wave_equation::{
    AcousticWaveEquation, AutodiffElasticWaveEquation, AutodiffWaveEquation, BoundaryCondition,
    Domain, ElasticWaveEquation, SourceTerm, SpatialDimension, TimeIntegration, WaveEquation,
};

// Future modules (to be implemented as needed):
// pub mod acoustic;      // Specialized acoustic wave equation types
// pub mod elastic;       // Specialized elastic wave equation types
// pub mod electromagnetic; // Maxwell's equations
// pub mod coupled;       // Multi-physics coupling (acoustic-elastic, etc.)
