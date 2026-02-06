//! Physics trait implementations for 2D Elastic PINN
//!
//! This module provides trait implementations that allow the PINN model to satisfy
//! the domain-layer physics specifications defined in `domain::physics::ElasticWaveEquation`.
//!
//! # Architecture
//!
//! The PINN neural network (`ElasticPINN2D`) is a solver implementation, not a physics
//! specification. To enable shared validation, testing, and comparison with other solvers
//! (FD, FEM, spectral), we wrap the PINN in a struct that implements the physics traits.
//!
//! ```text
//! domain::physics::ElasticWaveEquation (trait specification)
//!        ↑
//!        | implements
//!        |
//! ElasticPINN2DSolver (wrapper struct)
//!        |
//!        | contains
//!        |
//!        +-> ElasticPINN2D<B> (neural network)
//!        +-> Domain (spatial domain spec)
//!        +-> Material parameters (λ, μ, ρ)
//! ```
//!
//! # Design Rationale
//!
//! **Separation of Concerns**:
//! - `ElasticPINN2D<B>`: Neural network architecture (Burn tensors, autodiff)
//! - `ElasticPINN2DSolver`: Physics specification (ndarray, domain traits)
//!
//! This separation allows:
//! 1. Neural network to remain backend-agnostic (CPU/GPU, different autodiff engines)
//! 2. Physics traits to remain solver-agnostic (FD/FEM/PINN/analytical)
//! 3. Tensor conversions to be isolated in one place
//!
//! **Tensor Bridge**:
//! The PINN operates on Burn tensors for autodiff, but physics traits expect ndarray.
//! This module handles conversions between representations:
//! - Forward: ndarray → Burn tensor (for inference through trained network)
//! - Backward: Burn tensor → ndarray (for trait method returns)
//!
//! # Usage
//!
//! ```rust,ignore
//! use kwavers::solver::inverse::pinn::elastic_2d::{Config, ElasticPINN2D, ElasticPINN2DSolver};
//! use kwavers::domain::physics::{Domain, ElasticWaveEquation};
//!
//! // Define physics domain
//! let domain = Domain::new_2d(0.0, 1.0, 0.0, 1.0, 101, 101, BoundaryCondition::Absorbing { damping: 0.1 });
//!
//! // Material properties
//! let lambda = 1e9;  // Pa
//! let mu = 0.5e9;    // Pa
//! let rho = 1000.0;  // kg/m³
//!
//! // Create PINN model
//! let config = Config::forward_problem(lambda, mu, rho);
//! let device = Default::default();
//! let pinn = ElasticPINN2D::new(&config, &device)?;
//!
//! // Wrap in solver that implements physics traits
//! let solver = ElasticPINN2DSolver::new(pinn, domain, lambda, mu, rho);
//!
//! // Now can use as ElasticWaveEquation
//! let cp = solver.p_wave_speed();
//! let cs = solver.s_wave_speed();
//! let cfl_dt = solver.cfl_timestep();
//!
//! // Can validate against other solvers via shared trait
//! fn validate_solver<S: ElasticWaveEquation>(solver: &S) {
//!     // Common validation logic...
//! }
//! validate_solver(&solver);
//! ```

// Re-export submodules
pub mod solver;
pub mod traits;

// Re-export main types for convenience
#[cfg(feature = "pinn")]
pub use solver::ElasticPINN2DSolver;
