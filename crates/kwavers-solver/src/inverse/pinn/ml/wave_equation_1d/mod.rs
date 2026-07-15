//! Coeus-backed 1D Wave Equation Physics-Informed Neural Network
//!
//! This module provides a complete implementation of a Physics-Informed Neural Network (PINN)
//! for solving the 1D acoustic wave equation using the Coeus autodiff stack with
//! automatic differentiation.
//!
//! ## Overview
//!
//! Physics-Informed Neural Networks (PINNs) are a class of neural networks that embed
//! physical laws (expressed as partial differential equations) directly into the loss
//! function. This enables the network to learn solutions that simultaneously fit data
//! and satisfy the governing physics.
//!
//! ## Mathematical Foundation
//!
//! ### 1D Acoustic Wave Equation
//!
//! **Theorem (d'Alembert 1747, Euler 1744)**: The 1D acoustic wave equation describes
//! propagation of pressure/displacement disturbances in compressible fluids:
//!
//! **∂²u/∂t² = c²∂²u/∂x²**
//!
//! Where:
//! - u(x,t): Acoustic pressure or displacement field [Pa or m]
//! - c: Speed of sound in the medium (m/s), e.g., 343 m/s for air at 20°C
//! - x: Spatial coordinate (m)
//! - t: Time coordinate (s)
//!
//! **Derivation**: From conservation of mass (∂ρ/∂t + ∇·(ρv) = 0) and momentum
//! (ρ∂v/∂t + ∇p = 0) with linearization assumptions (small perturbations, p = c²ρ).
//!
//! **Well-Posedness**: Requires:
//! - Initial conditions: u(x,0) = f(x), ∂u/∂t(x,0) = g(x)
//! - Boundary conditions: Dirichlet (u = u₀) or Neumann (∂u/∂n = 0)
//!
//! ### Physics-Informed Loss Function
//!
//! **L_total = λ_data × L_data + λ_pde × L_pde + λ_bc × L_bc**
//!
//! Where:
//! - **L_data**: MSE between predictions and training data (data fidelity)
//! - **L_pde**: MSE of PDE residual (physics constraint)
//! - **L_bc**: MSE of boundary condition violations (boundary enforcement)
//!
//! ## Architecture
//!
//! The implementation follows Clean Architecture principles with clear separation of concerns:
//!
//! ### Domain Layer (Pure Business Logic)
//! - **config**: Configuration types with validation
//! - **types**: Domain types (metrics, convergence analysis)
//!
//! ### Application Layer (Use Cases)
//! - **trainer**: Training orchestration
//! - **physics**: Physics-informed loss computation
//!
//! ### Infrastructure Layer (Framework Integration)
//! - **network**: PINN neural network implementation
//! - **optimizer**: Gradient descent optimization
//!
//! ## Backends
//!
//! This implementation currently supports the Coeus CPU backend:
//!
//! - **NdArray**: CPU-only backend (fast compilation, good for development)
//!
//! PINN GPU execution is reserved for the Coeus training backend routed through
//! the provider-generic Hephaestus GPU traits. Concrete WGPU or CUDA providers
//! belong behind that trait boundary, not in this PINN module.
//!
//! ## Quick Start
//!
//! ### CPU Backend (Default)
//!
//! ```rust,ignore
//! use kwavers_solver::inverse::pinn::ml::wave_equation_1d::{
//!     PinnWave1D, PinnConfig, PinnTrainer
//! };
//! use leto::Array1;
//!
//! // Coeus compute backend (self-instantiates via `Default`; no device handle).
//! type Backend = coeus_core::MoiraiBackend;
//!
//! // Configuration
//! let config = PinnConfig::default();
//!
//! // Create trainer
//! let mut trainer = PinnTrainer::<Backend>::new(config)?;
//!
//! // Generate or load training data
//! let x_data = Array1::linspace(-1.0, 1.0, 100);
//! let t_data = Array1::linspace(0.0, 0.1, 100);
//! let u_data = generate_reference_data(&x_data, &t_data);
//!
//! // Train with physics-informed loss
//! let metrics = trainer.train(
//!     &x_data,
//!     &t_data,
//!     &u_data,
//!     343.0,  // wave speed (m/s)
//!     1000    // epochs
//! )?;
//!
//! // Make predictions
//! let x_test = Array1::linspace(-1.0, 1.0, 200);
//! let t_test = Array1::linspace(0.0, 0.1, 200);
//! let u_pred = trainer.pinn().predict(&x_test, &t_test)?;
//! ```
//!
//! ## Configuration Presets
//!
//! Pre-configured settings for common use cases:
//!
//! ```rust,ignore
//! // Default: Balanced CPU configuration
//! let config = PinnConfig::default();
//!
//! // Large-batch preset for future provider-backed GPU execution
//! let config = PinnConfig::for_gpu();
//!
//! // Prototyping: Fast iteration for development
//! let config = PinnConfig::for_prototyping();
//! ```
//!
//! ## Loss Weighting Strategies
//!
//! Balance data fidelity vs. physics constraints:
//!
//! ```rust,ignore
//! use kwavers_solver::inverse::pinn::ml::wave_equation_1d::LossWeights;
//!
//! // Data-driven: Emphasize fitting observations
//! let weights = LossWeights::data_driven();
//!
//! // Physics-driven: Emphasize PDE satisfaction
//! let weights = LossWeights::physics_driven();
//!
//! // Balanced: Equal weight to all components
//! let weights = LossWeights::balanced();
//! ```
//!
//! ## Training Metrics
//!
//! Monitor convergence and performance:
//!
//! ```rust,ignore
//! let metrics = trainer.train(...)?;
//!
//! // Loss history
//! println!("Final total loss: {:.6e}", metrics.total_loss.last().unwrap());
//! println!("Final data loss: {:.6e}", metrics.data_loss.last().unwrap());
//! println!("Final PDE loss: {:.6e}", metrics.pde_loss.last().unwrap());
//! println!("Final BC loss: {:.6e}", metrics.bc_loss.last().unwrap());
//!
//! // Training time
//! println!("Training time: {:.2}s", metrics.training_time_secs);
//! println!("Throughput: {:.2} epochs/s", metrics.throughput());
//!
//! // Convergence check
//! if metrics.is_converged(100, 1e-6) {
//!     println!("Training converged!");
//! }
//!
//! // Numerical stability
//! if metrics.has_numerical_issues() {
//!     println!("Warning: Numerical instabilities detected");
//! }
//! ```
//!
//! ## References
//!
//! ### Literature
//!
//! 1. **Raissi et al. (2019)**: "Physics-informed neural networks: A deep learning framework
//!    for solving forward and inverse problems involving nonlinear partial differential equations"
//!    Journal of Computational Physics, 378:686-707. DOI: 10.1016/j.jcp.2018.10.045
//!
//! 2. **Karniadakis et al. (2021)**: "Physics-informed machine learning"
//!    Nature Reviews Physics, 3:422-440. DOI: 10.1038/s42254-021-00314-5
//!
//! 3. **Wang et al. (2021)**: "Understanding and mitigating gradient flow pathologies in
//!    physics-informed neural networks" SIAM Journal on Scientific Computing, 43(5).
//!
//! 4. **d'Alembert (1747)**: "Recherches sur la courbe que forme une corde tenduë mise en vibration"
//!    Original derivation of 1D wave equation.
//!
//! 5. **Euler (1744)**: Foundation of wave mechanics from conservation laws.
//!
//! ### Frameworks
//!
//! - **Coeus**: Atlas-native autodiff/NN stack (`coeus_autograd`, `coeus_nn`,
//!   `coeus_optim`) over the `MoiraiBackend` compute backend.
//! - **Rust**: Edition 2021
//!
//! ## Performance Notes
//!
//! ### CPU Backend
//! - Fast compilation (~30s)
//! - Good for development and small problems
//! - Training speed: ~10-100 epochs/s (depends on network size)
//!
//! ### GPU Execution
//! - GPU training is not provided by this PINN module.
//! - Provider-generic GPU execution is owned by the Coeus + Hephaestus migration.
//! - WGPU and CUDA must implement the same provider trait contract before being
//!   exposed to solver-level PINN code.
//!
//! ### Recommendations
//! - Start with CPU backend for prototyping
//! - Switch to Coeus provider-backed training for production GPU workloads once
//!   that integration is present
//! - Use smaller networks (2-3 layers, 20-50 neurons) for CPU
//! - Use larger networks (4-6 layers, 100-200 neurons) for GPU
//!
//! ## Feature Flags
//!
//! - `pinn`: Basic PINN functionality with CPU backend
//! - GPU PINN execution: pending Coeus + Hephaestus provider integration
//!
//! ## Module Organization
//!
//! - **config**: Configuration types and validation
//! - **types**: Training metrics and analysis
//! - **network**: Neural network architecture
//! - **optimizer**: Parameter optimization
//! - **physics**: PDE residual and physics loss
//! - **trainer**: Training orchestration

// Public modules
pub mod config;
pub mod network;
pub mod optimizer;
pub mod physics;
#[cfg(test)]
mod tests;
pub mod trainer;
pub mod types;

// Re-export public API for convenience
pub use config::{LossWeights, PinnConfig};
pub use network::PinnWave1D;
pub use optimizer::SimpleOptimizer;
pub use trainer::PinnTrainer;
pub use types::TrainingMetrics;
