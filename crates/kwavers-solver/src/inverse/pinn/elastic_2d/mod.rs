//! 2D Elastic Wave PINN Solver
//!
//! This module implements a Physics-Informed Neural Network (PINN) for solving
//! 2D elastic wave equations. It provides both forward problem solving (given
//! material properties, compute wave propagation) and inverse problem solving
//! (given observed waveforms, estimate material properties).
//!
//! # Mathematical Formulation
//!
//! ## Governing Equations
//!
//! The 2D elastic wave equation in an isotropic medium:
//!
//! ```text
//! ρ ∂²u/∂t² = ∇·σ + f
//! ```
//!
//! where:
//! - u = (uₓ, uᵧ): displacement vector (m)
//! - ρ(x,y): density [kg/m³]
//! - σ: Cauchy stress tensor (Pa)
//! - f = (fₓ, fᵧ): body force [N/m³]
//!
//! ## Constitutive Relation (Hooke's Law)
//!
//! For isotropic elastic media:
//!
//! ```text
//! σ = λ·tr(ε)·I + 2μ·ε
//! ```
//!
//! where:
//! - λ(x,y): Lamé's first parameter (Pa)
//! - μ(x,y): Lamé's second parameter (shear modulus) (Pa)
//! - ε: strain tensor = ½(∇u + (∇u)ᵀ)
//! - I: identity tensor
//!
//! ## Expanded Form
//!
//! In component form (2D):
//!
//! ```text
//! ρ ∂²uₓ/∂t² = ∂σₓₓ/∂x + ∂σₓᵧ/∂y + fₓ
//! ρ ∂²uᵧ/∂t² = ∂σₓᵧ/∂x + ∂σᵧᵧ/∂y + fᵧ
//! ```
//!
//! where:
//!
//! ```text
//! σₓₓ = (λ + 2μ)·∂uₓ/∂x + λ·∂uᵧ/∂y
//! σᵧᵧ = λ·∂uₓ/∂x + (λ + 2μ)·∂uᵧ/∂y
//! σₓᵧ = μ·(∂uₓ/∂y + ∂uᵧ/∂x)
//! ```
//!
//! # PINN Architecture
//!
//! ## Network Structure
//!
//! The PINN approximates the displacement field as:
//!
//! ```text
//! u(x, y, t) ≈ u_θ(x, y, t)
//! ```
//!
//! where u_θ is a deep neural network with parameters θ, typically:
//! - Input: (x, y, t) ∈ ℝ³
//! - Hidden layers: 4-8 layers with 50-200 neurons each
//! - Activation: tanh, sin, or adaptive activation functions
//! - Output: (uₓ, uᵧ) ∈ ℝ²
//!
//! ## Loss Function
//!
//! The total loss is a weighted sum:
//!
//! ```text
//! L_total = λ_pde · L_pde + λ_bc · L_bc + λ_ic · L_ic + λ_data · L_data
//! ```
//!
//! ### PDE Residual Loss
//!
//! At interior collocation points {(xᵢ, yᵢ, tᵢ)}, enforce the PDE:
//!
//! ```text
//! L_pde = (1/N_interior) Σᵢ [ ||ρ ∂²uₓ/∂t² - ∂σₓₓ/∂x - ∂σₓᵧ/∂y - fₓ||²
//!                              + ||ρ ∂²uᵧ/∂t² - ∂σₓᵧ/∂x - ∂σᵧᵧ/∂y - fᵧ||² ]
//! ```
//!
//! All derivatives are computed via automatic differentiation through the network.
//!
//! ### Boundary Condition Loss
//!
//! At boundary points {(xᵦ, yᵦ, tᵦ)} ∈ ∂Ω:
//!
//! ```text
//! L_bc = (1/N_boundary) Σᵦ ||BC[u_θ](xᵦ, yᵦ, tᵦ) - g(xᵦ, yᵦ, tᵦ)||²
//! ```
//!
//! Supports:
//! - Dirichlet: u = g (prescribed displacement)
//! - Neumann: σ·n = g (prescribed traction)
//! - Free surface: σ·n = 0
//! - Absorbing: PML or damping conditions
//!
//! ### Initial Condition Loss
//!
//! At t=0:
//!
//! ```text
//! L_ic = (1/N_ic) Σ [ ||u_θ(x, y, 0) - u₀(x, y)||²
//!                    + ||∂u_θ/∂t(x, y, 0) - v₀(x, y)||² ]
//! ```
//!
//! ### Data Fitting Loss (Inverse Problems)
//!
//! At observation points {(x_obs, y_obs, t_obs)}:
//!
//! ```text
//! L_data = (1/N_obs) Σ ||u_θ(x_obs, y_obs, t_obs) - u_obs||²
//! ```
//!
//! # Usage
//!
//! ## Forward Problem (Wave Propagation Simulation)
//!
//! ```rust,ignore
//! use kwavers_solver::inverse::pinn::elastic_2d::{Config, ElasticPINN2D};
//! use kwavers_domain::geometry::RectangularDomain;
//!
//! // Define domain and material properties
//! let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
//! let lambda = 1e9;  // Pa
//! let mu = 0.5e9;    // Pa
//! let rho = 1000.0;  // kg/m³
//!
//! // Configure PINN
//! let config = Config {
//!     hidden_layers: 6,
//!     neurons_per_layer: 100,
//!     n_interior_points: 10000,
//!     n_boundary_points: 1000,
//!     learning_rate: 1e-3,
//!     n_epochs: 5000,
//!     ..Default::default()
//! };
//!
//! // Create and train PINN
//! let mut pinn = ElasticPINN2D::new(domain, lambda, mu, rho, config);
//! pinn.train();
//!
//! // Evaluate solution
//! let u = pinn.predict(&[0.5, 0.5, 0.1]);
//! println!("Displacement at (0.5, 0.5, t=0.1s): {:?}", u);
//! ```
//!
//! ## Inverse Problem (Material Parameter Estimation)
//!
//! ```rust,ignore
//! use kwavers_solver::inverse::pinn::elastic_2d::{Config, ElasticPINN2D};
//!
//! // Load observed displacement data
//! let observations = load_displacement_measurements();
//!
//! // Configure PINN for inverse problem
//! let config = Config {
//!     optimize_lambda: true,
//!     optimize_mu: true,
//!     data_loss_weight: 10.0,  // Strong data fitting
//!     ..Default::default()
//! };
//!
//! // Create PINN with unknown material properties (initial guess)
//! let mut pinn = ElasticPINN2D::new_inverse(domain, observations, config);
//!
//! // Train: jointly optimize network weights and material properties
//! pinn.train_inverse();
//!
//! // Extract estimated parameters
//! let lambda_est = pinn.estimated_lambda();
//! let mu_est = pinn.estimated_mu();
//! println!("Estimated λ = {:.2e} Pa, μ = {:.2e} Pa", lambda_est, mu_est);
//! ```
//!
//! # Module Structure
//!
//! - `pinn::geometry`: Collocation point sampling, interface conditions, refinement
//! - `config`: Hyperparameters and training configuration (to be implemented)
//! - `model`: Neural network architecture (to be implemented)
//! - `loss`: Physics-informed loss functions (to be implemented)
//! - `training`: Training loop and optimizer (to be implemented)
//! - `inference`: Trained model deployment and evaluation (to be implemented)
//!
//! # References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks" - JCP 378:686-707
//! - Haghighat et al. (2021): "A physics-informed deep learning framework for
//!   inversion and surrogate modeling in solid mechanics" - CMAME 379:113741
//! - Rao et al. (2021): "Physics-informed deep learning for computational
//!   elastodynamics without labeled data" - JEM 132:103448

pub mod adaptive_sampling;
pub mod config;
pub mod inference;
pub mod loss;
pub mod model;
pub mod physics_impl;
pub mod training;

#[cfg(all(test, feature = "pinn"))]
mod tests;

// Re-export key types
#[cfg(feature = "pinn")]
pub use adaptive_sampling::{AdaptiveSampler, BatchIterator, ElasticAdaptiveSamplingStrategy};

pub use super::geometry::{
    AdaptiveRefinement, CollocationSampler, CollocationSamplingStrategy, MultiRegionDomain,
    PinnGeometryInterfaceCondition,
};
pub use config::{
    Config, ElasticCollocationSamplingStrategy, ElasticPinnActivationFunction,
    ElasticPinnLrScheduler, ElasticPinnOptimizerType, LossWeights,
};

#[cfg(feature = "pinn")]
pub use inference::ElasticPinnPredictor;

#[cfg(feature = "pinn")]
pub use loss::{
    BoundaryData, BoundaryType, CollocationData, ElasticPinnLossComponents, InitialData,
    LossComputer, ObservationData,
};

pub use model::ElasticPINN2D;

#[cfg(feature = "pinn")]
pub use training::{ElasticPinnTrainingMetrics, TrainingData};

#[cfg(feature = "pinn")]
pub use physics_impl::ElasticPINN2DSolver;
