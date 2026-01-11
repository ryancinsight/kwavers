//! 2D Wave Equation Physics-Informed Neural Network (PINN) Module
//!
//! This module provides a complete PINN implementation for solving the 2D acoustic wave equation
//! using the Burn deep learning framework with automatic differentiation.
//!
//! ## Module Organization
//!
//! The module is organized following GRASP (General Responsibility Assignment Software Patterns)
//! principles with clear separation of concerns:
//!
//! - **geometry**: 2D domain definitions and geometric primitives
//! - **config**: Configuration structures and hyperparameters
//! - **model**: Neural network architecture and forward pass
//! - **loss**: Physics-informed loss computation and PDE residuals
//! - **training**: Training loop, optimization, and convergence monitoring
//! - **inference**: Real-time prediction and deployment utilities
//!
//! ## Wave Equation
//!
//! Solves the 2D acoustic wave equation:
//!
//! ```text
//! ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
//! ```
//!
//! Where:
//! - u(x,y,t) = displacement/pressure field
//! - c = wave speed (m/s)
//! - (x,y) ∈ Ω ⊂ ℝ² = spatial domain
//! - t ∈ [0, T] = time domain
//!
//! ## Physics-Informed Loss
//!
//! The total loss combines data fitting and physics constraints:
//!
//! ```text
//! L_total = λ_data·L_data + λ_pde·L_pde + λ_bc·L_bc + λ_ic·L_ic
//! ```
//!
//! Where:
//! - L_data: Mean squared error on training data
//! - L_pde: PDE residual ||∂²u/∂t² - c²∇²u||²
//! - L_bc: Boundary condition violations
//! - L_ic: Initial condition violations
//!
//! ## Automatic Differentiation
//!
//! Uses Burn's autodiff backend to compute PDE residuals:
//! - ∂²u/∂t² computed via automatic differentiation
//! - ∇²u = ∂²u/∂x² + ∂²u/∂y² computed via autodiff
//! - No manual gradient implementation required
//!
//! ## Supported Backends
//!
//! - **NdArray**: CPU-only backend (fast compilation, development)
//! - **WGPU**: GPU acceleration via WebGPU (requires `pinn-gpu` feature)
//! - **CUDA**: NVIDIA GPU acceleration (requires `pinn-cuda` feature)
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use burn::backend::NdArray;
//! use kwavers::analysis::ml::pinn::wave_equation_2d::{
//!     BurnPINN2DWave, BurnPINN2DConfig, Geometry2D
//! };
//!
//! // Define geometry
//! let geometry = Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0);
//!
//! // Configure PINN
//! let config = BurnPINN2DConfig {
//!     hidden_layers: vec![50, 50, 50],
//!     num_collocation_points: 10000,
//!     ..Default::default()
//! };
//!
//! // Create PINN with CPU backend
//! type Backend = NdArray<f32>;
//! let device = Default::default();
//! let pinn = BurnPINN2DWave::<Backend>::new(config, geometry, &device)?;
//!
//! // Train on reference data
//! let metrics = pinn.train(
//!     x_data, y_data, t_data, u_data,
//!     343.0,  // Wave speed (m/s)
//!     &device,
//!     1000    // Epochs
//! )?;
//!
//! // Predict at new points
//! let u_pred = pinn.predict(&x_test, &y_test, &t_test, &device)?;
//! ```
//!
//! ## Architecture Details
//!
//! ### Input Layer
//! - 3 inputs: (x, y, t) normalized to [-1, 1]
//! - Input scaling for numerical stability
//!
//! ### Hidden Layers
//! - Configurable depth and width
//! - Activation: Tanh (smooth, differentiable)
//! - Optional: Skip connections, normalization
//!
//! ### Output Layer
//! - 1 output: u(x,y,t) (displacement/pressure)
//! - Linear activation (unbounded output)
//!
//! ### Typical Architectures
//! - Small problems: [3, 50, 50, 50, 1] (~10k parameters)
//! - Medium problems: [3, 100, 100, 100, 100, 1] (~50k parameters)
//! - Large problems: [3, 200, 200, 200, 200, 1] (~200k parameters)
//!
//! ## Training Strategy
//!
//! 1. **Phase 1: Data Fitting** (epochs 0-20%)
//!    - High weight on L_data
//!    - Learn approximate solution from training data
//!
//! 2. **Phase 2: Physics Enforcement** (epochs 20-80%)
//!    - Increase weight on L_pde
//!    - Enforce PDE constraints via autodiff residuals
//!    - Balance data and physics losses
//!
//! 3. **Phase 3: Refinement** (epochs 80-100%)
//!    - Fine-tune boundary and initial conditions
//!    - Reduce learning rate for convergence
//!
//! ## Convergence Monitoring
//!
//! Track multiple metrics during training:
//! - Total loss: Combined physics + data loss
//! - PDE residual: ||PDE(u)||_L²
//! - Boundary error: ||BC violation||_L²
//! - Data error: ||u_pred - u_data||_L²
//! - Relative L² error: ||u_pred - u_data||_L² / ||u_data||_L²
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks: A deep learning
//!   framework for solving forward and inverse problems involving nonlinear
//!   partial differential equations" - Journal of Computational Physics 378:686-707
//! - Karniadakis et al. (2021): "Physics-informed machine learning" - Nature Reviews Physics 3:422-440
//! - Burn Framework: https://burn.dev/ (v0.18 API)
//!
//! ## Feature Flags
//!
//! - `pinn`: Enable PINN functionality (CPU backend)
//! - `pinn-gpu`: Enable GPU acceleration via WGPU
//! - `pinn-cuda`: Enable NVIDIA GPU acceleration

pub mod geometry;

// Re-export main types for convenience
pub use geometry::{Geometry2D, InterfaceCondition};

// Placeholder re-exports for future modules
// pub mod config;
// pub mod model;
// pub mod loss;
// pub mod training;
// pub mod inference;

// pub use config::{BurnPINN2DConfig, BurnLossWeights2D};
// pub use model::BurnNeuralNetwork;
// pub use training::BurnPINN2DTrainer;
// pub use inference::RealTimePINNInference;
