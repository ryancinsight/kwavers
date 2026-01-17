//! Burn-based 3D Wave Equation Physics-Informed Neural Network with Automatic Differentiation
//!
//! This module implements a PINN for the 3D acoustic wave equation using the Burn deep learning
//! framework with native automatic differentiation. This extends the 2D implementation to handle
//! three spatial dimensions with complex geometries and boundary conditions.
//!
//! ## Wave Equation
//!
//! Solves: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
//!
//! Where:
//! - u(x,y,z,t) = displacement/pressure field
//! - c(x,y,z) = spatially varying wave speed (m/s)
//! - x,y,z = spatial coordinates (m)
//! - t = time coordinate (s)
//!
//! ## Architecture Overview
//!
//! This module follows Clean Architecture principles with clear layer separation:
//!
//! ### Domain Layer
//! - **types**: Boundary and interface condition types
//! - **geometry**: 3D geometry definitions (rectangular, spherical, cylindrical)
//! - **config**: Configuration structures for training and loss weights
//!
//! ### Infrastructure Layer
//! - **network**: Neural network architecture with PDE residual computation
//! - **wavespeed**: Wave speed function wrapper with Module trait support
//! - **optimizer**: Simple SGD optimizer for parameter updates
//!
//! ### Application Layer
//! - **solver**: High-level PINN solver orchestration (training, prediction)
//!
//! ## Physics-Informed Loss
//!
//! L_total = λ_data × L_data + λ_pde × L_pde + λ_bc × L_bc + λ_ic × L_ic
//!
//! Where:
//! - L_data: MSE between predictions and training data
//! - L_pde: MSE of PDE residual (computed via finite differences)
//! - L_bc: MSE of boundary condition violations
//! - L_ic: MSE of initial condition violations
//!
//! ## Backends
//!
//! This implementation supports multiple Burn backends:
//!
//! - **NdArray**: CPU-only backend (fast compilation, good for development)
//! - **WGPU**: GPU acceleration via WebGPU (requires `pinn-gpu` feature)
//!
//! ## 3D Geometry Support
//!
//! - **Rectangular domains**: Standard 3D rectangular boxes
//! - **Spherical domains**: Sphere-shaped regions with radial boundaries
//! - **Cylindrical domains**: Cylindrical regions with axial symmetry
//! - **Multi-region domains**: Complex geometries with interface conditions
//!
//! ## Boundary Conditions
//!
//! - **Dirichlet**: u = 0 on boundaries (sound-hard)
//! - **Neumann**: ∂u/∂n = 0 on boundaries (sound-soft)
//! - **Absorbing**: Radiation boundary conditions
//! - **Periodic**: For infinite domains
//!
//! ## Heterogeneous Media Support
//!
//! - **Spatially varying wave speed**: c(x,y,z) functions
//! - **Multi-region domains**: Different materials in different regions
//! - **Interface conditions**: Continuity of pressure and normal velocity
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks" - JCP 378:686-707
//! - Burn Framework: https://burn.dev/ (v0.18 API)
//!
//! ## Examples
//!
//! ### Basic Usage with CPU Backend
//!
//! ```rust,ignore
//! use burn::backend::NdArray;
//! use kwavers::analysis::ml::pinn::burn_wave_equation_3d::{
//!     BurnPINN3DWave, BurnPINN3DConfig, Geometry3D
//! };
//! use kwavers::core::error::KwaversResult;
//!
//! // Create PINN with NdArray backend (CPU)
//! type Backend = NdArray<f32>;
//! let device = Default::default();
//! let config = BurnPINN3DConfig::default();
//! let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); // Unit cube
//! let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0; // Constant speed
//!
//! fn run() -> KwaversResult<()> {
//!     let mut pinn = BurnPINN3DWave::<Backend>::new(config, geometry, wave_speed, &device)?;
//!
//! // Train on reference data
//! let x_data = vec![0.5, 0.6, 0.7];
//! let y_data = vec![0.5, 0.5, 0.5];
//! let z_data = vec![0.5, 0.5, 0.5];
//! let t_data = vec![0.1, 0.2, 0.3];
//! let u_data = vec![0.0, 0.1, 0.0];
//!
//! let metrics = pinn.train(
//!     &x_data, &y_data, &z_data, &t_data, &u_data,
//!     &device, 1000
//! )?;
//!
//! if let Some(final_loss) = metrics.total_loss.last() {
//!     println!("Final loss: {:.6e}", final_loss);
//! }
//!
//! // Predict at new points
//! let x_test = vec![0.5, 0.6];
//! let y_test = vec![0.5, 0.5];
//! let z_test = vec![0.5, 0.5];
//! let t_test = vec![0.5, 0.6];
//! let u_pred = pinn.predict(&x_test, &y_test, &z_test, &t_test, &device)?;
//!     Ok(())
//! }
//! ```
//!
//! ### Heterogeneous Media (Layered)
//!
//! ```rust,ignore
//! use burn::backend::NdArray;
//! use kwavers::analysis::ml::pinn::burn_wave_equation_3d::{
//!     BurnPINN3DWave, BurnPINN3DConfig, Geometry3D
//! };
//!
//! // Define spatially varying wave speed (e.g., layered medium)
//! let wave_speed = |_x: f32, _y: f32, z: f32| {
//!     if z < 0.5 {
//!         1500.0 // Water layer
//!     } else {
//!         3000.0 // Tissue layer
//!     }
//! };
//!
//! type Backend = NdArray<f32>;
//! let device = Default::default();
//! let config = BurnPINN3DConfig::default();
//! let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
//!
//! let pinn = BurnPINN3DWave::<Backend>::new(config, geometry, wave_speed, &device)?;
//!
//! // Verify wave speeds
//! assert_eq!(pinn.get_wave_speed(0.5, 0.5, 0.3)?, 1500.0);
//! assert_eq!(pinn.get_wave_speed(0.5, 0.5, 0.7)?, 3000.0);
//! ```
//!
//! ### Complex Geometry (Spherical)
//!
//! ```rust,ignore
//! use burn::backend::NdArray;
//! use kwavers::analysis::ml::pinn::burn_wave_equation_3d::{
//!     BurnPINN3DWave, BurnPINN3DConfig, Geometry3D
//! };
//!
//! // Create spherical geometry
//! let geometry = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3); // Center + radius
//! let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;
//!
//! type Backend = NdArray<f32>;
//! let device = Default::default();
//! let config = BurnPINN3DConfig {
//!     hidden_layers: vec![128, 128, 128],
//!     num_collocation_points: 1000,
//!     ..Default::default()
//! };
//!
//! let pinn = BurnPINN3DWave::<Backend>::new(config, geometry, wave_speed, &device)?;
//! ```
//!
//! ## Feature Flags
//!
//! - `pinn`: Basic PINN functionality with CPU backend
//! - `pinn-gpu`: Adds GPU acceleration via WGPU backend

// Domain layer
pub mod config;
pub mod geometry;
pub mod types;

// Infrastructure layer
pub mod network;
pub mod optimizer;
pub mod wavespeed;

// Application layer
pub mod solver;

// Integration tests
#[cfg(test)]
mod tests;

// Public API exports
pub use config::{BurnLossWeights3D, BurnPINN3DConfig, BurnTrainingMetrics3D};
pub use geometry::Geometry3D;
pub use network::PINN3DNetwork;
pub use optimizer::SimpleOptimizer3D;
pub use solver::BurnPINN3DWave;
pub use types::{BoundaryCondition3D, InterfaceCondition3D};
pub use wavespeed::WaveSpeedFn3D;
