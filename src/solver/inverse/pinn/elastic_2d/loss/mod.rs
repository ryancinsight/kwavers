//! Loss Function Computations for Elastic 2D PINN
//!
//! This module implements the loss functions used to train the Physics-Informed
//! Neural Network for 2D elastic wave equations.
//!
//! # Mathematical Foundation
//!
//! The total loss is a weighted sum of multiple components:
//!
//! L_total = w_pde * L_pde + w_bc * L_bc + w_ic * L_ic + w_data * L_data
//!
//! where:
//! - L_pde: PDE residual loss (physics constraint)
//! - L_bc: Boundary condition loss
//! - L_ic: Initial condition loss
//! - L_data: Data fitting loss (for inverse problems)
//!
//! # PDE Residual
//!
//! For elastic wave equation in 2D:
//! ρ ∂²u/∂t² = ∇·σ + f
//!
//! The residual is:
//! R = ρ ∂²u/∂t² - (∂σ_xx/∂x + ∂σ_xy/∂y) - f_x
//! R = ρ ∂²u/∂t² - (∂σ_xy/∂x + ∂σ_yy/∂y) - f_y
//!
//! Loss: L_pde = (1/N) Σ |R|²

// Re-export submodules
pub mod computation;
pub mod data;
pub mod pde_residual;

// ============================================================================
// EXPLICIT RE-EXPORTS (PINN Loss API)
// ============================================================================

/// Loss computation engine
#[cfg(feature = "pinn")]
pub use computation::LossComputer;

// Data structures for loss computation
// TODO: Implement missing data types: BoundaryData, CollocationData, InitialData, LossComponents, ObservationData
// pub use data::{BoundaryData, CollocationData, InitialData, LossComponents, ObservationData};
