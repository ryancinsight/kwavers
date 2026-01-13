//! Data structures for PINN loss computation
//!
//! This module defines the data structures used for training Physics-Informed
//! Neural Networks, including collocation points, boundary conditions, and
//! observation data.

#[cfg(feature = "pinn")]
use burn::tensor::{backend::AutodiffBackend, Tensor};

// ============================================================================
// Data Structures
// ============================================================================

/// Collocation points for PDE residual computation
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct CollocationData<B: AutodiffBackend> {
    /// Spatial coordinates x [N, 1]
    pub x: Tensor<B, 2>,
    /// Spatial coordinates y [N, 1]
    pub y: Tensor<B, 2>,
    /// Time coordinates [N, 1]
    pub t: Tensor<B, 2>,
    /// Source term f_x (optional) [N, 1]
    pub source_x: Option<Tensor<B, 2>>,
    /// Source term f_y (optional) [N, 1]
    pub source_y: Option<Tensor<B, 2>>,
}

/// Boundary condition data
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct BoundaryData<B: AutodiffBackend> {
    /// Boundary points x [N, 1]
    pub x: Tensor<B, 2>,
    /// Boundary points y [N, 1]
    pub y: Tensor<B, 2>,
    /// Time coordinates [N, 1]
    pub t: Tensor<B, 2>,
    /// Boundary type for each point
    pub boundary_type: Vec<BoundaryType>,
    /// Target values (displacement or traction) [N, 2]
    pub values: Tensor<B, 2>,
}

/// Initial condition data
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct InitialData<B: AutodiffBackend> {
    /// Initial points x [N, 1]
    pub x: Tensor<B, 2>,
    /// Initial points y [N, 1]
    pub y: Tensor<B, 2>,
    /// Initial displacement [N, 2]
    pub displacement: Tensor<B, 2>,
    /// Initial velocity [N, 2]
    pub velocity: Tensor<B, 2>,
}

/// Observation data for inverse problems
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct ObservationData<B: AutodiffBackend> {
    /// Observation locations x [N, 1]
    pub x: Tensor<B, 2>,
    /// Observation locations y [N, 1]
    pub y: Tensor<B, 2>,
    /// Observation times [N, 1]
    pub t: Tensor<B, 2>,
    /// Observed displacement [N, 2]
    pub displacement: Tensor<B, 2>,
}

/// Boundary condition type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// Dirichlet: prescribed displacement
    Dirichlet,
    /// Neumann: prescribed traction (stress)
    Neumann,
    /// Free surface (stress-free)
    FreeSurface,
}

/// Individual loss components
#[cfg(feature = "pinn")]
#[derive(Debug, Clone)]
pub struct LossComponents {
    /// PDE residual loss
    pub pde: f64,
    /// Boundary condition loss
    pub boundary: f64,
    /// Initial condition loss
    pub initial: f64,
    /// Data fitting loss
    pub data: f64,
    /// Total weighted loss
    pub total: f64,
}

#[cfg(all(test, feature = "pinn"))]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_type() {
        assert_eq!(BoundaryType::Dirichlet, BoundaryType::Dirichlet);
        assert_ne!(BoundaryType::Dirichlet, BoundaryType::Neumann);
    }

    #[test]
    fn test_loss_components() {
        let components = LossComponents {
            pde: 1.0,
            boundary: 0.5,
            initial: 0.3,
            data: 0.2,
            total: 2.0,
        };
        assert_eq!(components.pde, 1.0);
        assert_eq!(components.total, 2.0);
    }
}
