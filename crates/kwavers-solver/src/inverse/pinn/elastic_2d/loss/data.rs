//! Data structures for PINN loss computation
//!
//! This module defines the data structures used for training Physics-Informed
//! Neural Networks, including collocation points, boundary conditions, and
//! observation data.

use coeus_autograd::Var;

// ============================================================================
// Data Structures
// ============================================================================

/// Collocation points for PDE residual computation
#[derive(Clone)]
pub struct CollocationData<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Spatial coordinates x [N, 1]
    pub x: Var<f32, B>,
    /// Spatial coordinates y [N, 1]
    pub y: Var<f32, B>,
    /// Time coordinates [N, 1]
    pub t: Var<f32, B>,
    /// Source term f_x (optional) [N, 1]
    pub source_x: Option<Var<f32, B>>,
    /// Source term f_y (optional) [N, 1]
    pub source_y: Option<Var<f32, B>>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for CollocationData<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollocationData")
            .field("has_source_x", &self.source_x.is_some())
            .field("has_source_y", &self.source_y.is_some())
            .finish_non_exhaustive()
    }
}

/// Boundary condition data
#[derive(Clone)]
pub struct BoundaryData<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Boundary points x [N, 1]
    pub x: Var<f32, B>,
    /// Boundary points y [N, 1]
    pub y: Var<f32, B>,
    /// Time coordinates [N, 1]
    pub t: Var<f32, B>,
    /// Boundary type for each point
    pub boundary_type: Vec<ElasticBoundaryCondition>,
    /// Target values (displacement or traction) [N, 2]
    pub values: Var<f32, B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for BoundaryData<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundaryData")
            .field("boundary_type", &self.boundary_type)
            .finish_non_exhaustive()
    }
}

/// Initial condition data
#[derive(Clone)]
pub struct InitialData<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Initial points x [N, 1]
    pub x: Var<f32, B>,
    /// Initial points y [N, 1]
    pub y: Var<f32, B>,
    /// Initial displacement [N, 2]
    pub displacement: Var<f32, B>,
    /// Initial velocity [N, 2]
    pub velocity: Var<f32, B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for InitialData<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InitialData").finish_non_exhaustive()
    }
}

/// Observation data for inverse problems
#[derive(Clone)]
pub struct ObservationData<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Observation locations x [N, 1]
    pub x: Var<f32, B>,
    /// Observation locations y [N, 1]
    pub y: Var<f32, B>,
    /// Observation times [N, 1]
    pub t: Var<f32, B>,
    /// Observed displacement [N, 2]
    pub displacement: Var<f32, B>,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for ObservationData<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservationData").finish_non_exhaustive()
    }
}

/// Boundary condition type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticBoundaryCondition {
    /// Dirichlet: prescribed displacement
    Dirichlet,
    /// Neumann: prescribed traction (stress)
    Neumann,
    /// Free surface (stress-free)
    FreeSurface,
}

/// Individual loss components
#[derive(Debug, Clone)]
pub struct ElasticPinnLossComponents {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_type() {
        assert_eq!(
            ElasticBoundaryCondition::Dirichlet,
            ElasticBoundaryCondition::Dirichlet
        );
        assert_ne!(
            ElasticBoundaryCondition::Dirichlet,
            ElasticBoundaryCondition::Neumann
        );
    }

    #[test]
    fn test_loss_components() {
        let components = ElasticPinnLossComponents {
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
