//! Physics task and task data structures for meta-learning.

use super::pde_type::PdeType;
use super::physics::MetaLearningPhysicsParameters;
use std::sync::Arc;

/// Physics task definition for meta-learning
///
/// Represents a single task instance in the meta-learning framework.
/// Each task corresponds to solving a specific PDE with given parameters,
/// geometry, and boundary/initial conditions.
///
/// # Literature
///
/// - Grant, E., et al. (2018). "Recasting gradient-based meta-learning as hierarchical Bayes"
///   *ICLR 2018*
#[derive(Debug, Clone)]
pub struct PhysicsTask {
    /// Unique task identifier
    ///
    /// Format suggestion: "{pde_type}_{geometry}_{id}"
    /// Example: "wave_rectangular_001"
    pub id: String,

    /// Type of PDE governing the physics
    pub pde_type: PdeType,

    /// Physics parameters (wave speed, density, etc.)
    pub physics_params: MetaLearningPhysicsParameters,

    /// Geometric domain specification
    ///
    /// Wrapped in Arc for efficient cloning across threads.
    pub geometry: Arc<crate::inverse::pinn::ml::BurnWave2dGeometry>,

    /// Boundary conditions
    pub boundary_conditions: Vec<crate::inverse::pinn::ml::BoundaryCondition2D>,

    /// Training data (optional for few-shot learning)
    pub training_data: Option<TaskData>,

    /// Validation data for meta-training
    pub validation_data: TaskData,
}

impl PhysicsTask {
    /// Create a new physics task
    pub fn new(
        id: String,
        pde_type: PdeType,
        physics_params: MetaLearningPhysicsParameters,
        geometry: Arc<crate::inverse::pinn::ml::BurnWave2dGeometry>,
        boundary_conditions: Vec<crate::inverse::pinn::ml::BoundaryCondition2D>,
        training_data: Option<TaskData>,
        validation_data: TaskData,
    ) -> Self {
        Self {
            id,
            pde_type,
            physics_params,
            geometry,
            boundary_conditions,
            training_data,
            validation_data,
        }
    }

    /// Get the computational complexity of this task
    pub fn complexity(&self) -> f64 {
        let pde_complexity = self.pde_type.complexity();

        let geometry_complexity = match self.geometry.as_ref() {
            crate::inverse::pinn::ml::BurnWave2dGeometry::Rectangular { .. } => 0.2,
            crate::inverse::pinn::ml::BurnWave2dGeometry::Circular { .. } => 0.4,
            crate::inverse::pinn::ml::BurnWave2dGeometry::MultiRegion { .. } => 1.0,
            _ => 0.6,
        };

        let bc_complexity = (self.boundary_conditions.len() as f64).min(10.0) / 10.0;

        0.5 * pde_complexity + 0.3 * geometry_complexity + 0.2 * bc_complexity
    }

    /// Check if task has sufficient data for training
    pub fn is_valid(&self) -> bool {
        !self.validation_data.collocation_points.is_empty()
    }
}

/// Task data for training/validation
///
/// Contains discretized spatial-temporal points for:
/// - Physics constraint evaluation (collocation points)
/// - Boundary condition enforcement (boundary data)
/// - Initial condition enforcement (initial data)
///
/// # Literature
///
/// - McKay, M. D., et al. (1979). "A comparison of three methods for selecting values of input variables"
/// - Wang, S., et al. (2020). "When and why PINNs fail to train"
#[derive(Debug, Clone, Default)]
pub struct TaskData {
    /// Collocation points (x, y, t) for PDE residual evaluation
    pub collocation_points: Vec<(f64, f64, f64)>,

    /// Boundary data (x, y, t, u) for boundary condition enforcement
    pub boundary_data: Vec<(f64, f64, f64, f64)>,

    /// Initial data (x, y, t=0, u, ∂u/∂t) for initial condition enforcement
    pub initial_data: Vec<(f64, f64, f64, f64, f64)>,
}

impl TaskData {
    /// Create empty task data
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create task data with specified capacities (for efficient allocation)
    pub fn with_capacity(
        collocation_capacity: usize,
        boundary_capacity: usize,
        initial_capacity: usize,
    ) -> Self {
        Self {
            collocation_points: Vec::with_capacity(collocation_capacity),
            boundary_data: Vec::with_capacity(boundary_capacity),
            initial_data: Vec::with_capacity(initial_capacity),
        }
    }

    /// Get total number of data points
    pub fn total_points(&self) -> usize {
        self.collocation_points.len() + self.boundary_data.len() + self.initial_data.len()
    }

    /// Check if task data is empty
    pub fn is_empty(&self) -> bool {
        self.total_points() == 0
    }

    /// Get statistics about point distribution
    pub fn statistics(&self) -> TaskDataStatistics {
        TaskDataStatistics {
            num_collocation: self.collocation_points.len(),
            num_boundary: self.boundary_data.len(),
            num_initial: self.initial_data.len(),
            total: self.total_points(),
        }
    }
}

/// Statistics about task data distribution
#[derive(Debug, Clone, Copy)]
pub struct TaskDataStatistics {
    pub num_collocation: usize,
    pub num_boundary: usize,
    pub num_initial: usize,
    pub total: usize,
}
