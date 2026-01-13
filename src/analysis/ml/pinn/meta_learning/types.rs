//! Domain Types for Meta-Learning
//!
//! This module defines the core domain types for meta-learning with Physics-Informed
//! Neural Networks, including task definitions, physics parameters, and training data structures.
//!
//! # Task Representation
//!
//! A meta-learning task represents a specific physics problem instance with:
//! - PDE type and parameters
//! - Geometric domain
//! - Boundary and initial conditions
//! - Training and validation data
//!
//! # Literature References
//!
//! 1. Finn, C., Abbeel, P., & Levine, S. (2017).
//!    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
//!    *ICML 2017*
//!
//! 2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
//!    "Physics-informed neural networks: A deep learning framework for solving forward
//!    and inverse problems involving nonlinear partial differential equations"
//!    *Journal of Computational Physics*, 378, 686-707.
//!    DOI: 10.1016/j.jcp.2018.10.045
//!
//! 3. Khodayi-mehr, R., & Zavlanos, M. M. (2019).
//!    "VarNet: Variational Neural Networks for the Solution of Partial Differential Equations"
//!    *arXiv:1912.07443*
//!
//! # Examples
//!
//! ```rust,ignore
//! use kwavers::analysis::ml::pinn::meta_learning::{
//!     PdeType, PhysicsTask, PhysicsParameters, TaskData
//! };
//! use std::sync::Arc;
//!
//! // Define physics parameters for a wave equation task
//! let params = PhysicsParameters {
//!     wave_speed: 343.0,  // Speed of sound in air (m/s)
//!     density: 1.2,        // Air density (kg/m³)
//!     viscosity: None,
//!     absorption: Some(0.01),
//!     nonlinearity: None,
//! };
//!
//! // Create a task
//! let task = PhysicsTask {
//!     id: "wave_rectangular_1".to_string(),
//!     pde_type: PdeType::Wave,
//!     physics_params: params,
//!     geometry: Arc::new(Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0)),
//!     boundary_conditions: vec![],
//!     training_data: None,
//!     validation_data: TaskData::default(),
//! };
//! ```

use std::sync::Arc;

/// Types of partial differential equations supported in meta-learning
///
/// Each PDE type represents a different physics domain with characteristic
/// mathematical structure and computational challenges.
///
/// # Complexity Ordering (for curriculum learning)
///
/// From simplest to most complex:
/// 1. **Wave**: Linear, second-order hyperbolic PDE
/// 2. **Diffusion**: Linear, second-order parabolic PDE
/// 3. **Acoustic**: Linear wave equation with medium heterogeneity
/// 4. **Elastic**: Coupled vector-valued wave equations
/// 5. **Electromagnetic**: Coupled Maxwell's equations (vector calculus)
/// 6. **Navier-Stokes**: Nonlinear, coupled momentum and continuity equations
///
/// # Literature
///
/// - Evans, L. C. (2010). *Partial Differential Equations* (Vol. 19). American Mathematical Society.
/// - Karniadakis, G. E., et al. (2021). "Physics-informed machine learning", *Nature Reviews Physics*, 3(6), 422-440.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PdeType {
    /// Linear wave equation: ∂²u/∂t² - c²∇²u = 0
    ///
    /// Complexity: Low
    /// Applications: Acoustics, seismology, electromagnetic waves
    Wave,

    /// Heat/diffusion equation: ∂u/∂t - α∇²u = 0
    ///
    /// Complexity: Low-Medium
    /// Applications: Heat transfer, mass diffusion, option pricing
    Diffusion,

    /// Navier-Stokes equations: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
    ///
    /// Complexity: Very High (nonlinear, coupled)
    /// Applications: Fluid dynamics, aerodynamics, weather prediction
    NavierStokes,

    /// Maxwell's equations: ∇×E = -∂B/∂t, ∇×B = μ₀(J + ε₀∂E/∂t)
    ///
    /// Complexity: High (coupled vector equations)
    /// Applications: Electromagnetic wave propagation, antenna design
    Electromagnetic,

    /// Acoustic wave equation with heterogeneous medium
    ///
    /// Complexity: Medium
    /// Applications: Medical ultrasound, sonar, room acoustics
    Acoustic,

    /// Elastic wave equations: ρ∂²u/∂t² = ∇·σ
    ///
    /// Complexity: High (coupled vector equations, tensor operations)
    /// Applications: Seismology, structural mechanics, elastography
    Elastic,
}

impl PdeType {
    /// Get the relative computational complexity (0.0 = easiest, 1.0 = hardest)
    ///
    /// Used for curriculum learning strategies to progressively increase
    /// task difficulty during meta-training.
    pub fn complexity(&self) -> f64 {
        match self {
            PdeType::Wave => 0.2,
            PdeType::Diffusion => 0.3,
            PdeType::Acoustic => 0.4,
            PdeType::Elastic => 0.6,
            PdeType::Electromagnetic => 0.7,
            PdeType::NavierStokes => 1.0,
        }
    }

    /// Get the typical number of coupled equations
    pub fn num_equations(&self) -> usize {
        match self {
            PdeType::Wave | PdeType::Diffusion | PdeType::Acoustic => 1,
            PdeType::Elastic => 2,         // 2D elasticity: (u_x, u_y)
            PdeType::Electromagnetic => 6, // E_x, E_y, E_z, B_x, B_y, B_z
            PdeType::NavierStokes => 4,    // u, v, w, p (or u, v, p in 2D)
        }
    }

    /// Check if the PDE is linear
    pub fn is_linear(&self) -> bool {
        matches!(
            self,
            PdeType::Wave | PdeType::Diffusion | PdeType::Acoustic | PdeType::Elastic
        )
    }
}

/// Physics parameters defining the task's governing equations
///
/// Different PDE types use different subsets of these parameters:
/// - **Wave/Acoustic**: `wave_speed`, `density`, `absorption`
/// - **Diffusion**: `density` (as diffusivity coefficient)
/// - **Navier-Stokes**: `density`, `viscosity`
/// - **Elastic**: `density`, `wave_speed` (as shear/longitudinal wave speeds)
#[derive(Debug, Clone)]
pub struct PhysicsParameters {
    /// Wave propagation speed (m/s)
    ///
    /// - Acoustic waves in air: ~343 m/s
    /// - Acoustic waves in water: ~1500 m/s
    /// - Acoustic waves in tissue: ~1540 m/s
    /// - Seismic P-waves: ~5000-8000 m/s
    pub wave_speed: f64,

    /// Material density (kg/m³)
    ///
    /// - Air: ~1.2 kg/m³
    /// - Water: ~1000 kg/m³
    /// - Soft tissue: ~1000-1100 kg/m³
    /// - Bone: ~1700-2000 kg/m³
    pub density: f64,

    /// Dynamic viscosity (Pa·s)
    ///
    /// Used for Navier-Stokes equations.
    /// - Air: ~1.8×10⁻⁵ Pa·s
    /// - Water: ~1.0×10⁻³ Pa·s
    /// - Blood: ~3-4×10⁻³ Pa·s
    pub viscosity: Option<f64>,

    /// Absorption coefficient (Np/m or dB/cm)
    ///
    /// Acoustic energy loss due to viscous friction and thermal conduction.
    /// - Air at 1 kHz: ~0.001 dB/m
    /// - Water at 1 MHz: ~0.025 dB/cm
    /// - Soft tissue at 1 MHz: ~0.5-1.0 dB/cm
    pub absorption: Option<f64>,

    /// Nonlinearity parameter (B/A or β)
    ///
    /// Characterizes nonlinear wave propagation (e.g., shock formation).
    /// - Water: B/A ≈ 5
    /// - Soft tissue: B/A ≈ 6-8
    /// - Used in Westervelt or KZK equations
    pub nonlinearity: Option<f64>,
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            wave_speed: 343.0, // Speed of sound in air at 20°C
            density: 1.2,      // Air density at sea level
            viscosity: None,
            absorption: None,
            nonlinearity: None,
        }
    }
}

impl PhysicsParameters {
    /// Create parameters for acoustic wave propagation in air
    pub fn acoustic_air() -> Self {
        Self {
            wave_speed: 343.0,
            density: 1.2,
            viscosity: None,
            absorption: Some(0.001),
            nonlinearity: None,
        }
    }

    /// Create parameters for acoustic wave propagation in water
    pub fn acoustic_water() -> Self {
        Self {
            wave_speed: 1500.0,
            density: 1000.0,
            viscosity: None,
            absorption: Some(0.025),
            nonlinearity: Some(5.0), // B/A for water
        }
    }

    /// Create parameters for acoustic wave propagation in soft tissue
    pub fn acoustic_tissue() -> Self {
        Self {
            wave_speed: 1540.0,
            density: 1050.0,
            viscosity: None,
            absorption: Some(0.5),   // At 1 MHz
            nonlinearity: Some(7.0), // Typical B/A for tissue
        }
    }

    /// Create parameters for fluid flow (Navier-Stokes)
    pub fn fluid(density: f64, viscosity: f64) -> Self {
        Self {
            wave_speed: 0.0, // Not used for N-S
            density,
            viscosity: Some(viscosity),
            absorption: None,
            nonlinearity: None,
        }
    }
}

/// Physics task definition for meta-learning
///
/// Represents a single task instance in the meta-learning framework.
/// Each task corresponds to solving a specific PDE with given parameters,
/// geometry, and boundary/initial conditions.
///
/// # Task Diversity
///
/// For effective meta-learning, tasks should be diverse across:
/// - PDE types (wave, diffusion, etc.)
/// - Physics parameters (wave speed, density, etc.)
/// - Geometric domains (rectangular, circular, irregular)
/// - Boundary conditions (Dirichlet, Neumann, mixed)
///
/// # Literature
///
/// - Grant, E., et al. (2018). "Recasting gradient-based meta-learning as hierarchical Bayes"
///   *ICLR 2018*
#[derive(Debug, Clone)]
pub struct PhysicsTask {
    /// Unique task identifier
    ///
    /// Used for tracking, debugging, and reproducibility.
    /// Format suggestion: "{pde_type}_{geometry}_{id}"
    /// Example: "wave_rectangular_001"
    pub id: String,

    /// Type of PDE governing the physics
    pub pde_type: PdeType,

    /// Physics parameters (wave speed, density, etc.)
    pub physics_params: PhysicsParameters,

    /// Geometric domain specification
    ///
    /// Defines the spatial region where the PDE is solved.
    /// Wrapped in Arc for efficient cloning across threads.
    pub geometry: Arc<crate::ml::pinn::Geometry2D>,

    /// Boundary conditions
    ///
    /// Specifies constraints at the domain boundaries.
    /// May include Dirichlet, Neumann, or Robin conditions.
    pub boundary_conditions: Vec<crate::ml::pinn::BoundaryCondition2D>,

    /// Training data (optional for few-shot learning)
    ///
    /// If provided, enables supervised learning component.
    /// If None, task relies purely on physics constraints.
    pub training_data: Option<TaskData>,

    /// Validation data for meta-training
    ///
    /// Used to compute task-specific validation loss during
    /// meta-training. Essential for evaluating adaptation quality.
    pub validation_data: TaskData,
}

impl PhysicsTask {
    /// Create a new physics task with validation
    pub fn new(
        id: String,
        pde_type: PdeType,
        physics_params: PhysicsParameters,
        geometry: Arc<crate::ml::pinn::Geometry2D>,
        boundary_conditions: Vec<crate::ml::pinn::BoundaryCondition2D>,
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
    ///
    /// Combines PDE complexity with geometric and boundary condition complexity.
    pub fn complexity(&self) -> f64 {
        let pde_complexity = self.pde_type.complexity();

        let geometry_complexity = match self.geometry.as_ref() {
            crate::ml::pinn::Geometry2D::Rectangular { .. } => 0.2,
            crate::ml::pinn::Geometry2D::Circular { .. } => 0.4,
            crate::ml::pinn::Geometry2D::MultiRegion { .. } => 1.0,
            _ => 0.6, // Default for other geometries
        };

        let bc_complexity = (self.boundary_conditions.len() as f64).min(10.0) / 10.0;

        // Weighted combination
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
/// # Data Generation Strategies
///
/// 1. **Latin Hypercube Sampling**: For collocation points
///    - Better space-filling than random sampling
///    - McKay, M. D., et al. (1979). "A comparison of three methods for selecting values of input variables"
///
/// 2. **Adaptive Sampling**: Focus on high-error regions
///    - Narayan, A., & Jakeman, J. D. (2014). "Adaptive Leja sparse grid constructions"
///
/// 3. **Sequential Sampling**: For time-dependent problems
///    - Causal training: respect temporal causality
///    - Wang, S., et al. (2020). "When and why PINNs fail to train"
#[derive(Debug, Clone)]
pub struct TaskData {
    /// Collocation points (x, y, t) for PDE residual evaluation
    ///
    /// These interior points are where the PDE constraints are enforced.
    /// More points = better physics constraint satisfaction but higher cost.
    /// Typical: 1000-10000 points
    pub collocation_points: Vec<(f64, f64, f64)>,

    /// Boundary data (x, y, t, u) for boundary condition enforcement
    ///
    /// Points on the domain boundary with known solution values.
    /// Essential for well-posed problem formulation.
    /// Typical: 100-1000 points
    pub boundary_data: Vec<(f64, f64, f64, f64)>,

    /// Initial data (x, y, t=0, u, ∂u/∂t) for initial condition enforcement
    ///
    /// Initial state of the system at t=0.
    /// For wave equations, includes both u and ∂u/∂t.
    /// For diffusion equations, only u is needed.
    /// Typical: 100-1000 points
    pub initial_data: Vec<(f64, f64, f64, f64, f64)>,
}

impl Default for TaskData {
    fn default() -> Self {
        Self {
            collocation_points: Vec::new(),
            boundary_data: Vec::new(),
            initial_data: Vec::new(),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pde_type_complexity() {
        assert_eq!(PdeType::Wave.complexity(), 0.2);
        assert_eq!(PdeType::Diffusion.complexity(), 0.3);
        assert_eq!(PdeType::NavierStokes.complexity(), 1.0);
        assert!(PdeType::Wave.complexity() < PdeType::NavierStokes.complexity());
    }

    #[test]
    fn test_pde_type_num_equations() {
        assert_eq!(PdeType::Wave.num_equations(), 1);
        assert_eq!(PdeType::Elastic.num_equations(), 2);
        assert_eq!(PdeType::Electromagnetic.num_equations(), 6);
        assert_eq!(PdeType::NavierStokes.num_equations(), 4);
    }

    #[test]
    fn test_pde_type_linearity() {
        assert!(PdeType::Wave.is_linear());
        assert!(PdeType::Diffusion.is_linear());
        assert!(!PdeType::NavierStokes.is_linear());
    }

    #[test]
    fn test_physics_parameters_default() {
        let params = PhysicsParameters::default();
        assert_eq!(params.wave_speed, 343.0);
        assert_eq!(params.density, 1.2);
        assert!(params.viscosity.is_none());
    }

    #[test]
    fn test_physics_parameters_presets() {
        let air = PhysicsParameters::acoustic_air();
        assert_eq!(air.wave_speed, 343.0);

        let water = PhysicsParameters::acoustic_water();
        assert_eq!(water.wave_speed, 1500.0);

        let tissue = PhysicsParameters::acoustic_tissue();
        assert_eq!(tissue.wave_speed, 1540.0);
        assert!(tissue.nonlinearity.is_some());
    }

    #[test]
    fn test_physics_parameters_fluid() {
        let fluid = PhysicsParameters::fluid(1000.0, 0.001);
        assert_eq!(fluid.density, 1000.0);
        assert_eq!(fluid.viscosity, Some(0.001));
    }

    #[test]
    fn test_task_data_default() {
        let data = TaskData::default();
        assert!(data.is_empty());
        assert_eq!(data.total_points(), 0);
    }

    #[test]
    fn test_task_data_with_capacity() {
        let data = TaskData::with_capacity(1000, 100, 50);
        assert!(data.is_empty());
        assert_eq!(data.collocation_points.capacity(), 1000);
        assert_eq!(data.boundary_data.capacity(), 100);
        assert_eq!(data.initial_data.capacity(), 50);
    }

    #[test]
    fn test_task_data_statistics() {
        let mut data = TaskData::default();
        data.collocation_points.push((0.0, 0.0, 0.0));
        data.boundary_data.push((0.0, 0.0, 0.0, 0.0));
        data.initial_data.push((0.0, 0.0, 0.0, 0.0, 0.0));

        let stats = data.statistics();
        assert_eq!(stats.num_collocation, 1);
        assert_eq!(stats.num_boundary, 1);
        assert_eq!(stats.num_initial, 1);
        assert_eq!(stats.total, 3);
    }

    #[test]
    fn test_task_data_not_empty() {
        let mut data = TaskData::default();
        data.collocation_points.push((0.0, 0.0, 0.0));
        assert!(!data.is_empty());
        assert_eq!(data.total_points(), 1);
    }
}
