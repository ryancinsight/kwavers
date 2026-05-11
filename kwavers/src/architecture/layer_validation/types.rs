//! Types for architecture layer validation.

use std::collections::HashMap;

/// Layer definition in the architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArchitectureLayer {
    /// Layer 0: Foundation (core/)
    Core = 0,
    /// Layer 1: Mathematics (math/)
    Math = 1,
    /// Layer 2: Domain Models (domain/)
    Domain = 2,
    /// Layer 3: Physics (physics/)
    Physics = 3,
    /// Layer 4: Solvers (solver/)
    Solver = 4,
    /// Layer 5: Simulation (simulation/)
    Simulation = 5,
    /// Layer 6: Clinical (clinical/)
    Clinical = 6,
    /// Layer 7: Analysis (analysis/)
    Analysis = 7,
    /// Layer 8: Infrastructure (infra/, gpu/, infrastructure/)
    Infrastructure = 8,
}

impl ArchitectureLayer {
    /// Get string representation.
    #[must_use] 
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Core => "Core (Layer 0)",
            Self::Math => "Math (Layer 1)",
            Self::Domain => "Domain (Layer 2)",
            Self::Physics => "Physics (Layer 3)",
            Self::Solver => "Solver (Layer 4)",
            Self::Simulation => "Simulation (Layer 5)",
            Self::Clinical => "Clinical (Layer 6)",
            Self::Analysis => "Analysis (Layer 7)",
            Self::Infrastructure => "Infrastructure (Layer 8)",
        }
    }

    /// Check if one layer can depend on another.
    ///
    /// Layer N can depend on Layer 0..N-1 (strictly downward).
    #[must_use] 
    pub fn can_depend_on(dependor: Self, dependency: Self) -> bool {
        dependor > dependency
    }

    /// Get the maximum layer a given layer can depend on.
    #[must_use] 
    pub fn max_dependency(layer: Self) -> Self {
        match layer {
            Self::Core => Self::Core,
            Self::Math => Self::Core,
            Self::Domain => Self::Math,
            Self::Physics => Self::Domain,
            Self::Solver => Self::Physics,
            Self::Simulation => Self::Solver,
            Self::Clinical => Self::Simulation,
            Self::Analysis => Self::Clinical,
            Self::Infrastructure => Self::Analysis,
        }
    }
}

/// Module dependency information.
#[derive(Debug, Clone)]
pub struct ModuleDependency {
    /// Module name (e.g., "solver::forward::fdtd")
    pub module: String,

    /// Layer this module belongs to.
    pub layer: ArchitectureLayer,

    /// Modules this depends on.
    pub dependencies: Vec<String>,

    /// Layers of dependencies.
    pub dependency_layers: Vec<ArchitectureLayer>,
}

/// Architecture validation result.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub passed: bool,

    /// Violations found.
    pub violations: Vec<LayerViolation>,

    /// Summary statistics.
    pub stats: ValidationStats,
}

/// A layer hierarchy violation.
#[derive(Debug, Clone)]
pub struct LayerViolation {
    /// Violating module.
    pub module: String,

    /// Layer of violating module.
    pub layer: ArchitectureLayer,

    /// Invalid dependency.
    pub invalid_dependency: String,

    /// Layer of dependency.
    pub dependency_layer: ArchitectureLayer,

    /// Violation type.
    pub violation_type: ViolationType,
}

/// Type of architecture violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationType {
    /// Upward dependency (violates layer hierarchy).
    UpwardDependency,

    /// Circular dependency.
    CircularDependency,

    /// Sideways dependency (same layer, different context).
    SidewaysDependency,
}

impl ViolationType {
    /// Get string representation.
    #[must_use] 
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UpwardDependency => "Upward Dependency",
            Self::CircularDependency => "Circular Dependency",
            Self::SidewaysDependency => "Sideways Dependency",
        }
    }
}

/// Validation statistics.
#[derive(Debug, Clone)]
pub struct ValidationStats {
    /// Total modules analyzed.
    pub total_modules: usize,

    /// Modules per layer.
    pub modules_per_layer: HashMap<ArchitectureLayer, usize>,

    /// Total dependencies.
    pub total_dependencies: usize,

    /// Violations found.
    pub violation_count: usize,
}
