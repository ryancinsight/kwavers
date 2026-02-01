//! Architecture Layer Validation and Enforcement
//!
//! Automated validation to prevent architectural drift and layer violations
//! in the deep vertical 9-layer hierarchy.
//!
//! ## Layer Hierarchy (Strict Downward Dependencies Only)
//!
//! ```
//! Layer 0 (Foundation)
//!   core/             - Error handling, time, constants, logging
//!   ↓ ONLY
//! Layer 1 (Math)
//!   math/             - Linear algebra, FFT, SIMD, numerics
//!   ↓ ONLY
//! Layer 2 (Domain)
//!   domain/           - 14 bounded contexts (grid, sensor, source, boundary, medium, etc.)
//!   ↓ ONLY
//! Layer 3 (Physics)
//!   physics/          - 5 domains (acoustics, thermal, EM, optics, chemistry)
//!   ↓ ONLY
//! Layer 4 (Solver)
//!   solver/           - FDTD, PSTD, SEM, BEM, FEM, inverse, coupled
//!   ↓ ONLY
//! Layer 5 (Simulation)
//!   simulation/       - Orchestration, factories, backends
//!   ↓ ONLY
//! Layer 6 (Clinical)
//!   clinical/         - Therapy, imaging, safety, monitoring, patient management
//!   ↓ ONLY
//! Layer 7 (Analysis)
//!   analysis/         - Signal processing, ML, visualization, validation
//!   ↓ ONLY
//! Layer 8 (Infrastructure)
//!   infra/            - I/O, API, runtime, cloud
//!   gpu/              - GPU-accelerated implementations
//!   infrastructure/   - Hardware abstraction, device management
//! ```
//!
//! ## Validation Rules
//!
//! 1. **No Circular Dependencies**: No module can depend on itself transitively
//! 2. **Downward Only**: Layer N can only depend on Layer N-1, N-2, ... (never upward)
//! 3. **No Sideways**: Layer N cannot depend on other Layer N modules (except bounded contexts within Layer)
//! 4. **Clear Boundaries**: Domain model ownership defined at each layer
//! 5. **SSOT**: Single Source of Truth for each concern across layers

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::{HashMap, HashSet};

/// Layer definition in the architecture
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
    /// Get string representation
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

    /// Check if one layer can depend on another
    pub fn can_depend_on(dependor: ArchitectureLayer, dependency: ArchitectureLayer) -> bool {
        // Layer N can depend on Layer 0..N-1
        dependor > dependency
    }

    /// Get maximum layer a given layer can depend on
    pub fn max_dependency(layer: ArchitectureLayer) -> ArchitectureLayer {
        match layer {
            Self::Core => Self::Core, // Core depends on nothing
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

/// Module dependency information
#[derive(Debug, Clone)]
pub struct ModuleDependency {
    /// Module name (e.g., "solver::forward::fdtd")
    pub module: String,

    /// Layer this module belongs to
    pub layer: ArchitectureLayer,

    /// Modules this depends on
    pub dependencies: Vec<String>,

    /// Layers of dependencies
    pub dependency_layers: Vec<ArchitectureLayer>,
}

/// Architecture validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,

    /// Violations found
    pub violations: Vec<LayerViolation>,

    /// Summary statistics
    pub stats: ValidationStats,
}

/// A layer hierarchy violation
#[derive(Debug, Clone)]
pub struct LayerViolation {
    /// Violating module
    pub module: String,

    /// Layer of violating module
    pub layer: ArchitectureLayer,

    /// Invalid dependency
    pub invalid_dependency: String,

    /// Layer of dependency
    pub dependency_layer: ArchitectureLayer,

    /// Violation type
    pub violation_type: ViolationType,
}

/// Type of architecture violation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationType {
    /// Upward dependency (violates layer hierarchy)
    UpwardDependency,

    /// Circular dependency
    CircularDependency,

    /// Sideways dependency (same layer, different context)
    SidewaysDependency,
}

impl ViolationType {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UpwardDependency => "Upward Dependency",
            Self::CircularDependency => "Circular Dependency",
            Self::SidewaysDependency => "Sideways Dependency",
        }
    }
}

/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStats {
    /// Total modules analyzed
    pub total_modules: usize,

    /// Modules per layer
    pub modules_per_layer: HashMap<ArchitectureLayer, usize>,

    /// Total dependencies
    pub total_dependencies: usize,

    /// Violations found
    pub violation_count: usize,
}

/// Architecture validator
#[derive(Debug)]
pub struct ArchitectureValidator {
    /// Module layer mappings
    module_layers: HashMap<String, ArchitectureLayer>,

    /// Module dependencies
    dependencies: HashMap<String, Vec<String>>,
}

impl Default for ArchitectureValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchitectureValidator {
    /// Create new architecture validator
    pub fn new() -> Self {
        Self {
            module_layers: HashMap::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Register a module at a specific layer
    pub fn register_module(&mut self, module: impl Into<String>, layer: ArchitectureLayer) {
        self.module_layers.insert(module.into(), layer);
    }

    /// Add a dependency between modules
    pub fn add_dependency(&mut self, from: impl Into<String>, to: impl Into<String>) {
        let from_str = from.into();
        let to_str = to.into();

        self.dependencies
            .entry(from_str)
            .or_insert_with(Vec::new)
            .push(to_str);
    }

    /// Validate the architecture
    pub fn validate(&self) -> KwaversResult<ValidationResult> {
        let mut violations = Vec::new();
        let mut total_deps = 0;

        // Check each module's dependencies
        for (module, deps) in &self.dependencies {
            let module_layer = *self
                .module_layers
                .get(module)
                .ok_or_else(|| KwaversError::InvalidInput(format!("Unknown module: {}", module)))?;

            for dep in deps {
                total_deps += 1;

                let dep_layer = *self.module_layers.get(dep).ok_or_else(|| {
                    KwaversError::InvalidInput(format!("Unknown dependency: {}", dep))
                })?;

                // Check for upward dependency
                if !ArchitectureLayer::can_depend_on(module_layer, dep_layer) {
                    violations.push(LayerViolation {
                        module: module.clone(),
                        layer: module_layer,
                        invalid_dependency: dep.clone(),
                        dependency_layer: dep_layer,
                        violation_type: ViolationType::UpwardDependency,
                    });
                }
            }
        }

        // Check for circular dependencies (simplified)
        self.check_circular_dependencies(&mut violations)?;

        // Build statistics
        let mut modules_per_layer: HashMap<ArchitectureLayer, usize> = HashMap::new();
        for layer in self.module_layers.values() {
            *modules_per_layer.entry(*layer).or_insert(0) += 1;
        }

        let passed = violations.is_empty();
        let stats = ValidationStats {
            total_modules: self.module_layers.len(),
            modules_per_layer,
            total_dependencies: total_deps,
            violation_count: violations.len(),
        };

        Ok(ValidationResult {
            passed,
            violations,
            stats,
        })
    }

    /// Check for circular dependencies (depth-first search)
    fn check_circular_dependencies(
        &self,
        violations: &mut Vec<LayerViolation>,
    ) -> KwaversResult<()> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut rec_stack: HashSet<String> = HashSet::new();

        for module in self.module_layers.keys() {
            if !visited.contains(module) {
                self.dfs_check_cycle(module, &mut visited, &mut rec_stack, violations)?;
            }
        }

        Ok(())
    }

    /// Depth-first search for cycles
    fn dfs_check_cycle(
        &self,
        module: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        violations: &mut Vec<LayerViolation>,
    ) -> KwaversResult<()> {
        visited.insert(module.to_string());
        rec_stack.insert(module.to_string());

        if let Some(deps) = self.dependencies.get(module) {
            for dep in deps {
                if !visited.contains(dep) {
                    self.dfs_check_cycle(dep, visited, rec_stack, violations)?;
                } else if rec_stack.contains(dep) {
                    // Circular dependency found
                    let module_layer = *self
                        .module_layers
                        .get(module)
                        .ok_or_else(|| KwaversError::InvalidInput("Unknown module".to_string()))?;
                    let dep_layer = *self.module_layers.get(dep).ok_or_else(|| {
                        KwaversError::InvalidInput("Unknown dependency".to_string())
                    })?;

                    violations.push(LayerViolation {
                        module: module.to_string(),
                        layer: module_layer,
                        invalid_dependency: dep.clone(),
                        dependency_layer: dep_layer,
                        violation_type: ViolationType::CircularDependency,
                    });
                }
            }
        }

        rec_stack.remove(module);
        Ok(())
    }

    /// Generate a validation report
    pub fn report(&self, result: &ValidationResult) -> String {
        let mut report = String::new();

        report.push_str("=== ARCHITECTURE VALIDATION REPORT ===\n\n");

        report.push_str(&format!(
            "Status: {}\n",
            if result.passed {
                "✓ PASSED"
            } else {
                "✗ FAILED"
            }
        ));

        report.push_str(&format!("Modules: {}\n", result.stats.total_modules));
        report.push_str(&format!(
            "Dependencies: {}\n",
            result.stats.total_dependencies
        ));
        report.push_str(&format!("Violations: {}\n\n", result.stats.violation_count));

        if !result.violations.is_empty() {
            report.push_str("VIOLATIONS:\n");
            for (i, violation) in result.violations.iter().enumerate() {
                report.push_str(&format!(
                    "\n{}. {} in {}\n",
                    i + 1,
                    violation.violation_type.as_str(),
                    violation.module
                ));
                report.push_str(&format!("   Layer: {}\n", violation.layer.as_str()));
                report.push_str(&format!(
                    "   Invalid Dependency: {} (Layer: {})\n",
                    violation.invalid_dependency,
                    violation.dependency_layer.as_str()
                ));
            }
        }

        report.push_str("\n=== END REPORT ===\n");
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_ordering() {
        assert!(ArchitectureLayer::can_depend_on(
            ArchitectureLayer::Solver,
            ArchitectureLayer::Physics
        ));
        assert!(!ArchitectureLayer::can_depend_on(
            ArchitectureLayer::Physics,
            ArchitectureLayer::Solver
        ));
    }

    #[test]
    fn test_max_dependency() {
        assert_eq!(
            ArchitectureLayer::max_dependency(ArchitectureLayer::Solver),
            ArchitectureLayer::Physics
        );
        assert_eq!(
            ArchitectureLayer::max_dependency(ArchitectureLayer::Clinical),
            ArchitectureLayer::Simulation
        );
    }

    #[test]
    fn test_valid_architecture() {
        let mut validator = ArchitectureValidator::new();

        // Register modules
        validator.register_module("core::error", ArchitectureLayer::Core);
        validator.register_module("math::linear_algebra", ArchitectureLayer::Math);
        validator.register_module("solver::fdtd", ArchitectureLayer::Solver);

        // Add valid dependencies (downward only)
        validator.add_dependency("math::linear_algebra", "core::error");
        validator.add_dependency("solver::fdtd", "math::linear_algebra");
        validator.add_dependency("solver::fdtd", "core::error");

        let result = validator.validate().unwrap();
        assert!(result.passed);
        assert_eq!(result.violations.len(), 0);
    }

    #[test]
    fn test_upward_dependency_violation() {
        let mut validator = ArchitectureValidator::new();

        validator.register_module("core::error", ArchitectureLayer::Core);
        validator.register_module("solver::fdtd", ArchitectureLayer::Solver);

        // Invalid: Core depends on Solver (upward)
        validator.add_dependency("core::error", "solver::fdtd");

        let result = validator.validate().unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(
            result.violations[0].violation_type,
            ViolationType::UpwardDependency
        );
    }

    #[test]
    fn test_validation_report() {
        let mut validator = ArchitectureValidator::new();
        validator.register_module("core::error", ArchitectureLayer::Core);
        validator.register_module("solver::fdtd", ArchitectureLayer::Solver);
        validator.add_dependency("core::error", "solver::fdtd");

        let result = validator.validate().unwrap();
        let report = validator.report(&result);

        assert!(report.contains("VALIDATION REPORT"));
        assert!(report.contains("FAILED"));
        assert!(report.contains("Upward Dependency"));
    }

    #[test]
    fn test_layer_display() {
        assert_eq!(ArchitectureLayer::Core.as_str(), "Core (Layer 0)");
        assert_eq!(ArchitectureLayer::Solver.as_str(), "Solver (Layer 4)");
        assert_eq!(
            ArchitectureLayer::Infrastructure.as_str(),
            "Infrastructure (Layer 8)"
        );
    }
}
