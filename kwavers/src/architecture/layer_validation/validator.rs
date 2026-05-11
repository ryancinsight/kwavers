//! Architecture validator implementation.

use super::types::{
    ArchitectureLayer, LayerViolation, ValidationResult, ValidationStats, ViolationType,
};
use crate::core::error::{KwaversError, KwaversResult};
use std::collections::{HashMap, HashSet};

/// Architecture validator enforcing strict downward-only layer dependencies.
#[derive(Debug)]
pub struct ArchitectureValidator {
    /// Module layer mappings.
    module_layers: HashMap<String, ArchitectureLayer>,

    /// Module dependencies.
    dependencies: HashMap<String, Vec<String>>,
}

impl Default for ArchitectureValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchitectureValidator {
    /// Create new architecture validator.
    #[must_use] 
    pub fn new() -> Self {
        Self {
            module_layers: HashMap::new(),
            dependencies: HashMap::new(),
        }
    }

    /// Register a module at a specific layer.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn register_module(&mut self, module: impl Into<String>, layer: ArchitectureLayer) {
        self.module_layers.insert(module.into(), layer);
    }

    /// Add a dependency between modules.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn add_dependency(&mut self, from: impl Into<String>, to: impl Into<String>) {
        self.dependencies
            .entry(from.into())
            .or_default()
            .push(to.into());
    }

    /// Validate the architecture.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn validate(&self) -> KwaversResult<ValidationResult> {
        let mut violations = Vec::new();
        let mut total_deps = 0;

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

        self.check_circular_dependencies(&mut violations)?;

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

    /// Check for circular dependencies using depth-first search.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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

    /// Depth-first search for cycle detection.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn dfs_check_cycle(
        &self,
        module: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        violations: &mut Vec<LayerViolation>,
    ) -> KwaversResult<()> {
        visited.insert(module.to_owned());
        rec_stack.insert(module.to_owned());

        if let Some(deps) = self.dependencies.get(module) {
            for dep in deps {
                if !visited.contains(dep) {
                    self.dfs_check_cycle(dep, visited, rec_stack, violations)?;
                } else if rec_stack.contains(dep) {
                    let module_layer = *self
                        .module_layers
                        .get(module)
                        .ok_or_else(|| KwaversError::InvalidInput("Unknown module".to_owned()))?;
                    let dep_layer = *self.module_layers.get(dep).ok_or_else(|| {
                        KwaversError::InvalidInput("Unknown dependency".to_owned())
                    })?;

                    violations.push(LayerViolation {
                        module: module.to_owned(),
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

    /// Generate a human-readable validation report.
    #[must_use] 
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
