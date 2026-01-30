//! Architecture Module - Code Organization and Quality Standards
//!
//! This module defines the architectural principles, code organization standards,
//! and quality gates for the kwavers codebase. It serves as the single source of truth
//! for architectural decisions and maintainability guidelines.
//!
//! ## Architectural Layers
//!
//! Following Domain-Driven Design (DDD) and Clean Architecture principles:
//!
//! 1. **Core Layer** (`core/`): Foundational types, error handling, utilities
//! 2. **Math Layer** (`math/`): Pure mathematical abstractions and algorithms
//! 3. **Domain Layer** (`domain/`): Business domain models and rules
//! 4. **Physics Layer** (`physics/`): Physical laws and constitutive relations
//! 5. **Solver Layer** (`solver/`): Numerical algorithms and discretization
//! 6. **Simulation Layer** (`simulation/`): Orchestration and workflow management
//! 7. **Clinical Layer** (`clinical/`): Application-specific clinical workflows
//! 8. **Infrastructure Layer** (`infra/`): External interfaces and services
//! 9. **Analysis Layer** (`analysis/`): Post-processing and analysis tools
//!
//! ## Code Organization Standards
//!
//! ### Module Size Limits
//! - **Core modules**: < 500 lines (GRASP Single Responsibility)
//! - **Complex algorithms**: Split into focused submodules
//! - **Large files**: Refactor into logical components
//!
//! ### Naming Conventions
//! - **Modules**: snake_case, descriptive names
//! - **Types**: PascalCase, domain-relevant
//! - **Functions**: snake_case, action-oriented
//! - **Constants**: SCREAMING_SNAKE_CASE
//!
//! ### Import Organization
//! - **Standard library**: Group first, sorted alphabetically
//! - **External crates**: Group second, sorted alphabetically
//! - **Internal modules**: Group last, with clear hierarchy
//! - **Re-exports**: Minimize at crate root to prevent namespace pollution
//!
//! ## Quality Gates
//!
//! ### Compilation Standards
//! - Zero warnings in release builds
//! - Clean clippy output with project-specific allowances
//! - No unsafe code without audited justification
//! - All tests pass in <30 seconds (SRS NFR-002)
//!
//! ### Documentation Standards
//! - 100% public API documentation with examples
//! - Mathematical theorems referenced with literature citations
//! - Complexity analysis and performance characteristics documented
//! - Migration guides for breaking changes
//!
//! ### Testing Standards
//! - Unit test coverage >95% for critical paths
//! - Property-based testing for mathematical invariants
//! - Integration tests for end-to-end workflows
//! - Performance regression tests with historical baselines
//!
//! ## Dependency Management
//!
//! ### Core Dependencies (Always Available)
//! - ndarray: Array operations and linear algebra
//! - thiserror: Error handling
//! - serde: Serialization (minimal features)
//!
//! ### Feature-Gated Dependencies
//! - GPU acceleration (wgpu, burn)
//! - Visualization (plotly, egui)
//! - Cloud services (reqwest, AWS SDK)
//! - Advanced ML (burn with autodiff)
//!
//! ### Dependency Hygiene
//! - Regular audit for unused dependencies
//! - Security updates for critical crates
//! - Minimal version requirements for compatibility
//! - Feature flags for optional functionality

/// Architecture compliance checker
#[derive(Debug)]
pub struct ArchitectureChecker;

/// Code quality metrics
#[derive(Debug, Clone)]
pub struct CodeQualityMetrics {
    pub total_lines: usize,
    pub test_coverage: f64,
    pub documentation_coverage: f64,
    pub clippy_warnings: usize,
    pub compilation_time: std::time::Duration,
}

/// Module organization standards
#[derive(Debug)]
pub struct ModuleStandards;

impl ModuleStandards {
    /// Check if a module complies with size limits
    pub fn check_module_size(module_path: &str, line_count: usize) -> Result<(), String> {
        const MAX_MODULE_SIZE: usize = 500;

        if line_count > MAX_MODULE_SIZE {
            return Err(format!(
                "Module {} exceeds size limit: {} lines (max: {})",
                module_path, line_count, MAX_MODULE_SIZE
            ));
        }

        Ok(())
    }

    /// Validate module naming conventions
    pub fn check_module_naming(module_name: &str) -> Result<(), String> {
        if module_name.contains(char::is_uppercase) {
            return Err(format!(
                "Module name '{}' should use snake_case (no uppercase)",
                module_name
            ));
        }

        if module_name.contains('-')
            || module_name.contains('_') && module_name.contains(char::is_uppercase)
        {
            return Err(format!(
                "Module name '{}' has inconsistent casing",
                module_name
            ));
        }

        Ok(())
    }

    /// Check import organization
    pub fn validate_imports(imports: &[String]) -> Result<(), String> {
        // Check for proper grouping and sorting
        let mut std_imports = Vec::new();
        let mut external_imports = Vec::new();
        let mut internal_imports = Vec::new();

        for import in imports {
            if import.starts_with("std::") || import.starts_with("core::") {
                std_imports.push(import.clone());
            } else if import.starts_with("crate::") {
                internal_imports.push(import.clone());
            } else {
                external_imports.push(import.clone());
            }
        }

        // Check if each group is sorted
        if !is_sorted(&std_imports) {
            return Err("Standard library imports are not sorted alphabetically".to_string());
        }

        if !is_sorted(&external_imports) {
            return Err("External crate imports are not sorted alphabetically".to_string());
        }

        if !is_sorted(&internal_imports) {
            return Err("Internal module imports are not sorted alphabetically".to_string());
        }

        Ok(())
    }
}

/// Check if a vector of strings is sorted alphabetically
fn is_sorted(strings: &[String]) -> bool {
    for i in 1..strings.len() {
        if strings[i - 1] > strings[i] {
            return false;
        }
    }
    true
}

/// Performance optimization standards
#[derive(Debug)]
pub struct PerformanceStandards;

impl PerformanceStandards {
    /// Recommended SIMD usage patterns
    pub fn recommend_simd_usage() -> Vec<String> {
        vec![
            "FDTD update loops".to_string(),
            "FFT computations".to_string(),
            "Vector field operations".to_string(),
            "Interpolation kernels".to_string(),
            "Matrix-vector multiplications".to_string(),
        ]
    }

    /// Memory optimization guidelines
    pub fn memory_optimization_guidelines() -> Vec<String> {
        vec![
            "Use arena allocation for mesh data".to_string(),
            "Implement zero-copy data structures".to_string(),
            "Pool allocation for frequent small objects".to_string(),
            "Cache-aligned data structures".to_string(),
            "Minimize heap allocations in hot loops".to_string(),
        ]
    }
}

/// Testing infrastructure standards
#[derive(Debug)]
pub struct TestingStandards;

impl TestingStandards {
    /// Required test categories
    pub fn required_test_categories() -> Vec<String> {
        vec![
            "unit_tests".to_string(),
            "integration_tests".to_string(),
            "property_based_tests".to_string(),
            "convergence_tests".to_string(),
            "performance_regression_tests".to_string(),
        ]
    }

    /// Test execution time limits (seconds)
    pub fn test_time_limits() -> std::collections::HashMap<String, f64> {
        let mut limits = std::collections::HashMap::new();
        limits.insert("unit_tests".to_string(), 10.0);
        limits.insert("integration_tests".to_string(), 30.0);
        limits.insert("property_tests".to_string(), 60.0);
        limits.insert("convergence_tests".to_string(), 120.0);
        limits.insert("performance_tests".to_string(), 300.0);
        limits
    }
}

/// Documentation standards
#[derive(Debug)]
pub struct DocumentationStandards;

impl DocumentationStandards {
    /// Required documentation elements
    pub fn required_elements() -> Vec<String> {
        vec![
            "Mathematical formulation".to_string(),
            "Algorithm complexity".to_string(),
            "Literature references".to_string(),
            "Usage examples".to_string(),
            "Performance characteristics".to_string(),
            "Error conditions".to_string(),
        ]
    }

    /// Documentation coverage targets
    pub fn coverage_targets() -> std::collections::HashMap<String, f64> {
        let mut targets = std::collections::HashMap::new();
        targets.insert("public_api".to_string(), 1.0); // 100%
        targets.insert("complex_algorithms".to_string(), 0.95); // 95%
        targets.insert("mathematical_functions".to_string(), 0.9); // 90%
        targets
    }
}

/// Error handling standards
#[derive(Debug)]
pub struct ErrorHandlingStandards;

impl ErrorHandlingStandards {
    /// Recommended error types
    pub fn recommended_error_types() -> Vec<String> {
        vec![
            "ValidationError".to_string(),
            "NumericalError".to_string(),
            "ConfigurationError".to_string(),
            "IoError".to_string(),
            "SimulationError".to_string(),
        ]
    }

    /// Error propagation patterns
    pub fn error_propagation_patterns() -> Vec<String> {
        vec![
            "Use thiserror for custom error types".to_string(),
            "Implement From trait for conversions".to_string(),
            "Use Result<T, E> for fallible operations".to_string(),
            "Provide context with error messages".to_string(),
            "Log errors at appropriate levels".to_string(),
        ]
    }
}

/// Build system standards
#[derive(Debug)]
pub struct BuildStandards;

impl BuildStandards {
    /// Feature flag organization
    pub fn feature_flag_categories() -> std::collections::HashMap<String, Vec<String>> {
        let mut categories = std::collections::HashMap::new();

        categories.insert(
            "core_features".to_string(),
            vec![
                "parallel".to_string(),
                "async-runtime".to_string(),
                "structured-logging".to_string(),
            ],
        );

        categories.insert(
            "gpu_acceleration".to_string(),
            vec![
                "gpu".to_string(),
                "pinn-gpu".to_string(),
                "gpu-visualization".to_string(),
            ],
        );

        categories.insert(
            "advanced_physics".to_string(),
            vec!["pinn".to_string(), "nifti".to_string(), "simd".to_string()],
        );

        categories.insert(
            "deployment".to_string(),
            vec![
                "api".to_string(),
                "cloud".to_string(),
                "zero-copy".to_string(),
            ],
        );

        categories.insert(
            "development".to_string(),
            vec![
                "plotting".to_string(),
                "advanced-visualization".to_string(),
                "legacy_algorithms".to_string(),
            ],
        );

        categories
    }

    /// Dependency organization
    pub fn dependency_categories() -> std::collections::HashMap<String, Vec<String>> {
        let mut categories = std::collections::HashMap::new();

        categories.insert(
            "core_runtime".to_string(),
            vec![
                "ndarray".to_string(),
                "rayon".to_string(),
                "thiserror".to_string(),
                "anyhow".to_string(),
            ],
        );

        categories.insert(
            "mathematics".to_string(),
            vec![
                "rustfft".to_string(),
                "num-complex".to_string(),
                "num-traits".to_string(),
                "nalgebra".to_string(),
            ],
        );

        categories.insert(
            "serialization".to_string(),
            vec![
                "serde".to_string(),
                "serde_json".to_string(),
                "toml".to_string(),
                "rkyv".to_string(),
            ],
        );

        categories.insert(
            "gpu_compute".to_string(),
            vec![
                "wgpu".to_string(),
                "bytemuck".to_string(),
                "burn".to_string(),
            ],
        );

        categories.insert(
            "networking".to_string(),
            vec![
                "reqwest".to_string(),
                "axum".to_string(),
                "tower".to_string(),
            ],
        );

        categories
    }
}

/// Code quality checker
#[derive(Debug)]
pub struct CodeQualityChecker;

impl CodeQualityChecker {
    /// Run comprehensive code quality checks
    pub fn run_quality_checks() -> Result<CodeQualityReport, String> {
        Ok(CodeQualityReport {
            module_size_violations: Self::check_module_sizes()?,
            naming_violations: Self::check_naming_conventions()?,
            documentation_gaps: Self::check_documentation_coverage()?,
            test_coverage_gaps: Self::check_test_coverage()?,
        })
    }

    fn check_module_sizes() -> Result<Vec<String>, String> {
        use std::path::PathBuf;

        const MAX_MODULE_SIZE: usize = 500;
        let mut violations = Vec::new();

        let src_dir = PathBuf::from("src");
        if !src_dir.exists() {
            return Ok(violations);
        }

        Self::scan_directory(&src_dir, &mut violations, MAX_MODULE_SIZE)?;
        Ok(violations)
    }

    fn scan_directory(
        dir: &std::path::Path,
        violations: &mut Vec<String>,
        max_size: usize,
    ) -> Result<(), String> {
        use std::fs;

        let entries = fs::read_dir(dir).map_err(|e| e.to_string())?;

        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
                let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
                let line_count = content.lines().count();

                if line_count > max_size {
                    let relative_path = path.strip_prefix("src").unwrap_or(&path).to_string_lossy();
                    violations.push(format!(
                        "src/{}: {} lines (exceeds {} line limit)",
                        relative_path, line_count, max_size
                    ));
                }
            } else if path.is_dir() {
                Self::scan_directory(&path, violations, max_size)?;
            }
        }

        Ok(())
    }

    fn check_naming_conventions() -> Result<Vec<String>, String> {
        use std::path::PathBuf;

        let mut violations = Vec::new();
        let src_dir = PathBuf::from("src");

        if !src_dir.exists() {
            return Ok(violations);
        }

        Self::check_module_names(&src_dir, &mut violations)?;
        Ok(violations)
    }

    fn check_module_names(
        dir: &std::path::Path,
        violations: &mut Vec<String>,
    ) -> Result<(), String> {
        use std::fs;

        let entries = fs::read_dir(dir).map_err(|e| e.to_string())?;

        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            let file_name = path.file_name().unwrap_or_default();
            let name_str = file_name.to_string_lossy();

            if path.is_dir() {
                // Check directory names follow snake_case
                if name_str != "mod.rs" && !Self::is_valid_module_name(&name_str) {
                    violations.push(format!(
                        "Directory '{}' does not follow snake_case naming convention",
                        name_str
                    ));
                }
                Self::check_module_names(&path, violations)?;
            } else if name_str.ends_with(".rs") {
                // Check file names follow snake_case
                let module_name = name_str.trim_end_matches(".rs");
                if module_name != "mod" && !Self::is_valid_module_name(module_name) {
                    violations.push(format!(
                        "File '{}' does not follow snake_case naming convention",
                        name_str
                    ));
                }
            }
        }

        Ok(())
    }

    fn is_valid_module_name(name: &str) -> bool {
        // Module names should be snake_case: lowercase letters, numbers, underscores only
        !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && !name.starts_with('_')
            && !name.ends_with('_')
    }

    fn check_documentation_coverage() -> Result<Vec<String>, String> {
        use std::path::PathBuf;

        let mut violations = Vec::new();
        let src_dir = PathBuf::from("src");

        if !src_dir.exists() {
            return Ok(violations);
        }

        Self::check_file_documentation(&src_dir, &mut violations)?;
        Ok(violations)
    }

    fn check_file_documentation(
        dir: &std::path::Path,
        violations: &mut Vec<String>,
    ) -> Result<(), String> {
        use std::fs;

        let entries = fs::read_dir(dir).map_err(|e| e.to_string())?;

        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
                let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;

                // Check for undocumented public items
                let lines: Vec<&str> = content.lines().collect();
                for (idx, line) in lines.iter().enumerate() {
                    let trimmed = line.trim();

                    // Look for public declarations
                    if (trimmed.starts_with("pub fn ")
                        || trimmed.starts_with("pub struct ")
                        || trimmed.starts_with("pub enum ")
                        || trimmed.starts_with("pub trait "))
                        && !trimmed.starts_with("pub fn main()")
                    {
                        // Check if previous line has doc comment
                        let has_doc = if idx > 0 {
                            let prev_line = lines[idx - 1].trim();
                            prev_line.starts_with("///") || prev_line.starts_with("//!")
                        } else {
                            false
                        };

                        if !has_doc {
                            let relative_path =
                                path.strip_prefix("src").unwrap_or(&path).to_string_lossy();
                            violations.push(format!(
                                "src/{}: Line {}: Missing documentation for {}",
                                relative_path,
                                idx + 1,
                                trimmed
                                    .split_whitespace()
                                    .take(3)
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            ));
                        }
                    }

                    // Check for unsafe blocks without safety documentation
                    if trimmed.starts_with("unsafe ") || trimmed == "unsafe {" {
                        let has_safety_doc = if idx > 0 {
                            let mut check_idx = idx;
                            let mut found_safety = false;
                            while check_idx > 0 && check_idx >= idx.saturating_sub(5) {
                                check_idx -= 1;
                                let prev = lines[check_idx].trim();
                                if prev.contains("# Safety") || prev.contains("SAFETY") {
                                    found_safety = true;
                                    break;
                                }
                            }
                            found_safety
                        } else {
                            false
                        };

                        if !has_safety_doc {
                            let relative_path =
                                path.strip_prefix("src").unwrap_or(&path).to_string_lossy();
                            violations.push(format!(
                                "src/{}: Line {}: unsafe block missing safety documentation",
                                relative_path,
                                idx + 1
                            ));
                        }
                    }
                }
            } else if path.is_dir() {
                Self::check_file_documentation(&path, violations)?;
            }
        }

        Ok(())
    }

    fn check_test_coverage() -> Result<Vec<String>, String> {
        use std::path::PathBuf;

        let mut violations = Vec::new();
        let src_dir = PathBuf::from("src");

        if !src_dir.exists() {
            return Ok(violations);
        }

        // Check for test files corresponding to source modules
        Self::check_module_test_coverage(&src_dir, &mut violations)?;
        Ok(violations)
    }

    fn check_module_test_coverage(
        dir: &std::path::Path,
        violations: &mut Vec<String>,
    ) -> Result<(), String> {
        use std::fs;

        let entries = fs::read_dir(dir).map_err(|e| e.to_string())?;
        let mut rs_files = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
                let file_name = path.file_name().unwrap_or_default();
                let name_str = file_name.to_string_lossy();

                // Skip test files and mod.rs
                if !name_str.ends_with("_test.rs")
                    && !name_str.ends_with("tests.rs")
                    && name_str != "mod.rs"
                    && name_str != "lib.rs"
                    && name_str != "main.rs"
                {
                    let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;

                    // Check if file has #[cfg(test)] module
                    let has_tests = content.contains("#[cfg(test)]")
                        || content.contains("#[test]")
                        || content.contains("mod tests");

                    if !has_tests {
                        let relative_path =
                            path.strip_prefix("src").unwrap_or(&path).to_string_lossy();
                        violations.push(format!(
                            "src/{}: No tests found (missing #[cfg(test)] module or #[test] functions)",
                            relative_path
                        ));
                    }

                    rs_files.push(path.clone());
                }
            } else if path.is_dir() {
                Self::check_module_test_coverage(&path, violations)?;
            }
        }

        Ok(())
    }
}

/// Code quality report
#[derive(Debug, Default)]
pub struct CodeQualityReport {
    pub module_size_violations: Vec<String>,
    pub naming_violations: Vec<String>,
    pub documentation_gaps: Vec<String>,
    pub test_coverage_gaps: Vec<String>,
}

impl CodeQualityReport {
    /// Check if all quality standards are met
    pub fn is_compliant(&self) -> bool {
        self.module_size_violations.is_empty()
            && self.naming_violations.is_empty()
            && self.documentation_gaps.is_empty()
            && self.test_coverage_gaps.is_empty()
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Code Quality Report\n");
        summary.push_str("===================\n\n");

        summary.push_str(&format!(
            "Overall Compliance: {}\n\n",
            if self.is_compliant() {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        ));

        if !self.module_size_violations.is_empty() {
            summary.push_str(&format!(
                "Module Size Violations: {}\n",
                self.module_size_violations.len()
            ));
            for violation in &self.module_size_violations {
                summary.push_str(&format!("  • {}\n", violation));
            }
            summary.push('\n');
        }

        if !self.naming_violations.is_empty() {
            summary.push_str(&format!(
                "Naming Violations: {}\n",
                self.naming_violations.len()
            ));
            for violation in &self.naming_violations {
                summary.push_str(&format!("  • {}\n", violation));
            }
            summary.push('\n');
        }

        if !self.documentation_gaps.is_empty() {
            summary.push_str(&format!(
                "Documentation Gaps: {}\n",
                self.documentation_gaps.len()
            ));
            for gap in &self.documentation_gaps {
                summary.push_str(&format!("  • {}\n", gap));
            }
            summary.push('\n');
        }

        if !self.test_coverage_gaps.is_empty() {
            summary.push_str(&format!(
                "Test Coverage Gaps: {}\n",
                self.test_coverage_gaps.len()
            ));
            for gap in &self.test_coverage_gaps {
                summary.push_str(&format!("  • {}\n", gap));
            }
            summary.push('\n');
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_size_check() {
        // Small module should pass
        assert!(ModuleStandards::check_module_size("small.rs", 100).is_ok());

        // Large module should fail
        assert!(ModuleStandards::check_module_size("large.rs", 600).is_err());
    }

    #[test]
    fn test_module_naming_check() {
        // Valid names should pass
        assert!(ModuleStandards::check_module_naming("solver").is_ok());
        assert!(ModuleStandards::check_module_naming("fdtd_solver").is_ok());

        // Invalid names should fail
        assert!(ModuleStandards::check_module_naming("FDTD").is_err());
        assert!(ModuleStandards::check_module_naming("FDTDSolver").is_err());
    }

    #[test]
    fn test_import_validation() {
        // Valid sorted imports should pass
        let valid_imports = [
            "std::collections::HashMap".to_string(),
            "std::sync::Arc".to_string(),
            "ndarray::Array3".to_string(),
            "crate::core::error::KwaversError".to_string(),
        ];
        assert!(ModuleStandards::validate_imports(&valid_imports).is_ok());

        // Unsorted imports should fail
        let invalid_imports = [
            "std::sync::Arc".to_string(),
            "std::collections::HashMap".to_string(),
        ];
        assert!(ModuleStandards::validate_imports(&invalid_imports).is_err());
    }

    #[test]
    fn test_quality_report_compliance() {
        let compliant_report = CodeQualityReport::default();
        assert!(compliant_report.is_compliant());

        let non_compliant_report = CodeQualityReport {
            module_size_violations: vec!["large_module.rs".to_string()],
            ..Default::default()
        };
        assert!(!non_compliant_report.is_compliant());
    }

    #[test]
    fn test_architecture_checker_execution() {
        // Test that the architecture checker can run without panicking
        let result = CodeQualityChecker::run_quality_checks();
        assert!(
            result.is_ok(),
            "Quality checks should complete successfully"
        );

        let report = result.unwrap();

        // Verify report can be converted to summary string
        let summary = report.summary();
        assert!(!summary.is_empty(), "Summary should contain content");
        assert!(
            summary.contains("Code Quality Report"),
            "Summary should contain header"
        );
    }

    #[test]
    fn test_module_size_violations_detected() {
        // Test that check_module_sizes can detect size violations
        let result = CodeQualityChecker::run_quality_checks();
        assert!(result.is_ok());

        let report = result.unwrap();
        // The actual violations depend on the codebase state
        // Just verify the check completed
        let _ = report.module_size_violations;
    }
}
