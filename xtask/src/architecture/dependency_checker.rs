//! Dependency Checker - Enforces Deep Vertical Hierarchy
//!
//! This module validates that the codebase follows the strict layered architecture
//! defined in the architectural audit. It prevents cross-contamination by ensuring
//! that dependencies only flow bottom-up through the layer stack.
//!
//! ## Architecture Layers (Bottom to Top)
//!
//! ```text
//! Layer 0: core       - Primitives (errors, constants, time)
//! Layer 1: math       - Mathematical primitives and numerics
//! Layer 2: domain     - Domain primitives (grid, medium, boundary, sources)
//! Layer 3: infra      - Infrastructure (IO, runtime, API building blocks)
//! Layer 4: physics    - Physics models
//! Layer 5: solver     - Numerical solvers
//! Layer 6: simulation - Simulation orchestration
//! Layer 7: analysis   - Post-processing and algorithms
//! Layer 8: clinical   - Clinical applications and workflows
//! Layer 9: gpu        - Hardware acceleration (optional)
//! ```
//!
//! ## Rules
//!
//! 1. Lower layers NEVER import from higher layers
//! 2. Each layer may import from layers 0 through N-1 (where N is current layer)
//! 3. `core` is accessible to all layers
//! 4. Optional features (gpu) cannot be required by core functionality

use std::fs;
use std::path::{Path, PathBuf};

/// Architecture layer definition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Layer {
    Core = 0,
    Math = 1,
    Domain = 2,
    Infra = 3,
    Physics = 4,
    Solver = 5,
    Simulation = 6,
    Analysis = 7,
    Clinical = 8,
    Gpu = 9,
}

impl Layer {
    /// Get layer from module path
    pub fn from_module_path(path: &str) -> Option<Self> {
        let components: Vec<&str> = path.split("::").collect();
        if components.len() < 2 || components[0] != "crate" {
            return None;
        }

        match components[1] {
            "core" => Some(Layer::Core),
            "math" => Some(Layer::Math),
            "domain" => Some(Layer::Domain),
            "infra" => Some(Layer::Infra),
            "physics" => Some(Layer::Physics),
            "solver" => Some(Layer::Solver),
            "simulation" => Some(Layer::Simulation),
            "analysis" => Some(Layer::Analysis),
            "clinical" => Some(Layer::Clinical),
            "gpu" => Some(Layer::Gpu),
            _ => None,
        }
    }

    /// Get layer name
    pub fn name(&self) -> &'static str {
        match self {
            Layer::Core => "core",
            Layer::Math => "math",
            Layer::Domain => "domain",
            Layer::Infra => "infra",
            Layer::Physics => "physics",
            Layer::Solver => "solver",
            Layer::Simulation => "simulation",
            Layer::Analysis => "analysis",
            Layer::Clinical => "clinical",
            Layer::Gpu => "gpu",
        }
    }

    /// Check if this layer can depend on another layer
    pub fn can_depend_on(&self, other: &Layer) -> bool {
        // Core is accessible to all
        if *other == Layer::Core {
            return true;
        }

        // GPU is optional and can depend on most layers
        if *self == Layer::Gpu {
            return *other == Layer::Gpu || *other <= Layer::Clinical;
        }

        (*self as u8) >= (*other as u8)
    }
}

/// A dependency violation
#[derive(Debug, Clone)]
pub struct Violation {
    pub file: PathBuf,
    pub line: usize,
    pub from_layer: Layer,
    pub to_layer: Layer,
    pub import_statement: String,
}

impl Violation {
    pub fn format(&self) -> String {
        format!(
            "{}:{}: Layer {} cannot depend on layer {}\n  Import: {}",
            self.file.display(),
            self.line,
            self.from_layer.name(),
            self.to_layer.name(),
            self.import_statement
        )
    }
}

/// Dependency checker
pub struct DependencyChecker {
    src_root: PathBuf,
    violations: Vec<Violation>,
}

impl DependencyChecker {
    /// Create a new dependency checker
    pub fn new(src_root: PathBuf) -> Self {
        Self {
            src_root,
            violations: Vec::new(),
        }
    }

    /// Check all files in the source tree
    pub fn check_all(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.violations.clear();
        self.check_directory(&self.src_root.clone())?;
        Ok(())
    }

    /// Recursively check a directory
    fn check_directory(&mut self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                self.check_directory(&path)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                self.check_file(&path)?;
            }
        }

        Ok(())
    }

    /// Check a single file for violations
    fn check_file(&mut self, file_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(file_path)?;

        // Determine the layer of this file
        let relative_path = file_path.strip_prefix(&self.src_root).unwrap_or(file_path);
        let path_str = relative_path.to_string_lossy();
        let components: Vec<&str> = path_str.split(std::path::MAIN_SEPARATOR).collect();

        if components.is_empty() {
            return Ok(());
        }

        let file_layer = match components[0] {
            "core" => Layer::Core,
            "infra" => Layer::Infra,
            "domain" => Layer::Domain,
            "math" => Layer::Math,
            "physics" => Layer::Physics,
            "solver" => Layer::Solver,
            "simulation" => Layer::Simulation,
            "clinical" => Layer::Clinical,
            "analysis" => Layer::Analysis,
            "gpu" => Layer::Gpu,
            _ => return Ok(()), // Not a layer module
        };

        // Check each use statement
        for (line_num, line) in content.lines().enumerate() {
            if let Some(import_layer) = self.extract_import_layer(line) {
                if !file_layer.can_depend_on(&import_layer) {
                    self.violations.push(Violation {
                        file: file_path.to_path_buf(),
                        line: line_num + 1,
                        from_layer: file_layer,
                        to_layer: import_layer,
                        import_statement: line.trim().to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Extract the layer from an import statement
    fn extract_import_layer(&self, line: &str) -> Option<Layer> {
        let trimmed = line.trim();

        if !trimmed.starts_with("use ") {
            return None;
        }

        // Extract the module path from the use statement
        // Examples:
        //   use crate::domain::grid::Grid;
        //   use crate::solver::forward::fdtd;
        //   use super::super::physics;

        if trimmed.contains("crate::") {
            // Find first component after "crate::"
            if let Some(start) = trimmed.find("crate::") {
                let rest = &trimmed[start + 7..]; // Skip "crate::"
                if let Some(end) = rest.find("::").or_else(|| rest.find(";")) {
                    let module = &rest[..end];
                    return Layer::from_module_path(&format!("crate::{module}"));
                }
            }
        }

        None
    }

    /// Get all violations found
    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }

}

/// Check for cross-contamination patterns
pub struct CrossContaminationDetector {
    src_root: PathBuf,
    patterns: Vec<ContaminationPattern>,
}

#[derive(Debug)]
pub struct ContaminationPattern {
    pub name: String,
    pub primary_location: String,
    pub contaminated_locations: Vec<String>,
    pub severity: Severity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Medium,
    High,
    Critical,
}

impl Severity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Medium => "MEDIUM",
            Severity::High => "HIGH",
            Severity::Critical => "CRITICAL",
        }
    }
}

impl CrossContaminationDetector {
    pub fn new(src_root: PathBuf) -> Self {
        let patterns = vec![
            ContaminationPattern {
                name: "Grid Operations".to_string(),
                primary_location: "domain/grid/".to_string(),
                contaminated_locations: vec![
                    "solver/forward/axisymmetric/coordinates.rs".to_string(),
                    "solver/forward/fdtd/numerics/staggered_grid.rs".to_string(),
                    "math/numerics/operators/differential.rs".to_string(),
                ],
                severity: Severity::High,
            },
            ContaminationPattern {
                name: "Boundary Conditions".to_string(),
                primary_location: "domain/boundary/".to_string(),
                contaminated_locations: vec![
                    "solver/utilities/cpml_integration.rs".to_string(),
                    "solver/forward/fdtd/numerics/boundary_stencils.rs".to_string(),
                ],
                severity: Severity::High,
            },
            ContaminationPattern {
                name: "Beamforming Algorithms".to_string(),
                primary_location: "analysis/signal_processing/beamforming/".to_string(),
                contaminated_locations: vec![
                    "domain/sensor/beamforming/".to_string(),
                    "domain/source/transducers/phased_array/beamforming.rs".to_string(),
                    "core/utils/sparse_matrix/beamforming.rs".to_string(),
                ],
                severity: Severity::High,
            },
            ContaminationPattern {
                name: "Medium Properties".to_string(),
                primary_location: "domain/medium/".to_string(),
                contaminated_locations: vec![
                    "solver/forward/axisymmetric/config.rs".to_string(),
                    "physics/acoustics/".to_string(),
                ],
                severity: Severity::Critical,
            },
        ];

        Self { src_root, patterns }
    }

    /// Check for contamination patterns
    pub fn check(&self) -> Vec<DetectedContamination> {
        let mut detected = Vec::new();

        for pattern in &self.patterns {
            let mut found_locations = Vec::new();

            for contaminated in &pattern.contaminated_locations {
                let path = self.src_root.join(contaminated);
                if path.exists() {
                    found_locations.push(contaminated.clone());
                }
            }

            if !found_locations.is_empty() {
                detected.push(DetectedContamination {
                    pattern_name: pattern.name.clone(),
                    severity: pattern.severity,
                    primary: pattern.primary_location.clone(),
                    contaminated: found_locations,
                });
            }
        }

        detected
    }

    /// Print contamination report
    pub fn print_report(&self, detected: &[DetectedContamination]) {
        if detected.is_empty() {
            println!("✅ No cross-contamination patterns detected!");
            return;
        }

        println!("⚠️  Cross-Contamination Analysis:\n");

        for contamination in detected {
            println!(
                "[{}] {} - Found in {} locations",
                contamination.severity.as_str(),
                contamination.pattern_name,
                contamination.contaminated.len()
            );
            println!("  Primary: {}", contamination.primary);
            println!("  Contaminated:");
            for location in &contamination.contaminated {
                println!("    - {}", location);
            }
            println!();
        }
    }
}

#[derive(Debug)]
pub struct DetectedContamination {
    pub pattern_name: String,
    pub severity: Severity,
    pub primary: String,
    pub contaminated: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_ordering() {
        assert!(Layer::Core < Layer::Math);
        assert!(Layer::Math < Layer::Domain);
        assert!(Layer::Domain < Layer::Infra);
        assert!(Layer::Infra < Layer::Physics);
        assert!(Layer::Physics < Layer::Solver);
        assert!(Layer::Solver < Layer::Simulation);
        assert!(Layer::Simulation < Layer::Analysis);
        assert!(Layer::Analysis < Layer::Clinical);
    }

    #[test]
    fn test_layer_dependencies() {
        // Domain can depend on Core
        assert!(Layer::Domain.can_depend_on(&Layer::Core));
        assert!(Layer::Domain.can_depend_on(&Layer::Math));

        // Solver can depend on Domain and Math
        assert!(Layer::Solver.can_depend_on(&Layer::Domain));
        assert!(Layer::Solver.can_depend_on(&Layer::Math));

        // Domain cannot depend on Solver
        assert!(!Layer::Domain.can_depend_on(&Layer::Solver));

        // Core cannot depend on anything except itself
        assert!(!Layer::Core.can_depend_on(&Layer::Domain));

        // GPU can depend on itself
        assert!(Layer::Gpu.can_depend_on(&Layer::Gpu));
    }

    #[test]
    fn test_extract_import_layer() {
        let checker = DependencyChecker::new(PathBuf::from("src"));

        assert_eq!(
            checker.extract_import_layer("use crate::domain::grid::Grid;"),
            Some(Layer::Domain)
        );

        assert_eq!(
            checker.extract_import_layer("use crate::solver::forward::fdtd::FdtdSolver;"),
            Some(Layer::Solver)
        );

        assert_eq!(
            checker.extract_import_layer("use std::collections::HashMap;"),
            None
        );
    }

    #[test]
    fn test_layer_from_module_path() {
        assert_eq!(
            Layer::from_module_path("crate::domain::grid"),
            Some(Layer::Domain)
        );

        assert_eq!(
            Layer::from_module_path("crate::solver::forward::fdtd"),
            Some(Layer::Solver)
        );

        assert_eq!(Layer::from_module_path("std::collections"), None);
    }
}
