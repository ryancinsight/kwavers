//! Architecture Validation Module
//!
//! This module provides tools to enforce and validate the deep vertical hierarchy
//! architecture of the kwavers codebase. It prevents cross-contamination between
//! layers and ensures that dependencies flow strictly bottom-up.
//!
//! ## Components
//!
//! - `dependency_checker`: Validates layer dependencies
//! - `cross_contamination`: Detects known contamination patterns
//! - Reports: Generation of architecture audit reports
//!
//! ## Usage
//!
//! ```bash
//! cargo xtask check-architecture
//! ```

pub mod dependency_checker;

pub use dependency_checker::{
    CrossContaminationDetector, DependencyChecker, DetectedContamination, Severity, Violation,
};

use std::path::PathBuf;

/// Run complete architecture validation
pub fn validate_architecture(
    src_root: PathBuf,
) -> Result<ArchitectureReport, Box<dyn std::error::Error>> {
    let report_src_root = src_root.clone();
    let mut checker = DependencyChecker::new(src_root.clone());
    checker.check_all()?;

    let contamination_detector = CrossContaminationDetector::new(src_root);
    let contamination = contamination_detector.check();

    Ok(ArchitectureReport {
        src_root: report_src_root,
        dependency_violations: checker.violations().to_vec(),
        cross_contamination: contamination,
    })
}

/// Complete architecture validation report
#[derive(Debug)]
pub struct ArchitectureReport {
    src_root: PathBuf,
    pub dependency_violations: Vec<Violation>,
    pub cross_contamination: Vec<DetectedContamination>,
}

impl ArchitectureReport {
    /// Check if architecture is clean
    pub fn is_clean(&self) -> bool {
        self.dependency_violations.is_empty() && self.cross_contamination.is_empty()
    }

    /// Get overall severity
    pub fn max_severity(&self) -> Option<Severity> {
        self.cross_contamination
            .iter()
            .map(|c| c.severity)
            .max_by_key(|s| *s as u8)
    }

    /// Print comprehensive report
    pub fn print(&self) {
        println!("═══════════════════════════════════════════════════════");
        println!("  KWAVERS ARCHITECTURE VALIDATION REPORT");
        println!("═══════════════════════════════════════════════════════\n");

        // Dependency violations
        if self.dependency_violations.is_empty() {
            println!("✅ Layer Dependencies: CLEAN");
        } else {
            println!(
                "❌ Layer Dependencies: {} VIOLATION(S)",
                self.dependency_violations.len()
            );

            // Print violations directly
            for v in &self.dependency_violations {
                println!("  {}", v.format());
            }
        }

        println!("───────────────────────────────────────────────────────\n");

        // Cross-contamination
        if self.cross_contamination.is_empty() {
            println!("✅ Cross-Contamination: NONE DETECTED");
        } else {
            println!(
                "⚠️  Cross-Contamination: {} PATTERN(S) DETECTED",
                self.cross_contamination.len()
            );

            let detector = CrossContaminationDetector::new(self.src_root.clone());
            detector.print_report(&self.cross_contamination);
        }

        println!("═══════════════════════════════════════════════════════");

        if self.is_clean() {
            println!("✅ OVERALL STATUS: CLEAN");
        } else {
            let severity = self.max_severity().unwrap_or(Severity::Medium);
            println!(
                "❌ OVERALL STATUS: VIOLATIONS DETECTED ({})",
                severity.as_str()
            );
            println!("\nRefer to ARCHITECTURAL_AUDIT_SPRINT_ANALYSIS.md for");
            println!("detailed refactoring plan and migration strategy.");
        }

        println!("═══════════════════════════════════════════════════════\n");
    }

    /// Generate markdown report
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("# Architecture Validation Report\n\n");
        md.push_str(&format!(
            "**Generated**: {}\n\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        ));

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str(&format!(
            "- **Layer Dependency Violations**: {}\n",
            self.dependency_violations.len()
        ));
        md.push_str(&format!(
            "- **Cross-Contamination Patterns**: {}\n",
            self.cross_contamination.len()
        ));
        md.push_str(&format!(
            "- **Overall Status**: {}\n\n",
            if self.is_clean() {
                "✅ CLEAN"
            } else {
                "❌ VIOLATIONS DETECTED"
            }
        ));

        // Dependency violations detail
        if !self.dependency_violations.is_empty() {
            md.push_str("## Layer Dependency Violations\n\n");
            md.push_str(&format!(
                "Found {} violations of the layered architecture:\n\n",
                self.dependency_violations.len()
            ));

            for v in &self.dependency_violations {
                md.push_str(&format!(
                    "- `{}:{}` - {} → {} : `{}`\n",
                    v.file.display(),
                    v.line,
                    v.from_layer.name(),
                    v.to_layer.name(),
                    v.import_statement
                ));
            }
            md.push_str("\n");
        }

        // Cross-contamination detail
        if !self.cross_contamination.is_empty() {
            md.push_str("## Cross-Contamination Patterns\n\n");

            for c in &self.cross_contamination {
                md.push_str(&format!(
                    "### {} [{}]\n\n",
                    c.pattern_name,
                    c.severity.as_str()
                ));
                md.push_str(&format!("**Primary Location**: `{}`\n\n", c.primary));
                md.push_str("**Contaminated Locations**:\n\n");
                for loc in &c.contaminated {
                    md.push_str(&format!("- `{}`\n", loc));
                }
                md.push_str("\n");
            }
        }

        md
    }

    /// Exit code based on severity
    pub fn exit_code(&self) -> i32 {
        if self.is_clean() {
            0
        } else if self.max_severity() == Some(Severity::Critical) {
            2
        } else {
            1
        }
    }
}
