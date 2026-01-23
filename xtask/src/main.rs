//! Kwavers automation tasks
//!
//! Provides automated module size checks, naming audits, test generation,
//! and complexity analysis following the problem statement requirements.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

mod architecture;
use architecture::validate_architecture;

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "Kwavers automation tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check module sizes against GRASP principles (<300 lines)
    CheckModules,
    /// Audit naming conventions for neutral naming
    AuditNaming,
    /// Check for stub implementations and placeholders
    /// TODO_AUDIT: P2 - Advanced Build Infrastructure - Implement comprehensive code quality and architecture enforcement tools
    /// DEPENDS ON: xtask/code_quality.rs, xtask/architecture/complexity_analysis.rs, xtask/testing/property_based.rs
    /// MISSING: Cyclomatic complexity analysis with McCabe metrics
    /// MISSING: Architecture violation detection with graph algorithms
    /// MISSING: Property-based testing integration with QuickCheck/Proptest
    /// MISSING: Code coverage analysis with branch coverage metrics
    /// MISSING: Performance regression testing with statistical analysis
    /// THEOREM: McCabe cyclomatic complexity: M = E - N + 2P (edges - nodes + connected components)
    /// THEOREM: Halstead complexity: Effort = (n1 + n2) log₂(n1 + n2) / (2 n2) for maintainability
    CheckStubs,
    /// Count configuration structs for SSOT violations
    CheckConfigs,
    /// Generate quality metrics report
    Metrics,
    /// Fix all automated issues
    Fix,
    /// Check architecture for layer violations and cross-contamination
    CheckArchitecture {
        /// Generate markdown report
        #[arg(long)]
        markdown: bool,
        /// Fail with non-zero exit code on violations
        #[arg(long)]
        strict: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::CheckModules => check_module_sizes(),
        Command::AuditNaming => audit_naming(),
        Command::CheckStubs => check_stubs(),
        Command::CheckConfigs => check_configs(),
        Command::Metrics => generate_metrics(),
        Command::Fix => fix_all(),
        Command::CheckArchitecture { markdown, strict } => check_architecture(markdown, strict),
    }
}

fn workspace_root() -> PathBuf {
    let xtask_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    xtask_dir
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| xtask_dir)
}

fn src_root() -> PathBuf {
    workspace_root().join("src")
}

/// Check module sizes against GRASP principles (<500 lines)
fn check_module_sizes() -> Result<()> {
    println!("🔍 Checking module sizes (GRASP: <500 lines)...");

    let mut violations = Vec::new();

    for entry in WalkDir::new(src_root()).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().map_or(false, |ext| ext == "rs") {
            let content = fs::read_to_string(entry.path())
                .with_context(|| format!("Failed to read {}", entry.path().display()))?;
            let line_count = content.lines().count();

            if line_count > 500 {
                violations.push((entry.path().to_path_buf(), line_count));
            }
        }
    }

    if violations.is_empty() {
        println!("✅ All modules comply with GRASP (<500 lines)");
    } else {
        println!("❌ {} modules exceed 500-line limit:", violations.len());
        for (path, lines) in &violations {
            println!("  {} ({} lines)", path.display(), lines);
        }
    }

    Ok(())
}

/// Audit naming conventions for neutral naming
fn audit_naming() -> Result<()> {
    println!("🔍 Auditing naming conventions...");

    let mut violations = Vec::new();

    // Patterns to detect - using word boundaries for precision
    let bad_patterns = [
        "_old",
        "_new",
        "_refactored",
        "_proper",
        "_enhanced",
        "_fixed",
        "_corrected",
        "_updated",
        "_improved",
    ];

    // Legitimate domain terms that contain pattern substrings but are valid
    let allowed_terms = [
        "temperature",    // Contains _temp but is domain term
        "temporal",       // Contains _temp but is domain term
        "tempered",       // Contains _temp but is domain term
        "properties",     // Contains _proper but is valid when not _proper suffix
        "property_based", // Contains _proper but is valid module name
    ];

    for entry in WalkDir::new(src_root()).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().map_or(false, |ext| ext == "rs") {
            let content = fs::read_to_string(entry.path())
                .with_context(|| format!("Failed to read {}", entry.path().display()))?;

            for (line_num, line) in content.lines().enumerate() {
                // Skip comments
                if line.trim_start().starts_with("//") {
                    continue;
                }

                for pattern in &bad_patterns {
                    if !line.contains(pattern) {
                        continue;
                    }

                    // Check if this is a legitimate domain term
                    let is_allowed = allowed_terms.iter().any(|term| line.contains(term));

                    if is_allowed {
                        continue;
                    }

                    // Check for word boundaries - pattern should be isolated or at end of identifier
                    // Valid violations: variable_old, x_new, was_corrected
                    // Invalid (false positives): temperature, temporal, properties
                    let lower_line = line.to_lowercase();

                    // Look for the pattern with word boundaries
                    if let Some(pos) = lower_line.find(pattern) {
                        let before = if pos > 0 {
                            lower_line.chars().nth(pos - 1)
                        } else {
                            None
                        };
                        let after = lower_line.chars().nth(pos + pattern.len());

                        // Check if pattern is at a word boundary
                        // Valid if preceded by letter/number/underscore and followed by non-letter
                        let is_word_boundary =
                            matches!(before, Some('_') | Some(' ') | Some('(') | Some(',') | None)
                                || matches!(
                                    after,
                                    Some(' ')
                                        | Some(')')
                                        | Some(',')
                                        | Some(';')
                                        | Some(':')
                                        | None
                                );

                        if is_word_boundary {
                            violations.push((
                                entry.path().to_path_buf(),
                                line_num + 1,
                                pattern.to_string(),
                                line.trim().to_string(),
                            ));
                            break; // Only report once per line
                        }
                    }
                }
            }
        }
    }

    if violations.is_empty() {
        println!("✅ All naming follows neutral conventions");
    } else {
        println!("❌ {} naming violations found:", violations.len());
        for (path, line, pattern, code) in &violations {
            println!("  {}:{} - '{}': {}", path.display(), line, pattern, code);
        }
    }

    Ok(())
}

/// Check for stub implementations and placeholders
fn check_stubs() -> Result<()> {
    println!("🔍 Checking for stub implementations...");

    let mut violations = Vec::new();
    let stub_patterns = [
        "TODO",
        "FIXME",
        "todo!",
        "unimplemented!",
        "panic!",
        "unreachable!",
        "stub",
        "placeholder",
        "not implemented",
    ];

    for entry in WalkDir::new(src_root()).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().map_or(false, |ext| ext == "rs") {
            // Skip xtask and tool files to avoid false positives
            if entry.path().to_string_lossy().contains("xtask")
                || entry.path().to_string_lossy().contains("main.rs")
            {
                continue;
            }

            let content = fs::read_to_string(entry.path())
                .with_context(|| format!("Failed to read {}", entry.path().display()))?;

            for (line_num, line) in content.lines().enumerate() {
                for pattern in &stub_patterns {
                    if line.contains(pattern)
                        && !line.trim_start().starts_with("//")  // Skip comments
                        && !line.contains("\"")                 // Skip string literals
                        && !line.contains("&amp;")                 // Skip HTML entities
                        && line.contains(pattern)
                    // Actual usage
                    {
                        violations.push((
                            entry.path().to_path_buf(),
                            line_num + 1,
                            pattern.to_string(),
                            line.trim().to_string(),
                        ));
                    }
                }
            }
        }
    }

    if violations.is_empty() {
        println!("✅ No stub implementations found");
    } else {
        println!("❌ {} stub implementations found:", violations.len());
        for (path, line, pattern, code) in &violations {
            println!("  {}:{} - '{}': {}", path.display(), line, pattern, code);
        }
    }

    Ok(())
}

/// Count configuration structs for SSOT violations
fn check_configs() -> Result<()> {
    println!("🔍 Checking configuration structs (SSOT compliance)...");

    let mut config_count = 0;
    let mut configs = Vec::new();

    for entry in WalkDir::new(src_root()).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().map_or(false, |ext| ext == "rs") {
            let content = fs::read_to_string(entry.path())
                .with_context(|| format!("Failed to read {}", entry.path().display()))?;

            for (line_num, line) in content.lines().enumerate() {
                if line.contains("struct") && line.contains("Config") {
                    config_count += 1;
                    configs.push((
                        entry.path().to_path_buf(),
                        line_num + 1,
                        line.trim().to_string(),
                    ));
                }
            }
        }
    }

    println!("📊 Found {} Config structs:", config_count);
    for (path, line, code) in &configs {
        println!("  {}:{} - {}", path.display(), line, code);
    }

    if config_count > 10 {
        println!(
            "⚠️  {} Config structs may violate SSOT principle",
            config_count
        );
    } else {
        println!("✅ Config struct count within reasonable limits");
    }

    Ok(())
}

/// Check architecture for layer violations and cross-contamination
fn check_architecture(markdown: bool, strict: bool) -> Result<()> {
    let src_root = src_root();

    if !src_root.exists() {
        eprintln!("❌ Source directory not found: {}", src_root.display());
        std::process::exit(1);
    }

    println!("🔍 Validating architecture...\n");

    let report = validate_architecture(src_root)
        .map_err(|e| anyhow::anyhow!("Failed to validate architecture: {}", e))?;

    if markdown {
        // Generate markdown report
        let md = report.to_markdown();
        let output_path = workspace_root().join("ARCHITECTURE_VALIDATION_REPORT.md");
        fs::write(&output_path, md).context("Failed to write markdown report")?;
        println!("📄 Markdown report written to: {}", output_path.display());
    } else {
        // Print to console
        report.print();
    }

    if strict && !report.is_clean() {
        std::process::exit(report.exit_code());
    }

    Ok(())
}

/// Generate comprehensive quality metrics
fn generate_metrics() -> Result<()> {
    println!("📊 Generating quality metrics...");

    check_module_sizes()?;
    println!();
    audit_naming()?;
    println!();
    check_stubs()?;
    println!();
    check_configs()?;

    Ok(())
}

/// Fix all automated issues
fn fix_all() -> Result<()> {
    println!("🔧 Running automated fixes...");

    // For now, just run metrics
    // TODO: Implement automated fixes for Debug derives, etc.
    generate_metrics()?;

    println!("\n🎯 Manual fixes required:");
    println!("  - Refactor oversized modules");
    println!("  - Consolidate Config structs");
    println!("  - Add missing Debug derives");

    Ok(())
}
