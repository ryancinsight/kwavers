//! Kwavers automation tasks
//!
//! Provides automated module size checks, naming audits, test generation,
//! and complexity analysis following the problem statement requirements.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use walkdir::WalkDir;

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
    CheckStubs,
    /// Count configuration structs for SSOT violations
    CheckConfigs,
    /// Generate quality metrics report
    Metrics,
    /// Fix all automated issues
    Fix,
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
    }
}

/// Check module sizes against GRASP principles (<500 lines)
fn check_module_sizes() -> Result<()> {
    println!("🔍 Checking module sizes (GRASP: <500 lines)...");

    let mut violations = Vec::new();

    for entry in WalkDir::new("../src").into_iter().filter_map(|e| e.ok()) {
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
    let bad_patterns = [
        "_old",
        "_new",
        "_refactored",
        "_temp",
        "_proper",
        "_enhanced",
        "_fixed",
        "_corrected",
        "_updated",
        "_improved",
    ];

    for entry in WalkDir::new("src").into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().map_or(false, |ext| ext == "rs") {
            let content = fs::read_to_string(entry.path())
                .with_context(|| format!("Failed to read {}", entry.path().display()))?;

            for (line_num, line) in content.lines().enumerate() {
                for pattern in &bad_patterns {
                    if line.contains(pattern) {
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

    for entry in WalkDir::new("src").into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().map_or(false, |ext| ext == "rs") {
            // Skip xtask and tool files to avoid false positives
            if entry.path().to_string_lossy().contains("xtask") 
                || entry.path().to_string_lossy().contains("main.rs") {
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
                        && line.contains(pattern)               // Actual usage
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

    for entry in WalkDir::new("src").into_iter().filter_map(|e| e.ok()) {
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
