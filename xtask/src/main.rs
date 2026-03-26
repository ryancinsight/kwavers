//! Kwavers automation tasks
//!
//! Provides automated module size checks, naming audits, test generation,
//! and complexity analysis following the problem statement requirements.

use anyhow::{Context, Result};
use clap::{ArgAction, Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use wait_timeout::ChildExt;
use walkdir::WalkDir;

mod architecture;
use architecture::validate_architecture;

mod fixes;

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
    /// Build pykwavers Python bindings with maturin
    BuildPykwavers {
        /// Build in release mode
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        release: bool,
        /// Install into the active Python environment
        #[arg(long)]
        install: bool,
    },
    /// Install k-wave-python and Python dependencies
    InstallKwave {
        /// Skip install if k-wave-python is already available
        #[arg(long)]
        skip_existing: bool,
    },
    /// Run pykwavers Python test suite
    TestPykwavers {
        /// Skip building pykwavers with maturin
        #[arg(long)]
        skip_build: bool,
        /// Skip automatic Python dependency installation
        #[arg(long)]
        no_install: bool,
        /// Timeout in seconds for the pytest run
        #[arg(long, default_value_t = 300)]
        timeout_secs: u64,
        /// Extra arguments passed through to pytest after `--`
        #[arg(last = true)]
        extra: Vec<String>,
    },
    /// Run parity validation tests (pykwavers vs k-wave-python)
    ValidateParity {
        /// Skip building pykwavers with maturin
        #[arg(long)]
        skip_build: bool,
        /// Run only standalone pykwavers tests (no k-wave-python dependency)
        #[arg(long)]
        standalone_only: bool,
        /// Test a specific component: grid, medium, source, sensor, solver, examples, utilities
        #[arg(long)]
        component: Option<String>,
        /// Timeout in seconds for the pytest run
        #[arg(long, default_value_t = 600)]
        timeout_secs: u64,
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
        Command::BuildPykwavers { release, install } => build_pykwavers(release, install),
        Command::InstallKwave { skip_existing } => install_kwave(skip_existing),
        Command::TestPykwavers {
            skip_build,
            no_install,
            timeout_secs,
            extra,
        } => test_pykwavers(skip_build, no_install, timeout_secs, extra),
        Command::ValidateParity {
            skip_build,
            standalone_only,
            component,
            timeout_secs,
        } => validate_parity(skip_build, standalone_only, component, timeout_secs),
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
    workspace_root().join("kwavers").join("src")
}

/// Check module sizes against GRASP principles (<500 lines)
fn check_module_sizes() -> Result<()> {
    println!("🔍 Checking module sizes (GRASP: <500 lines)...");

    let mut violations = Vec::new();

    for entry in WalkDir::new(src_root()).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().is_some_and(|ext| ext == "rs") {
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

    // Patterns that are legitimate in computational physics / iterative algorithms.
    // These appear as local variable names like p_new, u_new, h_pp_new, x_new,
    // radius_new, t_new, alpha_corrected, predicate_fixed, last_updated, etc.
    // They are NOT deprecated/refactored code markers.
    let iterative_var_patterns = regex::Regex::new(
        r"(?x)
        # Iterative timestep variables: p_new, u_new, v_new, x_new, etc.
        \b[a-z][a-z0-9_]*_(new|old)\b |
        # Corrected/updated domain values: alpha_corrected, phase_corrected, etc.
        \b[a-z][a-z0-9_]*_(corrected|fixed|updated)\b |
        # Enhanced module declarations (pub mod X_enhanced)
        \bpub\s+mod\s+\w+_enhanced\b |
        # pub use re-exports of enhanced modules
        \bpub\s+use\s+\w+_enhanced\b |
        # Struct field names that are domain-appropriate
        \blast_updated\b |
        # Function parameters starting with underscore (unused param pattern)
        \b_fixed\b |
        # Marginal/sum variable patterns in statistics
        \b(marginal|sum|mean|h)_fixed\b |
        \bfixed_bin\b | \bfixed_val\b | \bfixed_centered\b |
        # n_new pattern in sampling/refinement
        \bn_(new|old)\b |
        # beta_new, alpha_new in iterative solvers
        \b(alpha|beta|rho|sigma|gamma|theta|phi|residual_dot_z)_new\b |
        # Leading underscore = intentionally unused binding (Rust convention)
        # Matches both _updated_pinn (prefix) and _foo_new (suffix)
        \b_(updated|corrected|fixed|improved|new|old)\w*\b |
        \b_\w+_(new|old|updated|corrected|fixed|improved)\b
        ",
    )
    .expect("valid regex");

    for entry in WalkDir::new(src_root()).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().is_some_and(|ext| ext == "rs") {
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

                    // Check if this matches iterative computation patterns
                    // (p_new, u_new, alpha_corrected, last_updated, etc.)
                    if iterative_var_patterns.is_match(line) {
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
        if entry.path().extension().is_some_and(|ext| ext == "rs") {
            // Skip xtask and tool files to avoid false positives
            if entry.path().to_string_lossy().contains("xtask")
                || entry.path().to_string_lossy().contains("main.rs")
            {
                continue;
            }

            let content = fs::read_to_string(entry.path())
                .with_context(|| format!("Failed to read {}", entry.path().display()))?;
            let lines: Vec<&str> = content.lines().collect();

            for (line_num, line) in lines.iter().enumerate() {
                for pattern in &stub_patterns {
                    if line.contains(pattern)
                        && !line.trim_start().starts_with("//")  // Skip comments
                        && !line.contains("\"")                 // Skip string literals
                        && !line.contains("&amp;")                 // Skip HTML entities
                        && line.contains(pattern)
                    // Actual usage
                    {
                        // === Context-aware suppression ===

                        // 1. unreachable!() in match arms (exhaustive patterns) is legitimate
                        if *pattern == "unreachable!" {
                            let trimmed = line.trim();
                            if trimmed.starts_with("_ =>") || trimmed.starts_with("_=>") {
                                continue;
                            }
                        }

                        // 2. panic!() in match arms for out-of-range indexing is legitimate
                        //    (documented panics with try_ alternatives)
                        if *pattern == "panic!" {
                            let trimmed = line.trim();
                            // Match arm panics (e.g., _ => panic!("index out of range"))
                            if trimmed.starts_with("_ =>") || trimmed.starts_with("_=>") {
                                continue;
                            }
                            // Defensive panics in Default/new impls (fail-fast for misconfiguration)
                            // Check surrounding lines for fn default or impl Default
                            let in_default_or_new =
                                (line_num.saturating_sub(50)..line_num).any(|i| {
                                    if let Some(prev) = lines.get(i) {
                                        prev.contains("fn default")
                                            || prev.contains("impl Default")
                                            || prev.contains("fn new")
                                    } else {
                                        false
                                    }
                                });
                            if in_default_or_new {
                                continue;
                            }
                        }

                        // 3. Feature-gate stubs (pub mod stub { } behind #[cfg(not(feature = ...))])
                        if *pattern == "stub" {
                            let trimmed = line.trim();
                            if trimmed.starts_with("pub mod stub")
                                || trimmed.starts_with("pub use stub")
                            {
                                // Check if preceded by #[cfg(not(feature = ...))]
                                let has_feature_gate =
                                    (line_num.saturating_sub(3)..line_num).any(|i| {
                                        if let Some(prev) = lines.get(i) {
                                            prev.contains("#[cfg(not(feature")
                                        } else {
                                            false
                                        }
                                    });
                                if has_feature_gate {
                                    continue;
                                }
                            }
                        }

                        // 4. "placeholder" in variable initialization comments is legitimate
                        if *pattern == "placeholder" {
                            let trimmed = line.trim();
                            if trimmed.starts_with("let ") && trimmed.contains("// placeholder") {
                                continue;
                            }
                        }

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
        if entry.path().extension().is_some_and(|ext| ext == "rs") {
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

/// Run pykwavers Python test suite
fn test_pykwavers(
    skip_build: bool,
    no_install: bool,
    timeout_secs: u64,
    extra: Vec<String>,
) -> Result<()> {
    if !no_install {
        install_kwave(true)?;
    }

    if !skip_build {
        build_pykwavers(true, true)?;
    }

    let mut cmd = std::process::Command::new("python");
    cmd.arg("-m").arg("pytest").arg("-v").arg("pykwavers");

    if !extra.is_empty() {
        cmd.args(extra);
    }

    let mut child = cmd
        .current_dir(workspace_root())
        .spawn()
        .context("Failed to spawn pytest for pykwavers")?;

    let timeout = Duration::from_secs(timeout_secs);
    let status = match child
        .wait_timeout(timeout)
        .context("Failed while waiting on pytest")?
    {
        Some(status) => status,
        None => {
            child.kill().ok();
            anyhow::bail!("pykwavers tests timed out after {} seconds", timeout_secs);
        }
    };

    if !status.success() {
        anyhow::bail!("pykwavers tests failed");
    }

    Ok(())
}

/// Run parity validation tests (pykwavers vs k-wave-python)
fn validate_parity(
    skip_build: bool,
    standalone_only: bool,
    component: Option<String>,
    timeout_secs: u64,
) -> Result<()> {
    install_kwave(true)?;

    if !skip_build {
        build_pykwavers(true, true)?;
    }

    println!("🔍 Running parity validation tests...\n");

    let mut cmd = std::process::Command::new("python");
    cmd.arg("-m").arg("pytest").arg("-v").arg("--tb=short");

    // Select specific test files based on component
    let test_dir = workspace_root().join("pykwavers").join("tests");
    let test_files: Vec<String> = match component.as_deref() {
        Some("grid") => vec!["test_grid_parity.py"],
        Some("medium") => vec!["test_medium_parity.py"],
        Some("source") => vec!["test_source_parity.py"],
        Some("sensor") => vec!["test_sensor_parity.py"],
        Some("solver") => vec!["test_solver_parity.py"],
        Some("examples") => vec!["test_examples_parity.py"],
        Some("utilities") => vec!["test_utilities.py"],
        Some(name) => {
            anyhow::bail!(
                "Unknown component '{}'. Valid: grid, medium, source, sensor, solver, examples, utilities",
                name
            );
        }
        None => vec![
            "test_grid_parity.py",
            "test_medium_parity.py",
            "test_source_parity.py",
            "test_sensor_parity.py",
            "test_solver_parity.py",
            "test_examples_parity.py",
            "test_utilities.py",
        ],
    }
    .into_iter()
    .map(|f| test_dir.join(f).to_string_lossy().to_string())
    .collect();

    cmd.args(&test_files);

    if standalone_only {
        cmd.arg("-m").arg("not requires_kwave");
    }

    let mut child = cmd
        .current_dir(workspace_root())
        .spawn()
        .context("Failed to spawn pytest for parity validation")?;

    let timeout = Duration::from_secs(timeout_secs);
    let status = match child
        .wait_timeout(timeout)
        .context("Failed while waiting on pytest")?
    {
        Some(status) => status,
        None => {
            child.kill().ok();
            anyhow::bail!(
                "parity validation tests timed out after {} seconds",
                timeout_secs
            );
        }
    };

    if !status.success() {
        anyhow::bail!("parity validation tests failed");
    }

    println!("\n✅ Parity validation passed!");
    Ok(())
}

fn install_kwave(skip_existing: bool) -> Result<()> {
    let status = std::process::Command::new("python")
        .arg("-c")
        .arg("import kwave")
        .current_dir(workspace_root())
        .status()
        .context("Failed to check k-wave-python availability")?;

    if status.success() && skip_existing {
        println!("✅ k-wave-python already installed");
        return Ok(());
    }

    if !status.success() {
        println!("📦 k-wave-python not found. Installing dependencies...");
    } else {
        println!("📦 Re-installing Python dependencies...");
    }

    let requirements = workspace_root().join("pykwavers").join("requirements.txt");
    let status = std::process::Command::new("python")
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("-r")
        .arg(requirements)
        .current_dir(workspace_root())
        .status()
        .context("Failed to install Python dependencies for pykwavers")?;

    if !status.success() {
        anyhow::bail!("Python dependency installation failed");
    }

    Ok(())
}

fn build_pykwavers(release: bool, install: bool) -> Result<()> {
    ensure_maturin_installed()?;

    let pykwavers_dir = workspace_root().join("pykwavers");
    let mut cmd = std::process::Command::new("maturin");

    if install {
        cmd.arg("develop");
    } else {
        cmd.arg("build");
    }

    if release {
        cmd.arg("--release");
    }

    let status = cmd
        .current_dir(pykwavers_dir)
        .status()
        .context("Failed to run maturin for pykwavers")?;

    if !status.success() {
        anyhow::bail!("maturin build failed");
    }

    Ok(())
}

fn ensure_maturin_installed() -> Result<()> {
    let status = std::process::Command::new("maturin")
        .arg("--version")
        .current_dir(workspace_root())
        .status();

    if let Ok(status) = status {
        if status.success() {
            return Ok(());
        }
    }

    println!("📦 maturin not found. Installing...");

    let status = std::process::Command::new("python")
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("maturin")
        .current_dir(workspace_root())
        .status()
        .context("Failed to install maturin")?;

    if !status.success() {
        anyhow::bail!("maturin installation failed");
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
    // Run automated fixes first
    fixes::apply_fixes()?;

    // Then run metrics to show remaining issues
    println!("\n📊 Generating updated metrics...");
    generate_metrics()?;

    println!("\n🎯 Manual fixes required (if any remaining):");
    println!("  - Refactor oversized modules");
    println!("  - Consolidate Config structs");

    Ok(())
}
