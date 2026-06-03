//! Python validation types and thresholds.
//!
//! Full cross-language parity is tested in pykwavers/tests/.
//! This module provides the type definitions and tolerance constants
//! used by `python_validation_integration_test.rs`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// A single Python parity validation result.
#[derive(Debug, Clone)]
pub struct PythonValidationResult {
    /// Human-readable test name.
    pub test_name: String,
    /// Whether the test passed its tolerance threshold.
    pub passed: bool,
    /// L2 norm of the error field.
    pub l2_error: f64,
    /// Relative error (L2 error / reference norm).
    pub relative_error: f64,
    /// Maximum pointwise error.
    pub max_error: f64,
    /// Diagnostic message for failures.
    pub diagnostics: String,
}

impl PythonValidationResult {
    /// Construct a passing validation result.
    pub fn passing(test_name: impl Into<String>, diagnostics: impl Into<String>) -> Self {
        Self {
            test_name: test_name.into(),
            passed: true,
            l2_error: 0.0,
            relative_error: 0.0,
            max_error: 0.0,
            diagnostics: diagnostics.into(),
        }
    }

    /// Construct a failing validation result.
    pub fn failing(
        test_name: impl Into<String>,
        relative_error: f64,
        diagnostics: impl Into<String>,
    ) -> Self {
        Self {
            test_name: test_name.into(),
            passed: false,
            l2_error: relative_error,
            relative_error,
            max_error: relative_error,
            diagnostics: diagnostics.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PytestSummary {
    passed: usize,
    failed: usize,
    skipped: usize,
    xfailed: usize,
    xpassed: usize,
    errors: usize,
}

impl PytestSummary {
    fn total(self) -> usize {
        self.passed + self.failed + self.skipped + self.xfailed + self.xpassed + self.errors
    }

    fn failure_ratio(self) -> f64 {
        let total = self.total();
        if total == 0 {
            1.0
        } else {
            (self.failed + self.errors + self.xpassed) as f64 / total as f64
        }
    }
}

#[derive(Debug, Clone)]
pub struct PytestValidationTarget {
    pub test_name: &'static str,
    pub args: &'static [&'static str],
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn parse_pytest_summary(output: &str) -> Option<(String, PytestSummary)> {
    let summary_line = output.lines().rev().find(|line| {
        line.contains(" in ") && (line.contains(" passed") || line.contains(" failed"))
    })?;

    let mut summary = PytestSummary::default();
    for part in summary_line.split(',') {
        let trimmed = part.trim();
        let mut pieces = trimmed.split_whitespace();
        let count = match pieces.next().and_then(|value| value.parse::<usize>().ok()) {
            Some(count) => count,
            None => continue,
        };
        let label = match pieces.next() {
            Some(label) => label,
            None => continue,
        };

        match label {
            "passed" => summary.passed = count,
            "failed" => summary.failed = count,
            "skipped" => summary.skipped = count,
            "xfailed" => summary.xfailed = count,
            "xpassed" => summary.xpassed = count,
            "error" | "errors" => summary.errors = count,
            _ => {}
        }
    }

    Some((summary_line.trim().to_string(), summary))
}

pub fn run_pytest_validation(target: &PytestValidationTarget) -> PythonValidationResult {
    let workspace = repo_root();
    let pykwavers_dir = workspace.join("pykwavers");

    // Prefer `uv run python -m pytest` from the pykwavers venv; fall back to
    // bare `python -m pytest` for CI environments where uv is not available.
    let (program, leading_args): (&str, &[&str]) = if Command::new("uv")
        .arg("--version")
        .current_dir(&pykwavers_dir)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        ("uv", &["run", "python", "-m", "pytest"])
    } else {
        ("python", &["-m", "pytest"])
    };

    let mut command = Command::new(program);
    for &a in leading_args {
        command.arg(a);
    }
    command
        .arg("-q")
        .arg("-o")
        .arg("addopts=")
        .env("KWAVERS_SKIP_KWAVE", "1")
        .env("MPLBACKEND", "Agg")
        .current_dir(&pykwavers_dir);

    // Rebase test paths relative to pykwavers dir when they start with "pykwavers/".
    for arg in target.args {
        let rebased = arg.strip_prefix("pykwavers/").unwrap_or(arg);
        command.arg(rebased);
    }

    match command.output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = if stderr.trim().is_empty() {
                stdout.to_string()
            } else if stdout.trim().is_empty() {
                stderr.to_string()
            } else {
                format!("{stdout}\n{stderr}")
            };

            let (summary_line, summary) = parse_pytest_summary(&combined).unwrap_or_else(|| {
                (
                    "pytest summary unavailable".to_string(),
                    PytestSummary::default(),
                )
            });
            let diagnostics = if output.status.success() {
                summary_line
            } else {
                let mut tail: Vec<&str> = combined.lines().rev().take(12).collect();
                tail.reverse();
                format!("{summary_line}\n{}", tail.join("\n"))
            };

            if output.status.success()
                && summary.failed == 0
                && summary.errors == 0
                && summary.xpassed == 0
            {
                PythonValidationResult::passing(target.test_name, diagnostics)
            } else {
                PythonValidationResult::failing(
                    target.test_name,
                    summary.failure_ratio(),
                    diagnostics,
                )
            }
        }
        Err(error) => PythonValidationResult::failing(
            target.test_name,
            1.0,
            format!("failed to invoke python/pytest: {error}"),
        ),
    }
}

/// Tolerance constants matching k-Wave Python parity requirements.
pub mod thresholds {
    /// L2 tolerance for solver field comparison.
    pub const SOLVER_L2_TOLERANCE: f64 = 1e-3;
    /// Tolerance for grid dimension (integer, always exact).
    pub const GRID_DIMENSION_TOLERANCE: f64 = 0.0;
    /// Tolerance for grid spacing in metres.
    pub const GRID_SPACING_TOLERANCE: f64 = 1e-12;
    /// Tolerance for signal amplitude comparison.
    pub const SIGNAL_AMPLITUDE_TOLERANCE: f64 = 1e-6;
    /// Minimum acceptable Pearson correlation for sensor data.
    pub const SENSOR_CORRELATION_THRESHOLD: f64 = 0.999;
}
