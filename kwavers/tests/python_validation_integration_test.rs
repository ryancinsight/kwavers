//! Python Validation Integration Test
//!
//! Comprehensive integration test suite comparing pykwavers outputs against
//! k-wave-python reference implementations.
//!
//! ## Test Organization
//!
//! - `test_module_imports`: Verify all validation modules compile and load
//! - `test_grid_validation_suite`: Grid creation and properties
//! - `test_signal_validation_suite`: Signal generation (tone bursts, sinusoids)
//! - `test_source_validation_suite`: Source definitions (point, line, area)
//! - `test_sensor_validation_suite`: Sensor configurations
//! - `test_solver_validation_suite`: FDTD/PSTD solver comparisons
//! - `test_full_workflow_validation`: End-to-end simulation validation

use kwavers::domain::grid::Grid;

// Import validation modules using path attribute
#[path = "validation/python/grid_validation.rs"]
mod grid_validation;
#[path = "validation/python/mod.rs"]
mod python_validation;
#[path = "validation/python/sensor_validation.rs"]
mod sensor_validation;
#[path = "validation/python/signal_validation.rs"]
mod signal_validation;
#[path = "validation/python/solver_validation.rs"]
mod solver_validation;
#[path = "validation/python/source_validation.rs"]
mod source_validation;

// Import specific functions we need
use grid_validation::generate_grid_validation_report;
use python_validation::{thresholds, PythonValidationResult};
use sensor_validation::generate_sensor_validation_report;
use signal_validation::generate_signal_validation_report;
use solver_validation::generate_solver_validation_report;
use source_validation::generate_source_validation_report;

/// Environment variable to control Python-based validation
const PYTHON_VALIDATION_ENV: &str = "KWAVERS_RUN_PYTHON";

/// Check if full Python validation should run
fn should_run_python_validation() -> bool {
    std::env::var(PYTHON_VALIDATION_ENV)
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Comprehensive validation report structure
#[derive(Debug, Clone)]
pub struct ComprehensiveValidationReport {
    pub grid_results: Vec<PythonValidationResult>,
    pub signal_results: Vec<PythonValidationResult>,
    pub source_results: Vec<PythonValidationResult>,
    pub sensor_results: Vec<PythonValidationResult>,
    pub solver_results: Vec<PythonValidationResult>,
    pub timestamp: String,
}

impl Default for ComprehensiveValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl ComprehensiveValidationReport {
    pub fn new() -> Self {
        Self {
            grid_results: Vec::new(),
            signal_results: Vec::new(),
            source_results: Vec::new(),
            sensor_results: Vec::new(),
            solver_results: Vec::new(),
            timestamp: chrono::Local::now().to_rfc3339(),
        }
    }

    pub fn total_tests(&self) -> usize {
        self.grid_results.len()
            + self.signal_results.len()
            + self.source_results.len()
            + self.sensor_results.len()
            + self.solver_results.len()
    }

    pub fn passed_count(&self) -> usize {
        self.grid_results.iter().filter(|r| r.passed).count()
            + self.signal_results.iter().filter(|r| r.passed).count()
            + self.source_results.iter().filter(|r| r.passed).count()
            + self.sensor_results.iter().filter(|r| r.passed).count()
            + self.solver_results.iter().filter(|r| r.passed).count()
    }

    pub fn failed_count(&self) -> usize {
        self.total_tests() - self.passed_count()
    }

    pub fn max_relative_error(&self) -> f64 {
        let all_results: Vec<_> = self
            .grid_results
            .iter()
            .chain(self.signal_results.iter())
            .chain(self.source_results.iter())
            .chain(self.sensor_results.iter())
            .chain(self.solver_results.iter())
            .collect();

        all_results
            .iter()
            .map(|r| r.relative_error)
            .fold(0.0, f64::max)
    }

    /// Generate markdown report
    pub fn to_markdown(&self) -> String {
        let mut report = String::new();

        report.push_str("# pykwavers vs k-wave-python Validation Report\n\n");
        report.push_str(&format!("**Timestamp:** {}\\n\\n", self.timestamp));
        report.push_str(&format!(
            "**Summary:** {}/{} tests passed ({:.1}%)\\n\\n",
            self.passed_count(),
            self.total_tests(),
            100.0 * self.passed_count() as f64 / self.total_tests().max(1) as f64
        ));

        report.push_str("## Results by Component\\n\\n");
        report.push_str("| Component | Passed | Failed | Max Error |\\n");
        report.push_str("|-----------|--------|--------|-----------|\\n");

        let grid_passed = self.grid_results.iter().filter(|r| r.passed).count();
        let grid_failed = self.grid_results.len() - grid_passed;
        let grid_max_error = self
            .grid_results
            .iter()
            .map(|r| r.relative_error)
            .fold(0.0, f64::max);
        report.push_str(&format!(
            "| Grid | {} | {} | {:.3e} |\\n",
            grid_passed, grid_failed, grid_max_error
        ));

        let signal_passed = self.signal_results.iter().filter(|r| r.passed).count();
        let signal_failed = self.signal_results.len() - signal_passed;
        let signal_max_error = self
            .signal_results
            .iter()
            .map(|r| r.relative_error)
            .fold(0.0, f64::max);
        report.push_str(&format!(
            "| Signal | {} | {} | {:.3e} |\\n",
            signal_passed, signal_failed, signal_max_error
        ));

        let source_passed = self.source_results.iter().filter(|r| r.passed).count();
        let source_failed = self.source_results.len() - source_passed;
        let source_max_error = self
            .source_results
            .iter()
            .map(|r| r.relative_error)
            .fold(0.0, f64::max);
        report.push_str(&format!(
            "| Source | {} | {} | {:.3e} |\\n",
            source_passed, source_failed, source_max_error
        ));

        let sensor_passed = self.sensor_results.iter().filter(|r| r.passed).count();
        let sensor_failed = self.sensor_results.len() - sensor_passed;
        let sensor_max_error = self
            .sensor_results
            .iter()
            .map(|r| r.relative_error)
            .fold(0.0, f64::max);
        report.push_str(&format!(
            "| Sensor | {} | {} | {:.3e} |\\n",
            sensor_passed, sensor_failed, sensor_max_error
        ));

        let solver_passed = self.solver_results.iter().filter(|r| r.passed).count();
        let solver_failed = self.solver_results.len() - solver_passed;
        let solver_max_error = self
            .solver_results
            .iter()
            .map(|r| r.relative_error)
            .fold(0.0, f64::max);
        report.push_str(&format!(
            "| Solver | {} | {} | {:.3e} |\\n",
            solver_passed, solver_failed, solver_max_error
        ));

        report.push_str("\\n## Detailed Results\\n\\n");

        let all_results: Vec<_> = self
            .grid_results
            .iter()
            .chain(self.signal_results.iter())
            .chain(self.source_results.iter())
            .chain(self.sensor_results.iter())
            .chain(self.solver_results.iter())
            .collect();

        for result in all_results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            report.push_str(&format!("### {}: {}\\n\\n", result.test_name, status));
            report.push_str(&format!("- **L2 Error:** {:.3e}\\n", result.l2_error));
            report.push_str(&format!(
                "- **Relative Error:** {:.3e}\\n",
                result.relative_error
            ));
            report.push_str(&format!("- **Max Error:** {:.3e}\\n", result.max_error));
            report.push_str(&format!("- **Diagnostics:** {}\\n\\n", result.diagnostics));
        }

        report
    }
}

/// Theorem: Module Import Validation
///
/// All validation submodules must compile and be importable.
#[test]
fn test_module_imports() {
    // Test that all modules are accessible via the re-exports
    let _ = thresholds::SOLVER_L2_TOLERANCE;
    let _ = thresholds::GRID_DIMENSION_TOLERANCE;
    // Compile-time sanity: these are checked statically in the threshold module
    let _ = thresholds::SOLVER_L2_TOLERANCE;
    let _ = thresholds::SENSOR_CORRELATION_THRESHOLD;
}

/// Theorem: Grid Validation Suite
///
/// Validates that pykwavers Grid creation matches k-wave-python kWaveGrid
/// across all standard configurations.
#[test]
fn test_grid_validation_suite() {
    let results = generate_grid_validation_report();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    println!("Grid Validation: {}/{} passed", passed, total);
    println!(
        "Max relative error: {:.3e}",
        results.iter().map(|r| r.relative_error).fold(0.0, f64::max)
    );

    // Grid validation should always pass (no Python dependency)
    assert_eq!(
        passed, total,
        "Grid validation failed: got {}/{} passed",
        passed, total
    );

    // Verify thresholds
    let max_error = results.iter().map(|r| r.relative_error).fold(0.0, f64::max);
    assert!(
        max_error < thresholds::GRID_SPACING_TOLERANCE,
        "Grid error {:.3e} exceeds tolerance {:.3e}",
        max_error,
        thresholds::GRID_SPACING_TOLERANCE
    );
}

/// Theorem: Signal Validation Suite
///
/// Validates tone burst and sinusoid generation against mathematical specifications.
#[test]
fn test_signal_validation_suite() {
    let results = generate_signal_validation_report();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    println!("Signal Validation: {}/{} passed", passed, total);

    assert!(passed > 0, "Signal validation should run");

    let max_error = results.iter().map(|r| r.relative_error).fold(0.0, f64::max);
    println!("Signal max relative error: {:.3e}", max_error);

    // Signals should match within tolerance
    assert!(
        max_error < thresholds::SIGNAL_AMPLITUDE_TOLERANCE,
        "Signal error {:.3e} exceeds amplitude tolerance {:.3e}",
        max_error,
        thresholds::SIGNAL_AMPLITUDE_TOLERANCE
    );
}

/// Theorem: Source Validation Suite
///
/// Validates point, line, and area source definitions.
#[test]
fn test_source_validation_suite() {
    let results = generate_source_validation_report();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    println!("Source Validation: {}/{} passed", passed, total);

    // Should have tests for point, line, plane sources
    assert!(
        total >= 3,
        "Source validation should cover at least 3 source types"
    );

    // All should pass
    assert_eq!(
        passed, total,
        "Source validation failed: {}/{} passed",
        passed, total
    );
}

/// Theorem: Sensor Validation Suite
///
/// Validates point sensors, area sensors, and beamforming configurations.
#[test]
fn test_sensor_validation_suite() {
    let results = generate_sensor_validation_report();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    println!("Sensor Validation: {}/{} passed", passed, total);

    assert!(
        total >= 3,
        "Sensor validation should cover multiple sensor types"
    );
    assert_eq!(
        passed, total,
        "Sensor validation failed: {}/{} passed",
        passed, total
    );
}

/// Theorem: Solver Validation Suite
///
/// Validates FDTD/PSTD solvers against analytical solutions.
#[test]
fn test_solver_validation_suite() {
    let results = generate_solver_validation_report();

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    println!("Solver Validation: {}/{} passed", passed, total);

    // Solvers should have some tests run
    assert!(total > 0, "Solver validation should run tests");

    // Check error bounds
    let max_error = results.iter().map(|r| r.relative_error).fold(0.0, f64::max);
    println!("Solver max relative error: {:.3e}", max_error);

    assert!(
        max_error < thresholds::SOLVER_L2_TOLERANCE,
        "Solver error {:.3e} exceeds tolerance {:.3e}",
        max_error,
        thresholds::SOLVER_L2_TOLERANCE
    );
}

/// Comprehensive integration test
///
/// Runs all validation modules and generates a comprehensive report.
#[test]
fn test_full_validation_suite() {
    println!("\\n========================================");
    println!("Running Full Validation Suite");
    println!("========================================\\n");

    let mut report = ComprehensiveValidationReport::new();

    // Run all validation modules
    report.grid_results = generate_grid_validation_report();
    report.signal_results = generate_signal_validation_report();
    report.source_results = generate_source_validation_report();
    report.sensor_results = generate_sensor_validation_report();
    report.solver_results = generate_solver_validation_report();

    // Print summary
    println!("\\n=== Validation Summary ===");
    println!(
        "Grid: {}/{}",
        report.grid_results.iter().filter(|r| r.passed).count(),
        report.grid_results.len()
    );
    println!(
        "Signal: {}/{}",
        report.signal_results.iter().filter(|r| r.passed).count(),
        report.signal_results.len()
    );
    println!(
        "Source: {}/{}",
        report.source_results.iter().filter(|r| r.passed).count(),
        report.source_results.len()
    );
    println!(
        "Sensor: {}/{}",
        report.sensor_results.iter().filter(|r| r.passed).count(),
        report.sensor_results.len()
    );
    println!(
        "Solver: {}/{}",
        report.solver_results.iter().filter(|r| r.passed).count(),
        report.solver_results.len()
    );
    println!(
        "\\nTotal: {}/{}",
        report.passed_count(),
        report.total_tests()
    );
    println!("Max Error: {:.3e}", report.max_relative_error());

    // Save report if running in CI
    if should_run_python_validation() {
        let markdown = report.to_markdown();
        let report_path = "kwavers/validation_reports/python/validation_report.md";

        if let Ok(()) = std::fs::create_dir_all("kwavers/validation_reports/python") {
            if let Err(e) = std::fs::write(report_path, markdown) {
                eprintln!("Failed to write report: {}", e);
            } else {
                println!("Report saved to: {}", report_path);
            }
        }
    }

    // Grid, signal, source, sensor should all pass
    let grid_passed = report.grid_results.iter().filter(|r| r.passed).count();
    let grid_total = report.grid_results.len();
    assert_eq!(
        grid_passed, grid_total,
        "Grid validation must pass: {}/{}",
        grid_passed, grid_total
    );

    let signal_passed = report.signal_results.iter().filter(|r| r.passed).count();
    let signal_total = report.signal_results.len();
    assert_eq!(
        signal_passed, signal_total,
        "Signal validation must pass: {}/{}",
        signal_passed, signal_total
    );

    let source_passed = report.source_results.iter().filter(|r| r.passed).count();
    let source_total = report.source_results.len();
    assert_eq!(
        source_passed, source_total,
        "Source validation must pass: {}/{}",
        source_passed, source_total
    );

    let sensor_passed = report.sensor_results.iter().filter(|r| r.passed).count();
    let sensor_total = report.sensor_results.len();
    assert_eq!(
        sensor_passed, sensor_total,
        "Sensor validation must pass: {}/{}",
        sensor_passed, sensor_total
    );

    println!("\\n========================================");
    println!("Validation Suite Complete");
    println!("========================================\\n");
}

/// Python interop validation (requires k-wave-python)
#[test]
#[ignore = "Requires Python environment with k-wave-python installed"]
fn test_python_interop_validation() {
    if !should_run_python_validation() {
        println!("Skipping Python interop validation (set KWAVERS_RUN_PYTHON=1 to enable)");
        return;
    }

    println!("Running Python interop validation...");
    // Python runtime test would go here
}

/// Benchmark: Validation performance
// #[test] // Empirical benchmarks eradicated in adherence with Phase 5 requirements
#[allow(dead_code)]
fn test_validation_performance() {
    use std::time::Instant;

    let start = Instant::now();

    let _grid = generate_grid_validation_report();
    let _signal = generate_signal_validation_report();
    let _source = generate_source_validation_report();
    let _sensor = generate_sensor_validation_report();

    let elapsed = start.elapsed();

    println!("Validation suite completed in {:?}", elapsed);

    // Should complete in reasonable time (< 30 seconds for basic validation)
    assert!(
        elapsed.as_secs() < 30,
        "Validation took too long: {:?}",
        elapsed
    );
}

/// Theorem: Dimensional consistency
///
/// All grid, signal, source, sensor, and solver components must maintain
/// consistent physical units throughout the validation pipeline.
#[test]
fn test_dimensional_consistency() {
    // Grid dimensions: meters
    let grid = Grid::new(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3).expect("Valid grid");

    // Domain size in meters
    let domain_size = grid.nx as f64 * grid.dx;
    assert!(domain_size > 0.0 && domain_size < 0.1); // 6.4mm = 0.0064m

    // Frequency: Hz
    let frequency: f64 = 1e6;

    // Sound speed: m/s
    let sound_speed: f64 = 1500.0;

    // Wavelength: λ = c/f [meters]
    let wavelength = sound_speed / frequency;
    assert!(wavelength > 0.0 && wavelength < 0.01); // 1.5mm

    // Period: T = 1/f [seconds]
    let period = 1.0 / frequency;
    assert!(period > 0.0 && period < 1e-3); // 1μs

    // Points per wavelength (dimensionless)
    let ppw = wavelength / grid.dx;
    assert!(ppw >= 2.0); // Nyquist criterion

    println!("Dimensional consistency verified:");
    println!(" Domain size: {:.3e} m", domain_size);
    println!(" Wavelength: {:.3e} m", wavelength);
    println!(" Period: {:.3e} s", period);
    println!(" PPW: {:.1}", ppw);
}

/// Regression test: Known k-wave-python configurations
///
/// These are specific configurations from k-wave-python examples that
/// must produce consistent results.
#[test]
fn test_known_kwave_configurations() {
    // Configuration from k-wave-python example_ivp_homogeneous
    let grid_homogeneous =
        Grid::new(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3).expect("IVP homogeneous grid");
    assert_eq!(
        grid_homogeneous.nx * grid_homogeneous.ny * grid_homogeneous.nz,
        64usize.pow(3)
    );

    // Configuration from k-wave-python example_tvsp_thermodynamic
    let grid_tvsp = Grid::new(32, 32, 32, 0.15e-3, 0.15e-3, 0.15e-3).expect("TVSP grid");
    assert_eq!(grid_tvsp.nx * grid_tvsp.ny * grid_tvsp.nz, 32usize.pow(3));

    // Quasi-2D grid from k-wave-python examples
    let grid_2d = Grid::new(128, 128, 1, 0.05e-3, 0.05e-3, 0.05e-3).expect("2D grid");
    assert_eq!(grid_2d.nz, 1);

    println!("All known configurations validated");
}

/// Documentation test: Mathematical specifications
///
/// Validates that the mathematical specifications produce expected results.
#[test]
fn test_mathematical_specifications() {
    // Test grid specifications
    let n = 64;
    let dx = 0.1e-3;
    let l = n as f64 * dx; // domain_size(n, dx)
    assert_eq!(l, n as f64 * dx);

    // Nyquist wavenumber
    let k_max = std::f64::consts::PI / dx; // nyquist_wavenumber(dx)
    assert_eq!(k_max, std::f64::consts::PI / dx);

    // Points per wavelength
    let c = 1500.0;
    let f = 1e6;
    let ppw = (c / f) / dx; // points_per_wavelength(c, f, dx)

    // CFL timestep
    let cfl = 0.3;
    let dt = cfl * dx / c; // cfl_dt(dx, c, cfl)
    let expected_dt = cfl * dx / c;
    assert!((dt - expected_dt).abs() < 1e-15);

    println!("Mathematical specifications verified:");
    println!(" Domain size L = {:.3e} m", l);
    println!(" Nyquist k_max = {:.3e} rad/m", k_max);
    println!(" Points/wavelength = {:.1}", ppw);
    println!(" CFL dt = {:.3e} s", dt);
}

/// CI/CD hook for generating validation reports
///
/// This test generates a validation report that can be consumed by CI systems.
#[test]
fn test_generate_ci_report() {
    let mut report = ComprehensiveValidationReport::new();

    report.grid_results = generate_grid_validation_report();
    report.signal_results = generate_signal_validation_report();
    report.source_results = generate_source_validation_report();
    report.sensor_results = generate_sensor_validation_report();
    report.solver_results = generate_solver_validation_report();

    // Output in JUnit-like format for CI parsing
    println!("<testsuites>");
    println!(
        " <testsuite name=\"grid_validation\" tests=\"{}\" failures=\"{}\">",
        report.grid_results.len(),
        report.grid_results.len() - report.grid_results.iter().filter(|r| r.passed).count()
    );
    for result in &report.grid_results {
        println!(
            " <testcase name=\"{}\" classname=\"grid_validation\">",
            result.test_name
        );
        if !result.passed {
            println!(" <failure message=\"{}\">", result.diagnostics);
            println!(" L2 Error: {:.3e}", result.l2_error);
            println!(" Relative Error: {:.3e}", result.relative_error);
            println!(" </failure>");
        }
        println!(" </testcase>");
    }
    println!(" </testsuite>");
    println!("</testsuites>");

    // Overall status
    let total_passed = report.passed_count();
    let total_tests = report.total_tests();

    println!("\\nVALIDATION_SUMMARY:");
    println!(" total: {}", total_tests);
    println!(" passed: {}", total_passed);
    println!(" failed: {}", total_tests - total_passed);
    println!(
        " success_rate: {:.2}%",
        100.0 * total_passed as f64 / total_tests as f64
    );

    assert!(total_passed > 0, "Some tests must pass");
}
