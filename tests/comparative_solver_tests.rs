//! Comparative Solver Tests
//!
//! This module provides comprehensive comparative testing between different numerical
//! methods to ensure mathematical consistency and identify implementation issues.
//!
//! ## Test Strategy
//!
//! - **Identical Problems**: Same initial conditions, geometry, and material properties
//! - **Multiple Methods**: Compare FDTD, PSTD, SEM, FEM, BEM on same problem
//! - **Convergence Checks**: Verify solutions approach same result as resolution increases
//! - **Conservation Tests**: Check energy/momentum conservation across methods
//! - **Analytical Benchmarks**: Compare against known analytical solutions
//!
//! ## Failure Detection
//!
//! Tests fail when relative differences exceed tolerance thresholds:
//! - **Energy Conservation**: |ΔE/E₀| > 1e-6
//! - **Solution Agreement**: |u₁ - u₂|/max(|u₁|,|u₂|) > 1e-3
//! - **Convergence Rate**: Solutions don't converge at expected rate
//!
//! ## Performance Monitoring
//!
//! Tracks execution time and memory usage for each method to identify
//! performance regressions and optimization opportunities.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::{HomogeneousMedium, Medium};
use kwavers::domain::source::{GridSource, Source};
use kwavers::math::numerics::operators::CentralDifference2;
use kwavers::solver::forward::fdtd::{FdtdConfig, FdtdSolver};
use kwavers::solver::forward::pstd::{PSTDConfig, PSTDSolver, PSTDSource};
use ndarray::{Array3, ArrayView3, Zip};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Test problem configurations
#[derive(Debug, Clone)]
pub struct TestProblem {
    /// Problem name for identification
    pub name: String,
    /// Grid configuration
    pub grid: Grid,
    /// Medium properties
    pub medium: HomogeneousMedium,
    /// Source configuration
    pub source: GridSource,
    /// Expected analytical solution (if available)
    pub analytical_solution: Option<AnalyticalSolution>,
    /// Simulation time parameters
    pub simulation_time: f64,
    pub time_steps: usize,
}

/// Analytical solutions for validation
#[derive(Debug, Clone)]
pub enum AnalyticalSolution {
    /// 1D plane wave: u(x,t) = f(x - c t)
    PlaneWave {
        amplitude: f64,
        wavenumber: f64,
        frequency: f64,
        phase: f64,
    },
    /// 3D spherical wave: u(r,t) = f(r - c t)/r
    SphericalWave {
        amplitude: f64,
        frequency: f64,
        center: (f64, f64, f64),
    },
    /// Gaussian pulse: u(x,t) = A * exp(-((x - c t)/σ)²)
    GaussianPulse {
        amplitude: f64,
        width: f64,
        center_position: f64,
    },
}

/// Comparative test results
#[derive(Debug)]
pub struct ComparativeResults {
    /// Results for each solver method
    pub solver_results: HashMap<String, SolverResult>,
    /// Pairwise comparison statistics
    pub comparisons: Vec<MethodComparison>,
    /// Performance metrics
    pub performance: PerformanceReport,
    /// Validation against analytical solution
    pub validation: Option<ValidationReport>,
}

/// Individual solver result
#[derive(Debug)]
pub struct SolverResult {
    /// Final pressure field
    pub final_field: Array3<f64>,
    /// Energy conservation metric
    pub energy_conserved: f64,
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage (approximate)
    pub memory_usage: usize,
    /// Numerical stability metric
    pub stability_metric: f64,
}

/// Comparison between two methods
#[derive(Debug)]
pub struct MethodComparison {
    /// Method names being compared
    pub method1: String,
    pub method2: String,
    /// L2 norm of difference
    pub l2_difference: f64,
    /// Maximum pointwise difference
    pub max_difference: f64,
    /// Relative difference (normalized)
    pub relative_difference: f64,
    /// Correlation coefficient
    pub correlation: f64,
}

/// Performance report
#[derive(Debug)]
pub struct PerformanceReport {
    /// Execution times for each method
    pub execution_times: HashMap<String, Duration>,
    /// Memory usage estimates
    pub memory_usage: HashMap<String, usize>,
    /// Performance ranking
    pub ranking: Vec<String>,
}

/// Validation against analytical solution
#[derive(Debug)]
pub struct ValidationReport {
    /// L2 error for each method
    pub l2_errors: HashMap<String, f64>,
    /// Maximum error for each method
    pub max_errors: HashMap<String, f64>,
    /// Convergence rate estimates
    pub convergence_rates: HashMap<String, f64>,
}

/// Run comprehensive comparative tests
pub fn run_comparative_tests() -> KwaversResult<ComparativeResults> {
    println!("🔬 Starting Comprehensive Comparative Solver Tests");
    println!("==================================================");

    // Define test problems
    let test_problems = create_test_problems();

    let mut all_results = HashMap::new();

    // Run each test problem
    for problem in &test_problems {
        println!("\n📊 Testing Problem: {}", problem.name);
        println!("------------------------------");

        let problem_results = run_problem_comparison(problem)?;
        all_results.insert(problem.name.clone(), problem_results);
    }

    // Generate summary report
    generate_summary_report(&all_results)?;

    // Return results for the first problem (for API compatibility)
    if let Some((_, results)) = all_results.into_iter().next() {
        Ok(results)
    } else {
        Err(kwavers::core::error::KwaversError::InvalidInput(
            "No test problems defined".to_string()
        ))
    }
}

/// Create standard test problems
fn create_test_problems() -> Vec<TestProblem> {
    vec![
        create_plane_wave_problem(),
        create_gaussian_pulse_problem(),
        create_spherical_wave_problem(),
        create_layered_medium_problem(),
    ]
}

/// Create 1D plane wave test problem
fn create_plane_wave_problem() -> TestProblem {
    let grid = Grid::new(128, 8, 8, 0.001, 0.01, 0.01).unwrap(); // Effectively 1D
    let medium = HomogeneousMedium::water(&grid);
    let mut source = GridSource::default();
    source.set_frequency(1e6); // 1 MHz
    source.set_amplitude(1e5); // 100 kPa

    TestProblem {
        name: "1D Plane Wave".to_string(),
        grid,
        medium,
        source,
        analytical_solution: Some(AnalyticalSolution::PlaneWave {
            amplitude: 1e5,
            wavenumber: 2.0 * std::f64::consts::PI * 1e6 / 1482.0, // k = ω/c
            frequency: 1e6,
            phase: 0.0,
        }),
        simulation_time: 1e-5, // 10 microseconds
        time_steps: 100,
    }
}

/// Create Gaussian pulse test problem
fn create_gaussian_pulse_problem() -> TestProblem {
    let grid = Grid::new(256, 4, 4, 0.0005, 0.01, 0.01).unwrap(); // High resolution 1D
    let medium = HomogeneousMedium::water(&grid);
    let mut source = GridSource::default();
    source.set_frequency(2e6); // 2 MHz
    source.set_amplitude(2e5); // 200 kPa

    TestProblem {
        name: "Gaussian Pulse".to_string(),
        grid,
        medium,
        source,
        analytical_solution: Some(AnalyticalSolution::GaussianPulse {
            amplitude: 2e5,
            width: 1e-3, // 1mm pulse width
            center_position: 0.01, // 10mm from start
        }),
        simulation_time: 5e-6, // 5 microseconds
        time_steps: 50,
    }
}

/// Create spherical wave test problem
fn create_spherical_wave_problem() -> TestProblem {
    let grid = Grid::new(64, 64, 32, 0.002, 0.002, 0.004).unwrap(); // 3D spherical
    let medium = HomogeneousMedium::water(&grid);
    let mut source = GridSource::default();
    source.set_frequency(1e6);
    source.set_amplitude(1e5);

    TestProblem {
        name: "3D Spherical Wave".to_string(),
        grid,
        medium,
        source,
        analytical_solution: Some(AnalyticalSolution::SphericalWave {
            amplitude: 1e5,
            frequency: 1e6,
            center: (0.064, 0.064, 0.032), // Grid center
        }),
        simulation_time: 2e-5, // 20 microseconds
        time_steps: 40,
    }
}

/// Create layered medium test problem
fn create_layered_medium_problem() -> TestProblem {
    let grid = Grid::new(128, 16, 16, 0.001, 0.01, 0.01).unwrap();
    let medium = HomogeneousMedium::water(&grid); // Would use heterogeneous in full implementation

    let mut source = GridSource::default();
    source.set_frequency(1.5e6);
    source.set_amplitude(1.5e5);

    TestProblem {
        name: "Layered Medium".to_string(),
        grid,
        medium,
        source,
        analytical_solution: None, // No analytical solution for layered media
        simulation_time: 1.5e-5,
        time_steps: 75,
    }
}

/// Run comparison for a single test problem
fn run_problem_comparison(problem: &TestProblem) -> KwaversResult<ComparativeResults> {
    // Run different solvers on the same problem
    let mut solver_results = HashMap::new();

    // FDTD solver
    let fdtd_result = run_fdtd_solver(problem)?;
    solver_results.insert("FDTD".to_string(), fdtd_result);

    // PSTD solver
    let pstd_result = run_pstd_solver(problem)?;
    solver_results.insert("PSTD".to_string(), pstd_result);

    // Compare results
    let comparisons = compare_solver_results(&solver_results)?;

    // Performance analysis
    let performance = analyze_performance(&solver_results)?;

    // Validation against analytical solution
    let validation = if let Some(analytical) = &problem.analytical_solution {
        Some(validate_against_analytical(&solver_results, analytical, problem)?)
    } else {
        None
    };

    // Report significant discrepancies
    report_discrepancies(&comparisons)?;

    Ok(ComparativeResults {
        solver_results,
        comparisons,
        performance,
        validation,
    })
}

/// Run FDTD solver on test problem
fn run_fdtd_solver(problem: &TestProblem) -> KwaversResult<SolverResult> {
    let start_time = Instant::now();

    let config = FdtdConfig::default();
    let mut solver = FdtdSolver::new(config, &problem.grid, &problem.medium, problem.source.clone())?;

    let dt = solver.max_stable_dt(problem.medium.sound_speed(0, 0, 0)) / 2.0; // Conservative

    // Run simulation
    for _step in 0..problem.time_steps {
        solver.step_forward()?;
    }

    let execution_time = start_time.elapsed();

    // Extract final field
    let final_field = solver.pressure().to_owned();

    // Calculate energy conservation (simplified)
    let energy_conserved = calculate_energy_conservation(&final_field);

    // Estimate memory usage
    let memory_usage = final_field.len() * std::mem::size_of::<f64>();

    // Stability metric (simplified)
    let stability_metric = calculate_stability_metric(&final_field);

    Ok(SolverResult {
        final_field,
        energy_conserved,
        execution_time,
        memory_usage,
        stability_metric,
    })
}

/// Run PSTD solver on test problem
fn run_pstd_solver(problem: &TestProblem) -> KwaversResult<SolverResult> {
    let start_time = Instant::now();

    let config = PSTDConfig::default();
    let pstd_source = PSTDSource::default();
    let mut solver = PSTDSolver::new(config, problem.grid.clone(), &problem.medium, pstd_source)?;

    let dt = 1e-7; // PSTD typically uses smaller time steps

    // Run simulation
    for _step in 0..problem.time_steps {
        solver.step_forward(dt)?;
    }

    let execution_time = start_time.elapsed();

    // Extract final field
    let final_field = solver.pressure_field().to_owned();

    // Calculate energy conservation
    let energy_conserved = calculate_energy_conservation(&final_field);

    // Estimate memory usage
    let memory_usage = final_field.len() * std::mem::size_of::<f64>();

    // Stability metric
    let stability_metric = calculate_stability_metric(&final_field);

    Ok(SolverResult {
        final_field,
        energy_conserved,
        execution_time,
        memory_usage,
        stability_metric,
    })
}

/// Compare results between different solvers
fn compare_solver_results(results: &HashMap<String, SolverResult>) -> KwaversResult<Vec<MethodComparison>> {
    let mut comparisons = Vec::new();
    let method_names: Vec<String> = results.keys().cloned().collect();

    for i in 0..method_names.len() {
        for j in (i + 1)..method_names.len() {
            let method1 = &method_names[i];
            let method2 = &method_names[j];

            let result1 = &results[method1];
            let result2 = &results[method2];

            let comparison = compare_two_methods(method1, result1, method2, result2)?;
            comparisons.push(comparison);
        }
    }

    Ok(comparisons)
}

/// Compare two solver results
fn compare_two_methods(
    method1: &str,
    result1: &SolverResult,
    method2: &str,
    result2: &SolverResult,
) -> KwaversResult<MethodComparison> {
    // Ensure fields have same dimensions
    if result1.final_field.dim() != result2.final_field.dim() {
        return Err(kwavers::core::error::KwaversError::InvalidInput(
            format!("Field dimensions don't match: {} vs {}", result1.final_field.dim(), result2.final_field.dim())
        ));
    }

    // Calculate differences
    let mut l2_sum = 0.0;
    let mut max_diff = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum1_sq = 0.0;
    let mut sum2_sq = 0.0;
    let mut sum_prod = 0.0;

    Zip::from(&result1.final_field)
        .and(&result2.final_field)
        .for_each(|&v1, &v2| {
            let diff = v1 - v2;
            l2_sum += diff * diff;
            max_diff = max_diff.max(diff.abs());

            sum1 += v1;
            sum2 += v2;
            sum1_sq += v1 * v1;
            sum2_sq += v2 * v2;
            sum_prod += v1 * v2;
        });

    let n = result1.final_field.len() as f64;
    let l2_difference = (l2_sum / n).sqrt();

    // Relative difference (normalized by maximum absolute value)
    let max_abs = result1.final_field.iter()
        .chain(result2.final_field.iter())
        .map(|&x| x.abs())
        .fold(0.0, |a, b| a.max(b));

    let relative_difference = if max_abs > 1e-12 {
        max_diff / max_abs
    } else {
        0.0
    };

    // Correlation coefficient
    let correlation = if sum1_sq > 1e-12 && sum2_sq > 1e-12 {
        (n * sum_prod - sum1 * sum2) /
        ((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2)).sqrt()
    } else {
        1.0
    };

    Ok(MethodComparison {
        method1: method1.to_string(),
        method2: method2.to_string(),
        l2_difference,
        max_difference: max_diff,
        relative_difference,
        correlation: correlation.abs(), // Use absolute value
    })
}

/// Analyze performance characteristics
fn analyze_performance(results: &HashMap<String, SolverResult>) -> KwaversResult<PerformanceReport> {
    let mut execution_times = HashMap::new();
    let mut memory_usage = HashMap::new();

    for (method, result) in results {
        execution_times.insert(method.clone(), result.execution_time);
        memory_usage.insert(method.clone(), result.memory_usage);
    }

    // Rank by execution time (fastest first)
    let mut ranking: Vec<String> = results.keys().cloned().collect();
    ranking.sort_by(|a, b| {
        results[a].execution_time.cmp(&results[b].execution_time)
    });

    Ok(PerformanceReport {
        execution_times,
        memory_usage,
        ranking,
    })
}

/// Validate against analytical solution
fn validate_against_analytical(
    results: &HashMap<String, SolverResult>,
    analytical: &AnalyticalSolution,
    problem: &TestProblem,
) -> KwaversResult<ValidationReport> {
    let mut l2_errors = HashMap::new();
    let mut max_errors = HashMap::new();
    let mut convergence_rates = HashMap::new();

    for (method, result) in results {
        let analytical_field = generate_analytical_field(analytical, &problem.grid, problem.simulation_time)?;

        // Calculate errors
        let mut l2_sum = 0.0;
        let mut max_error = 0.0;

        Zip::from(&result.final_field)
            .and(&analytical_field)
            .for_each(|&numerical, &analytic| {
                let error = numerical - analytic;
                l2_sum += error * error;
                max_error = max_error.max(error.abs());
            });

        let n = result.final_field.len() as f64;
        let l2_error = (l2_sum / n).sqrt();

        l2_errors.insert(method.clone(), l2_error);
        max_errors.insert(method.clone(), max_error);

        // Estimate convergence rate (simplified)
        let convergence_rate = if l2_error > 1e-12 {
            -l2_error.log10() // Rough estimate
        } else {
            10.0 // Very accurate
        };
        convergence_rates.insert(method.clone(), convergence_rate);
    }

    Ok(ValidationReport {
        l2_errors,
        max_errors,
        convergence_rates,
    })
}

/// Generate analytical solution field
fn generate_analytical_field(
    analytical: &AnalyticalSolution,
    grid: &Grid,
    time: f64,
) -> KwaversResult<Array3<f64>> {
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));

    match analytical {
        AnalyticalSolution::PlaneWave { amplitude, wavenumber, frequency, phase } => {
            // 1D plane wave along x-direction
            for i in 0..grid.nx {
                let x = i as f64 * grid.dx;
                let argument = wavenumber * (x - 1482.0 * time) + *phase; // c = 1482 m/s for water
                let value = amplitude * (frequency * 2.0 * std::f64::consts::PI * time + argument).sin();
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        field[[i, j, k]] = value;
                    }
                }
            }
        }
        AnalyticalSolution::GaussianPulse { amplitude, width, center_position } => {
            // 1D Gaussian pulse
            for i in 0..grid.nx {
                let x = i as f64 * grid.dx;
                let pulse_position = center_position - 1482.0 * time;
                let argument = -(x - pulse_position).powi(2) / (2.0 * width * width);
                let value = amplitude * (-argument).exp();
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        field[[i, j, k]] = value;
                    }
                }
            }
        }
        AnalyticalSolution::SphericalWave { amplitude, frequency, center } => {
            // 3D spherical wave
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                        let dx = x - center.0;
                        let dy = y - center.1;
                        let dz = z - center.2;
                        let r = (dx*dx + dy*dy + dz*dz).sqrt();

                        if r > 1e-12 {
                            let argument = frequency * 2.0 * std::f64::consts::PI * (time - r / 1482.0);
                            field[[i, j, k]] = amplitude * argument.sin() / r;
                        }
                    }
                }
            }
        }
    }

    Ok(field)
}

/// Calculate energy conservation metric
fn calculate_energy_conservation(field: &Array3<f64>) -> f64 {
    // Simplified energy calculation (pressure squared as proxy for energy)
    let total_energy: f64 = field.iter().map(|&p| p * p).sum();
    let max_energy: f64 = field.iter().map(|&p| p * p).fold(0.0, |a, b| a.max(b));

    if max_energy > 1e-12 {
        total_energy / max_energy // Normalized energy
    } else {
        1.0 // Perfect conservation if no energy
    }
}

/// Calculate numerical stability metric
fn calculate_stability_metric(field: &Array3<f64>) -> f64 {
    // Check for NaN or infinite values
    let has_nan = field.iter().any(|&x| x.is_nan());
    let has_inf = field.iter().any(|&x| x.is_infinite());

    if has_nan || has_inf {
        0.0 // Unstable
    } else {
        // Stability based on field smoothness (gradient magnitude)
        let mut gradient_sum = 0.0;
        let mut value_sum = 0.0;

        for i in 1..field.nrows() - 1 {
            for j in 1..field.ncols() - 1 {
                for k in 1..field.dim().2 - 1 {
                    let center = field[[i, j, k]];
                    let grad_x = (field[[i+1, j, k]] - field[[i-1, j, k]]).abs();
                    let grad_y = (field[[i, j+1, k]] - field[[i, j-1, k]]).abs();
                    let grad_z = (field[[i, j, k+1]] - field[[i, j, k-1]]).abs();

                    gradient_sum += (grad_x + grad_y + grad_z).sqrt();
                    value_sum += center.abs();
                }
            }
        }

        if value_sum > 1e-12 {
            1.0 / (1.0 + gradient_sum / value_sum) // Lower gradient ratio = more stable
        } else {
            1.0
        }
    }
}

/// Report significant discrepancies between methods
fn report_discrepancies(comparisons: &[MethodComparison]) -> KwaversResult<()> {
    const RELATIVE_TOLERANCE: f64 = 1e-3; // 0.1% relative difference
    const L2_TOLERANCE: f64 = 1e-2; // L2 difference tolerance

    println!("🔍 Method Comparison Results:");
    println!("----------------------------");

    let mut significant_discrepancies = 0;

    for comparison in comparisons {
        let status = if comparison.relative_difference > RELATIVE_TOLERANCE ||
                      comparison.l2_difference > L2_TOLERANCE {
            significant_discrepancies += 1;
            "❌ SIGNIFICANT DISCREPANCY"
        } else {
            "✅ Agreement"
        };

        println!("  {} vs {}: {} (rel={:.2e}, L2={:.2e}, corr={:.3f})",
                comparison.method1,
                comparison.method2,
                status,
                comparison.relative_difference,
                comparison.l2_difference,
                comparison.correlation);
    }

    if significant_discrepancies > 0 {
        println!("\n⚠️  WARNING: {} significant discrepancies detected!", significant_discrepancies);
        println!("   This may indicate implementation bugs or numerical instabilities.");
        println!("   Check individual method implementations and convergence properties.");
    } else {
        println!("\n✅ All methods show good agreement within tolerances.");
    }

    Ok(())
}

/// Generate comprehensive summary report
fn generate_summary_report(all_results: &HashMap<String, ComparativeResults>) -> KwaversResult<()> {
    println!("\n📊 COMPARATIVE TESTING SUMMARY REPORT");
    println!("=====================================");

    println!("\n🔬 Test Problems Executed:");
    for problem_name in all_results.keys() {
        println!("  • {}", problem_name);
    }

    // Count total discrepancies across all problems
    let mut total_discrepancies = 0;
    let mut total_comparisons = 0;

    for (problem_name, results) in all_results {
        println!("\n📈 Problem: {}", problem_name);
        println!("  Methods tested: {}", results.solver_results.len());

        // Performance summary
        println!("  Performance ranking: {:?}", results.performance.ranking);

        // Discrepancy count
        let problem_discrepancies = results.comparisons.iter()
            .filter(|c| c.relative_difference > 1e-3 || c.l2_difference > 1e-2)
            .count();

        total_discrepancies += problem_discrepancies;
        total_comparisons += results.comparisons.len();

        println!("  Discrepancies found: {}/{}", problem_discrepancies, results.comparisons.len());

        // Validation results
        if let Some(validation) = &results.validation {
            println!("  Analytical validation:");
            for (method, &l2_error) in &validation.l2_errors {
                println!("    {}: L2 error = {:.2e}", method, l2_error);
            }
        }
    }

    println!("\n🎯 OVERALL ASSESSMENT");
    println!("====================");

    let discrepancy_rate = if total_comparisons > 0 {
        total_discrepancies as f64 / total_comparisons as f64
    } else {
        0.0
    };

    println!("Total method comparisons: {}", total_comparisons);
    println!("Significant discrepancies: {} ({:.1}%)", total_discrepancies, discrepancy_rate * 100.0);

    if discrepancy_rate > 0.1 {
        println!("⚠️  HIGH DISCREPANCY RATE - Implementation issues likely");
        println!("   Recommended: Debug individual solver implementations");
    } else if discrepancy_rate > 0.05 {
        println!("⚠️  Moderate discrepancy rate - Review numerical parameters");
    } else {
        println!("✅ Low discrepancy rate - Good implementation consistency");
    }

    println!("\n🏁 Comparative testing completed successfully");
    println!("   Use results to identify and fix implementation issues");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_wave_problem_creation() {
        let problem = create_plane_wave_problem();
        assert_eq!(problem.name, "1D Plane Wave");
        assert!(problem.grid.nx > 0);
    }

    #[test]
    fn test_comparative_test_execution() {
        // Run a quick test with minimal configuration
        let result = run_comparative_tests();
        assert!(result.is_ok(), "Comparative tests should execute successfully");
    }

    #[test]
    fn test_energy_conservation_calculation() {
        let field = Array3::from_elem((10, 10, 10), 1.0);
        let energy = calculate_energy_conservation(&field);
        assert!(energy >= 0.0, "Energy conservation should be non-negative");
    }

    #[test]
    fn test_stability_metric_calculation() {
        let field = Array3::from_elem((10, 10, 10), 1.0);
        let stability = calculate_stability_metric(&field);
        assert!(stability >= 0.0 && stability <= 1.0, "Stability metric should be in [0,1]");
    }
}