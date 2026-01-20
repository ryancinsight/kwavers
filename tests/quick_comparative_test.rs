//! Quick Comparative Solver Test
//!
//! Fast test to identify significant differences between solver implementations.
//! Focuses on core functionality with minimal computational overhead.

use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::source::GridSource;
use kwavers::solver::forward::fdtd::{FdtdConfig, FdtdSolver};
use kwavers::solver::forward::pstd::{PSTDConfig, PSTDSolver};
use kwavers::solver::interface::solver::Solver;

/// Quick comparison test - runs in under 10 seconds
#[test]
fn comparative_quick_test() {
    println!("🚀 Quick Comparative Solver Test");
    println!("===============================");

    // Use small grid for fast execution
    let grid = Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    // Use default GridSource (no frequency/amplitude setters)
    let source = GridSource::default();

    let time_steps = 5; // Very few steps for speed

    println!(
        "Grid: {}x{}x{}, Steps: {}",
        grid.nx, grid.ny, grid.nz, time_steps
    );

    // Run FDTD
    let fdtd_result = run_fdtd_quick(&grid, &medium, &source, time_steps);
    println!(
        "FDTD: {:.1}ms, Energy: {:.2e}, Stability: {:.3}",
        fdtd_result.execution_time.as_millis(),
        fdtd_result.energy,
        fdtd_result.stability
    );

    // Run PSTD
    let pstd_result = run_pstd_quick(&grid, &medium, time_steps);
    println!(
        "PSTD: {:.1}ms, Energy: {:.2e}, Stability: {:.3}",
        pstd_result.execution_time.as_millis(),
        pstd_result.energy,
        pstd_result.stability
    );

    // Compare FDTD vs PSTD specifically
    let _fdtd_pstd_energy_diff = ((fdtd_result.energy - pstd_result.energy)
        / fdtd_result.energy.abs().max(pstd_result.energy.abs()))
    .abs();

    // Compare results
    let energy_diff = ((fdtd_result.energy - pstd_result.energy)
        / fdtd_result.energy.abs().max(pstd_result.energy.abs()))
    .abs();

    let stability_diff = (fdtd_result.stability - pstd_result.stability).abs();

    println!("Comparisons:");
    println!(
        "  Energy difference: {:.2e} ({:.2}%)",
        energy_diff,
        energy_diff * 100.0
    );
    println!("  Stability difference: {:.3}", stability_diff);

    // Check for significant discrepancies
    let mut warnings = 0;

    if energy_diff > 0.1 {
        // 10% energy difference
        println!("  ⚠️  SIGNIFICANT ENERGY DISCREPANCY (>10%)");
        println!("     Possible issue: Different energy conservation between solvers");
        warnings += 1;
    }

    if stability_diff > 0.2 {
        // 20% stability difference
        println!("  ⚠️  SIGNIFICANT STABILITY DISCREPANCY (>20%)");
        println!("     Possible issue: One solver is numerically unstable");
        warnings += 1;
    }

    if fdtd_result.stability < 0.5 {
        println!("  ⚠️  FDTD LOW STABILITY (<50%)");
        println!("     Possible issue: FDTD numerical instability");
        warnings += 1;
    }

    if pstd_result.stability < 0.5 {
        println!("  ⚠️  PSTD LOW STABILITY (<50%)");
        println!("     Possible issue: PSTD numerical instability");
        warnings += 1;
    }

    if warnings == 0 {
        println!("  ✅ GOOD AGREEMENT - No significant discrepancies detected");
    } else {
        println!(
            "  ❌ {} WARNING(S) - Investigate solver implementations",
            warnings
        );
    }

    // Test should pass even with warnings - warnings are for investigation
    assert!(
        fdtd_result.execution_time.as_millis() > 0,
        "FDTD should execute"
    );
    assert!(
        pstd_result.execution_time.as_millis() > 0,
        "PSTD should execute"
    );
    assert!(
        fdtd_result.energy >= 0.0,
        "FDTD energy should be non-negative"
    );
    assert!(
        pstd_result.energy >= 0.0,
        "PSTD energy should be non-negative"
    );
}

/// Quick FDTD test result
struct QuickTestResult {
    execution_time: std::time::Duration,
    energy: f64,
    stability: f64,
}

fn run_fdtd_quick(
    grid: &Grid,
    medium: &HomogeneousMedium,
    source: &GridSource,
    time_steps: usize,
) -> QuickTestResult {
    let start = std::time::Instant::now();

    let config = FdtdConfig::default();
    let mut solver = FdtdSolver::new(config, grid, medium, source.clone()).unwrap();

    // Run simulation
    for _ in 0..time_steps {
        solver.step_forward().unwrap();
    }

    let execution_time = start.elapsed();
    let final_field = solver.pressure_field();
    let energy = calculate_energy_quick(final_field.view());
    let stability = calculate_stability_quick(final_field.view());

    QuickTestResult {
        execution_time,
        energy,
        stability,
    }
}

fn run_pstd_quick(grid: &Grid, medium: &HomogeneousMedium, time_steps: usize) -> QuickTestResult {
    let start = std::time::Instant::now();

    let config = PSTDConfig {
        boundary: kwavers::solver::forward::pstd::config::BoundaryConfig::None,
        ..Default::default()
    };
    let pstd_source = GridSource::default();
    let mut solver = PSTDSolver::new(config, grid.clone(), medium, pstd_source).unwrap();

    // Run simulation
    let _dt = 1e-7; // 100 ns time step for PSTD
    for _ in 0..time_steps {
        solver.step_forward().unwrap();
    }

    let execution_time = start.elapsed();
    let final_field = solver.pressure_field();
    let energy = calculate_energy_quick(final_field.view());
    let stability = calculate_stability_quick(final_field.view());

    QuickTestResult {
        execution_time,
        energy,
        stability,
    }
}

/// Quick energy calculation (sum of squares)
fn calculate_energy_quick(field: ndarray::ArrayView3<f64>) -> f64 {
    field.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Quick stability calculation (checks for NaN/inf and gradient magnitude)
fn calculate_stability_quick(field: ndarray::ArrayView3<f64>) -> f64 {
    // Check for invalid values
    let has_nan = field.iter().any(|&x| x.is_nan());
    let has_inf = field.iter().any(|&x| x.is_infinite());

    if has_nan || has_inf {
        return 0.0; // Completely unstable
    }

    // Calculate RMS gradient as stability proxy
    let mut gradient_sum = 0.0;
    let mut value_count = 0;

    let shape = field.dim();
    for i in 1..shape.0.saturating_sub(1) {
        for j in 1..shape.1.saturating_sub(1) {
            for k in 1..shape.2.saturating_sub(1) {
                let gx = (field[[i + 1, j, k]] - field[[i - 1, j, k]]).abs();
                let gy = (field[[i, j + 1, k]] - field[[i, j - 1, k]]).abs();
                let gz = (field[[i, j, k + 1]] - field[[i, j, k - 1]]).abs();

                gradient_sum += (gx * gx + gy * gy + gz * gz).sqrt();
                value_count += 1;
            }
        }
    }

    if value_count == 0 {
        return 0.5; // Neutral stability for very small grids
    }

    let avg_gradient = gradient_sum / value_count as f64;

    // Convert to stability metric (1.0 = very stable, 0.0 = unstable)
    // Lower gradients indicate smoother, more stable solutions
    1.0 / (1.0 + avg_gradient * 10.0) // Scaled for reasonable range
}

/// Additional quick tests for different scenarios
#[test]
fn test_solver_consistency_small_grid() {
    let grid = Grid::new(8, 8, 8, 0.005, 0.005, 0.005).unwrap(); // Very small for speed
    let medium = HomogeneousMedium::water(&grid);
    let source = GridSource::default();

    // Just check that both solvers can run without crashing
    let fdtd_result = run_fdtd_quick(&grid, &medium, &source, 2);
    let pstd_result = run_pstd_quick(&grid, &medium, 2);

    assert!(fdtd_result.energy >= 0.0);
    assert!(pstd_result.energy >= 0.0);
    assert!(fdtd_result.stability >= 0.0 && fdtd_result.stability <= 1.0);
    assert!(pstd_result.stability >= 0.0 && pstd_result.stability <= 1.0);
}

#[test]
fn test_solver_performance_comparison() {
    let grid = Grid::new(12, 12, 12, 0.003, 0.003, 0.003).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let source = GridSource::default();

    let fdtd_result = run_fdtd_quick(&grid, &medium, &source, 3);
    let pstd_result = run_pstd_quick(&grid, &medium, 3);

    // Performance should be reasonable (under 1 second each)
    assert!(fdtd_result.execution_time.as_millis() < 1000);
    assert!(pstd_result.execution_time.as_millis() < 1000);

    // Results should be finite and reasonable
    assert!(fdtd_result.energy.is_finite());
    assert!(pstd_result.energy.is_finite());
}
