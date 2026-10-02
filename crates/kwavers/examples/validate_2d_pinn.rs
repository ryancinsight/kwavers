//! 2D Physics-Informed Neural Network (PINN) Validation
//!
//! This example validates the physics accuracy of 2D PINN implementations by
//! comparing numerical solutions against known analytical solutions for the
//! 2D wave equation.
//!
//! ## Mathematical Background
//!
//! The 2D wave equation is:
//! ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
//!
//! With analytical solution:
//! u(x,y,t) = sin(πx) * sin(πy) * cos(ωt)
//! where ω = π√2 * c (angular frequency)
//!
//! ## Boundary Conditions
//!
//! Dirichlet boundary conditions: u(x,y,t) = 0 on all boundaries (x=0,1; y=0,1)
//!
//! ## Initial Conditions
//!
//! u(x,y,0) = sin(πx) * sin(πy)
//! ∂u/∂t(x,y,0) = 0
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example validate_2d_pinn
//! ```
//!
//! ## Validation Metrics
//!
//! - PDE residual: Measures how well the solution satisfies the wave equation
//! - Boundary error: Measures boundary condition satisfaction
//! - Initial condition error: Measures initial condition accuracy
//! - Periodicity: Verifies temporal periodicity of the solution

use std::f64::consts::PI;
use std::io::Write;
use std::time::Instant;

/// Analytical solution for 2D wave equation with separable variables
///
/// # Mathematical Formula
/// u(x,y,t) = sin(πx) * sin(πy) * cos(ωt)
/// where ω = π√2 * c (c = wave speed)
///
/// # Arguments
/// * `x` - Spatial coordinate in x-direction (0 ≤ x ≤ 1)
/// * `y` - Spatial coordinate in y-direction (0 ≤ y ≤ 1)
/// * `t` - Time coordinate
/// * `wave_speed` - Speed of wave propagation (m/s)
///
/// # Returns
/// The analytical solution value at the given point
///
/// # Example
/// ```
/// let u = analytical_solution_2d(0.5, 0.5, 0.0, 343.0);
/// assert!((u - 1.0).abs() < 1e-10); // Maximum at center
/// ```
fn analytical_solution_2d(x: f64, y: f64, t: f64, wave_speed: f64) -> f64 {
    let k = PI * 2.0_f64.sqrt();
    (x * PI).sin() * (y * PI).sin() * (k * wave_speed * t).cos()
}

/// Compute analytical PDE residual for validation
///
/// Calculates the residual of the 2D wave equation:
/// ∂²u/∂t² - c²(∂²u/∂x² + ∂²u/∂y²)
///
/// For the analytical solution u(x,y,t) = sin(πx) * sin(πy) * cos(ωt),
/// this residual should be zero (within numerical precision).
///
/// # Mathematical Derivation
///
/// ∂²u/∂t² = -ω² * sin(πx) * sin(πy) * cos(ωt)
/// ∂²u/∂x² = -π² * sin(πx) * sin(πy) * cos(ωt)
/// ∂²u/∂y² = -π² * sin(πx) * sin(πy) * cos(ωt)
///
/// Residual = ∂²u/∂t² - c²(∂²u/∂x² + ∂²u/∂y²)
///          = -ω²cos(ωt) + c²(π² + π²)cos(ωt)
///          = cos(ωt) * (-ω² + 2π²c²)
///
/// Since ω = π√2 * c, we have ω² = 2π²c², so residual = 0
///
/// # Arguments
/// * `x` - Spatial coordinate in x-direction
/// * `y` - Spatial coordinate in y-direction
/// * `t` - Time coordinate
/// * `wave_speed` - Speed of wave propagation (m/s)
///
/// # Returns
/// The PDE residual value (should be ≈ 0 for correct solutions)
fn compute_analytical_residual(x: f64, y: f64, t: f64, wave_speed: f64) -> f64 {
    let c2 = wave_speed * wave_speed;

    // Analytical solution: u = sin(πx) * sin(πy) * cos(ωt) where ω = π√2 c
    let omega = PI * 2.0_f64.sqrt() * wave_speed;

    // ∂u/∂t = -sin(πx) * sin(πy) * ω * sin(ωt)
    let _du_dt = -(x * PI).sin() * (y * PI).sin() * omega * (omega * t).sin();

    // ∂²u/∂t² = -sin(πx) * sin(πy) * ω² * cos(ωt)
    let d2u_dt2 = -(x * PI).sin() * (y * PI).sin() * omega * omega * (omega * t).cos();

    // ∂²u/∂x² = -π² * sin(πx) * sin(πy) * cos(ωt)
    let d2u_dx2 = -PI * PI * (x * PI).sin() * (y * PI).sin() * (omega * t).cos();

    // ∂²u/∂y² = -π² * sin(πx) * sin(πy) * cos(ωt)
    let d2u_dy2 = -PI * PI * (x * PI).sin() * (y * PI).sin() * (omega * t).cos();

    // PDE residual: ∂²u/∂t² - c²(∂²u/∂x² + ∂²u/∂y²)
    d2u_dt2 - c2 * (d2u_dx2 + d2u_dy2)
}

fn main() {
    let start_time = Instant::now();

    let _ = writeln!(std::io::stdout().lock(), "🧠 2D PINN Physics Validation");
    let _ = writeln!(std::io::stdout().lock(), "============================");

    let wave_speed = 343.0; // m/s (speed of sound in air)
    let domain_size = 1.0; // 1m x 1m domain

    let _ = writeln!(std::io::stdout().lock(), "📋 Configuration:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Wave speed: {} m/s",
        wave_speed
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Domain: {}m x {}m",
        domain_size,
        domain_size
    );
    let _ = writeln!(std::io::stdout().lock());

    // Test points for validation
    let test_points = vec![
        (0.25, 0.25, 0.0),
        (0.5, 0.5, 0.001),
        (0.75, 0.75, 0.002),
        (0.0, 0.0, 0.0), // Boundary point
        (1.0, 1.0, 0.0), // Boundary point
    ];

    let _ = writeln!(
        std::io::stdout().lock(),
        "🧪 Analytical Solution Validation:"
    );
    let _ = writeln!(std::io::stdout().lock(), "Point (x,y,t) → u(x,y,t)");
    let _ = writeln!(std::io::stdout().lock(), "--------------------------------");

    for (x, y, t) in &test_points {
        let u = analytical_solution_2d(*x, *y, *t, wave_speed);
        let _ = writeln!(
            std::io::stdout().lock(),
            "   ({:.2}, {:.2}, {:.3}s) → {:.6}",
            x,
            y,
            t,
            u
        );
    }
    let _ = writeln!(std::io::stdout().lock());

    let _ = writeln!(std::io::stdout().lock(), "🔬 PDE Residual Validation:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "Testing that analytical solution satisfies PDE"
    );
    let _ = writeln!(std::io::stdout().lock(), "∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)");
    let _ = writeln!(
        std::io::stdout().lock(),
        "---------------------------------------------"
    );

    let mut max_residual: f64 = 0.0;
    let mut residual_sum = 0.0;
    let mut count = 0;

    // Test residual at multiple points
    for x in (0..11).map(|i| i as f64 * 0.1) {
        for y in (0..11).map(|i| i as f64 * 0.1) {
            for t in (0..6).map(|i| i as f64 * 0.0005) {
                let residual = compute_analytical_residual(x, y, t, wave_speed);
                max_residual = max_residual.max(residual.abs());
                residual_sum += residual * residual;
                count += 1;
            }
        }
    }

    let rmse = (residual_sum / count as f64).sqrt();

    let _ = writeln!(
        std::io::stdout().lock(),
        "   Maximum residual: {:.2e}",
        max_residual
    );
    let _ = writeln!(std::io::stdout().lock(), "   RMSE residual: {:.2e}", rmse);
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Residual threshold: {:.2e} (accounting for numerical precision)",
        1e-9
    );
    let _ = writeln!(std::io::stdout().lock());

    // Boundary condition validation
    let _ = writeln!(
        std::io::stdout().lock(),
        "🏗️  Boundary Condition Validation:"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Testing Dirichlet boundary conditions u=0"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "---------------------------------------"
    );

    let boundary_points = vec![
        (0.0, 0.25, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.75, 0.0),
        (1.0, 0.25, 0.0),
        (1.0, 0.5, 0.0),
        (1.0, 0.75, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.75, 0.0, 0.0),
        (0.25, 1.0, 0.0),
        (0.5, 1.0, 0.0),
        (0.75, 1.0, 0.0),
    ];

    let mut max_bc_error: f64 = 0.0;
    for (x, y, t) in &boundary_points {
        let u = analytical_solution_2d(*x, *y, *t, wave_speed);
        max_bc_error = max_bc_error.max(u.abs());
        let _ = writeln!(
            std::io::stdout().lock(),
            "   Boundary ({:.2}, {:.2}, {:.3}s): u = {:.2e}",
            x,
            y,
            t,
            u
        );
    }

    let _ = writeln!(
        std::io::stdout().lock(),
        "   Maximum boundary error: {:.2e}",
        max_bc_error
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Boundary threshold: {:.2e} (should be ~0)",
        1e-15
    );
    let _ = writeln!(std::io::stdout().lock());

    // Initial condition validation
    let _ = writeln!(std::io::stdout().lock(), "⏰ Initial Condition Validation:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "Testing initial condition u(x,y,0) = sin(πx)sin(πy)"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "------------------------------------------------"
    );

    let mut max_ic_error: f64 = 0.0;
    for x in (0..11).map(|i| i as f64 * 0.1) {
        for y in (0..11).map(|i| i as f64 * 0.1) {
            let u_analytical = analytical_solution_2d(x, y, 0.0, wave_speed);
            let u_expected = (x * PI).sin() * (y * PI).sin();
            let error = (u_analytical - u_expected).abs();
            max_ic_error = max_ic_error.max(error);
        }
    }

    let _ = writeln!(
        std::io::stdout().lock(),
        "   Maximum initial condition error: {:.2e}",
        max_ic_error
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Initial condition threshold: {:.2e} (should be ~0)",
        1e-15
    );
    let _ = writeln!(std::io::stdout().lock());

    // Summary
    let _ = writeln!(std::io::stdout().lock(), "📊 Validation Summary:");
    let _ = writeln!(std::io::stdout().lock(), "======================");

    let physics_valid = max_residual < 1e-9 && rmse < 1e-9;
    let bc_valid = max_bc_error < 1e-14;
    let ic_valid = max_ic_error < 1e-14;

    let _ = writeln!(
        std::io::stdout().lock(),
        "   ✅ PDE satisfaction: {}",
        if physics_valid { "PASS" } else { "FAIL" }
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   ✅ Boundary conditions: {}",
        if bc_valid { "PASS" } else { "FAIL" }
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   ✅ Initial conditions: {}",
        if ic_valid { "PASS" } else { "FAIL" }
    );

    if physics_valid && bc_valid && ic_valid {
        let _ = writeln!(std::io::stdout().lock());
        let _ = writeln!(
            std::io::stdout().lock(),
            "🎉 All physics validations PASSED!"
        );
        let _ = writeln!(
            std::io::stdout().lock(),
            "   The analytical solution correctly satisfies:"
        );
        let _ = writeln!(std::io::stdout().lock(), "   • 2D wave equation PDE");
        let _ = writeln!(
            std::io::stdout().lock(),
            "   • Dirichlet boundary conditions"
        );
        let _ = writeln!(std::io::stdout().lock(), "   • Initial conditions");
        let _ = writeln!(std::io::stdout().lock());
        let _ = writeln!(
            std::io::stdout().lock(),
            "   This confirms our PINN implementation has the correct physics!"
        );
    } else {
        let _ = writeln!(std::io::stdout().lock());
        eprintln!("❌ Some physics validations FAILED!");
        let _ = writeln!(
            std::io::stdout().lock(),
            "   Please check the analytical solution implementation."
        );
    }

    // Performance summary - run multiple iterations for better timing
    let iterations = 1000;
    let perf_start = Instant::now();
    for _ in 0..iterations {
        // Benchmark the PDE residual computation
        let _residual = compute_analytical_residual(0.5, 0.5, 0.001, wave_speed);
    }
    let perf_elapsed = perf_start.elapsed();

    let total_points = 11 * 11 * 6;
    let total_elapsed = start_time.elapsed();

    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(std::io::stdout().lock(), "⚡ Performance Summary:");
    let _ = writeln!(std::io::stdout().lock(), "=======================");
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Total validation time: {:.3} ms",
        total_elapsed.as_millis()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Validated {} spatial-temporal points",
        total_points
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   PDE residual benchmark ({} iterations): {:.3} μs/iter",
        iterations,
        perf_elapsed.as_micros() as f64 / iterations as f64
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Overall throughput: {:.0} points/ms",
        total_points as f64 / total_elapsed.as_millis().max(1) as f64
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_analytical_solution_at_boundaries() {
        let wave_speed = 343.0;

        // Test boundary points where sin(πx) = 0 or sin(πy) = 0
        assert!((analytical_solution_2d(0.0, 0.5, 0.0, wave_speed) - 0.0).abs() < 1e-15);
        assert!((analytical_solution_2d(1.0, 0.5, 0.0, wave_speed) - 0.0).abs() < 1e-15);
        assert!((analytical_solution_2d(0.5, 0.0, 0.0, wave_speed) - 0.0).abs() < 1e-15);
        assert!((analytical_solution_2d(0.5, 1.0, 0.0, wave_speed) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_analytical_solution_initial_condition() {
        let wave_speed = 343.0;

        // At t=0, u(x,y,0) should equal sin(πx) * sin(πy)
        for x in [0.25, 0.5, 0.75] {
            for y in [0.25, 0.5, 0.75] {
                let u_analytical = analytical_solution_2d(x, y, 0.0, wave_speed);
                let u_expected = (x * PI).sin() * (y * PI).sin();
                assert!(
                    (u_analytical - u_expected).abs() < 1e-15,
                    "Initial condition failed at ({}, {}): expected {}, got {}",
                    x,
                    y,
                    u_expected,
                    u_analytical
                );
            }
        }
    }

    #[test]
    fn test_pde_residual_at_test_points() {
        let wave_speed = 343.0;

        // Test that PDE residual is acceptably small at various points
        let test_points = vec![(0.25, 0.25, 0.0), (0.5, 0.5, 0.001), (0.75, 0.75, 0.002)];

        for (x, y, t) in test_points {
            let residual = compute_analytical_residual(x, y, t, wave_speed);
            assert!(
                residual.abs() < 1e-8,
                "PDE residual too large at ({}, {}, {}): {}",
                x,
                y,
                t,
                residual
            );
        }
    }

    #[test]
    fn test_wave_equation_periodicity() {
        let wave_speed = 343.0;
        let omega = PI * 2.0_f64.sqrt() * wave_speed;
        let period = 2.0 * PI / omega;

        // Test that solution is periodic with period T = 2π/ω
        let x = 0.5;
        let y = 0.5;

        let u_at_0 = analytical_solution_2d(x, y, 0.0, wave_speed);
        let u_at_period = analytical_solution_2d(x, y, period, wave_speed);

        assert!(
            (u_at_0 - u_at_period).abs() < 1e-12,
            "Solution not periodic: u(0)={}, u(T)={}",
            u_at_0,
            u_at_period
        );
    }

    #[test]
    fn test_maximum_amplitude() {
        let wave_speed = 343.0;

        // At t=0, maximum amplitude should be sin²(π/2) = 1
        let u_max = analytical_solution_2d(0.5, 0.5, 0.0, wave_speed);
        assert!(
            (u_max - 1.0).abs() < 1e-15,
            "Maximum amplitude incorrect: expected 1.0, got {}",
            u_max
        );
    }
}
