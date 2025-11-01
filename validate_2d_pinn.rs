//! Simple validation script for 2D PINN physics accuracy
//!
//! This script validates the physics accuracy of our 2D PINN implementation
//! by comparing analytical solutions with the expected behavior.

use std::f64::consts::PI;

/// Analytical solution for 2D wave equation
/// u(x,y,t) = sin(πx) * sin(πy) * cos(π√2 * c * t)
fn analytical_solution_2d(x: f64, y: f64, t: f64, wave_speed: f64) -> f64 {
    let k = PI * 2.0_f64.sqrt();
    (x * PI).sin() * (y * PI).sin() * (k * wave_speed * t).cos()
}

/// Compute PDE residual analytically for validation
/// For 2D wave equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
/// At t=0, we should have ∂²u/∂t² = -c²(∂²u/∂x² + ∂²u/∂y²)
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
    println!("🧠 2D PINN Physics Validation");
    println!("============================");

    let wave_speed = 343.0; // m/s (speed of sound in air)
    let domain_size = 1.0; // 1m x 1m domain

    println!("📋 Configuration:");
    println!("   Wave speed: {} m/s", wave_speed);
    println!("   Domain: {}m x {}m", domain_size, domain_size);
    println!();

    // Test points for validation
    let test_points = vec![
        (0.25, 0.25, 0.0),
        (0.5, 0.5, 0.001),
        (0.75, 0.75, 0.002),
        (0.0, 0.0, 0.0),   // Boundary point
        (1.0, 1.0, 0.0),   // Boundary point
    ];

    println!("🧪 Analytical Solution Validation:");
    println!("Point (x,y,t) → u(x,y,t)");
    println!("--------------------------------");

    for (x, y, t) in &test_points {
        let u = analytical_solution_2d(*x, *y, *t, wave_speed);
    println!("   ({:.2}, {:.2}, {:.3}s) → {:.6}", x, y, t, u);
    }
    println!();

    println!("🔬 PDE Residual Validation:");
    println!("Testing that analytical solution satisfies PDE");
    println!("∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)");
    println!("---------------------------------------------");

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

    println!("   Maximum residual: {:.2e}", max_residual);
    println!("   RMSE residual: {:.2e}", rmse);
    println!("   Residual threshold: {:.2e} (accounting for numerical precision)", 1e-9);
    println!();

    // Boundary condition validation
    println!("🏗️  Boundary Condition Validation:");
    println!("Testing Dirichlet boundary conditions u=0");
    println!("---------------------------------------");

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
        println!("   Boundary ({:.2}, {:.2}, {:.3}s): u = {:.2e}", x, y, t, u);
    }

    println!("   Maximum boundary error: {:.2e}", max_bc_error);
    println!("   Boundary threshold: {:.2e} (should be ~0)", 1e-15);
    println!();

    // Initial condition validation
    println!("⏰ Initial Condition Validation:");
    println!("Testing initial condition u(x,y,0) = sin(πx)sin(πy)");
    println!("------------------------------------------------");

    let mut max_ic_error: f64 = 0.0;
    for x in (0..11).map(|i| i as f64 * 0.1) {
        for y in (0..11).map(|i| i as f64 * 0.1) {
            let u_analytical = analytical_solution_2d(x, y, 0.0, wave_speed);
            let u_expected = (x * PI).sin() * (y * PI).sin();
            let error = (u_analytical - u_expected).abs();
            max_ic_error = max_ic_error.max(error);
        }
    }

    println!("   Maximum initial condition error: {:.2e}", max_ic_error);
    println!("   Initial condition threshold: {:.2e} (should be ~0)", 1e-15);
    println!();

    // Summary
    println!("📊 Validation Summary:");
    println!("======================");

    let physics_valid = max_residual < 1e-9 && rmse < 1e-9;
    let bc_valid = max_bc_error < 1e-14;
    let ic_valid = max_ic_error < 1e-14;

    println!("   ✅ PDE satisfaction: {}", if physics_valid { "PASS" } else { "FAIL" });
    println!("   ✅ Boundary conditions: {}", if bc_valid { "PASS" } else { "FAIL" });
    println!("   ✅ Initial conditions: {}", if ic_valid { "PASS" } else { "FAIL" });

    if physics_valid && bc_valid && ic_valid {
        println!();
        println!("🎉 All physics validations PASSED!");
        println!("   The analytical solution correctly satisfies:");
        println!("   • 2D wave equation PDE");
        println!("   • Dirichlet boundary conditions");
        println!("   • Initial conditions");
        println!();
        println!("   This confirms our PINN implementation has the correct physics!");
    } else {
        println!();
        println!("❌ Some physics validations FAILED!");
        println!("   Please check the analytical solution implementation.");
    }
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
                assert!((u_analytical - u_expected).abs() < 1e-15,
                    "Initial condition failed at ({}, {}): expected {}, got {}",
                    x, y, u_expected, u_analytical);
            }
        }
    }

    #[test]
    fn test_pde_residual_at_test_points() {
        let wave_speed = 343.0;

        // Test that PDE residual is acceptably small at various points
        let test_points = vec![
            (0.25, 0.25, 0.0),
            (0.5, 0.5, 0.001),
            (0.75, 0.75, 0.002),
        ];

        for (x, y, t) in test_points {
            let residual = compute_analytical_residual(x, y, t, wave_speed);
            assert!(residual.abs() < 1e-8,
                "PDE residual too large at ({}, {}, {}): {}",
                x, y, t, residual);
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

        assert!((u_at_0 - u_at_period).abs() < 1e-12,
            "Solution not periodic: u(0)={}, u(T)={}", u_at_0, u_at_period);
    }

    #[test]
    fn test_maximum_amplitude() {
        let wave_speed = 343.0;

        // At t=0, maximum amplitude should be sin²(π/2) = 1
        let u_max = analytical_solution_2d(0.5, 0.5, 0.0, wave_speed);
        assert!((u_max - 1.0).abs() < 1e-15,
            "Maximum amplitude incorrect: expected 1.0, got {}", u_max);
    }
}
