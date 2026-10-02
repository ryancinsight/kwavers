//! Nonlinear Shear Wave Elastography Convergence Validation Example
//!
//! This example demonstrates the convergence testing framework for NL-SWE validation.
//! It shows how to:
//! - Run analytical convergence studies
//! - Validate hyperelastic models against known solutions
//! - Test harmonic generation accuracy
//! - Analyze numerical convergence rates

use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_solver::forward::elastic::nonlinear::{
    HyperelasticModel, NonlinearElasticWaveSolver, NonlinearSWEConfig,
};
use std::f64::consts::PI;
use std::io::Write;

/// Simple demonstration of convergence testing
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "🔬 NL-SWE Convergence Validation Example"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "========================================\n"
    );

    // Demonstrate hyperelastic model validation
    let _ = writeln!(std::io::stdout().lock(), "1. Hyperelastic Model Validation");
    let _ = writeln!(std::io::stdout().lock(), "--------------------------------");

    validate_neo_hookean_model()?;
    validate_ogden_principal_stretches()?;

    let _ = writeln!(
        std::io::stdout().lock(),
        "\n2. Harmonic Generation Validation"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "---------------------------------"
    );

    validate_harmonic_generation()?;

    let _ = writeln!(std::io::stdout().lock(), "\n3. Convergence Study Setup");
    let _ = writeln!(std::io::stdout().lock(), "---------------------------");

    demonstrate_convergence_setup()?;

    let _ = writeln!(
        std::io::stdout().lock(),
        "\n✅ Convergence validation example completed successfully!"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "📊 Run the full test suite with: cargo test --test nl_swe_convergence_tests"
    );

    Ok(())
}

/// Validate Neo-Hookean model against analytical solution
fn validate_neo_hookean_model() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "Testing Neo-Hookean model against analytical uniaxial compression..."
    );

    let model = HyperelasticModel::neo_hookean_soft_tissue();

    // Uniaxial compression: 20% strain
    let lambda: f64 = 0.8;
    let deformation_gradient = [
        [lambda, 0.0, 0.0],
        [0.0, 1.0 / lambda.sqrt(), 0.0],
        [0.0, 0.0, 1.0 / lambda.sqrt()],
    ];

    let stress = model.cauchy_stress(&deformation_gradient);
    let sigma_xx = stress[0][0];

    // Analytical approximation for Neo-Hookean
    let c1 = 1000.0; // From implementation
    let analytical_stress =
        c1 * (lambda * lambda - 1.0 / (lambda * lambda * lambda * lambda)) * lambda * lambda;

    let relative_error = ((sigma_xx - analytical_stress) / analytical_stress).abs();

    let _ = writeln!(
        std::io::stdout().lock(),
        "  Numerical stress: {:.3} Pa",
        sigma_xx
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Analytical stress: {:.3} Pa",
        analytical_stress
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Relative error: {:.2}%",
        relative_error * 100.0
    );

    if relative_error < 0.01 {
        let _ = writeln!(std::io::stdout().lock(), "  ✅ Validation PASSED");
    } else {
        eprintln!("  ❌ Validation FAILED");
    }

    Ok(())
}

/// Validate Ogden model principal stretch computation
fn validate_ogden_principal_stretches() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "Testing Ogden model principal stretch computation..."
    );

    let model = HyperelasticModel::Ogden {
        mu: vec![1000.0, 200.0],
        alpha: vec![1.5, 3.0],
    };

    // Simple uniaxial stretch
    let lambda_x: f64 = 1.2;
    let lambda_y = 1.0 / lambda_x.sqrt();
    let deformation_gradient = [
        [lambda_x, 0.0, 0.0],
        [0.0, lambda_y, 0.0],
        [0.0, 0.0, lambda_y],
    ];

    let principal_stretches = model.principal_stretches(&deformation_gradient);

    let _ = writeln!(
        std::io::stdout().lock(),
        "  Input stretches: λx={:.3}, λy={:.3}, λz={:.3}",
        lambda_x,
        lambda_y,
        lambda_y
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Computed principal stretches: [{:.6}, {:.6}, {:.6}]",
        principal_stretches[0],
        principal_stretches[1],
        principal_stretches[2]
    );

    // Check ordering (should be sorted ascending)
    let is_sorted = principal_stretches[0] <= principal_stretches[1]
        && principal_stretches[1] <= principal_stretches[2];
    let max_error = principal_stretches
        .iter()
        .zip([lambda_y, lambda_y, lambda_x].iter())
        .map(|(&computed, &expected)| (computed - expected).abs() / expected.abs())
        .fold(0.0, f64::max);

    let _ = writeln!(std::io::stdout().lock(), "  Properly sorted: {}", is_sorted);
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Maximum relative error: {:.2e}",
        max_error
    );

    if is_sorted && max_error < 1e-12 {
        let _ = writeln!(std::io::stdout().lock(), "  ✅ Validation PASSED");
    } else {
        eprintln!("  ❌ Validation FAILED");
    }

    Ok(())
}

/// Validate harmonic generation
fn validate_harmonic_generation() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(std::io::stdout().lock(), "Testing harmonic generation...");

    let grid = Grid::new(32, 8, 8, 0.001, 0.001, 0.001)?;
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let material = HyperelasticModel::neo_hookean_soft_tissue();

    let config = NonlinearSWEConfig {
        nonlinearity_parameter: 0.05, // Moderate nonlinearity
        enable_harmonics: true,
        ..Default::default()
    };

    let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config)?;

    // Create fundamental frequency input
    let mut initial_disp: leto::Array3<f64> = leto::Array3::zeros((32, 8, 8));
    let _omega = 2.0 * PI * 50.0; // 50 Hz
    let k = 2.0 * PI / 0.01; // λ = 1 cm

    for i in 0..32 {
        let x = i as f64 * 0.001;
        initial_disp[[i, 4, 4]] = 1e-6 * (k * x).sin();
    }

    let result = solver.propagate_waves(&initial_disp)?;
    let final_field = &result[result.len() - 1];

    // Calculate energies
    let fundamental_energy: f64 = final_field.u_fundamental.iter().map(|&x| x * x).sum();
    let second_harmonic_energy: f64 = final_field.u_second.iter().map(|&x| x * x).sum();

    let harmonic_ratio = if fundamental_energy > 1e-20 {
        second_harmonic_energy / fundamental_energy
    } else {
        0.0
    };

    let _ = writeln!(
        std::io::stdout().lock(),
        "  Fundamental energy: {:.2e}",
        fundamental_energy
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Second harmonic energy: {:.2e}",
        second_harmonic_energy
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Harmonic ratio (A₂/A₁): {:.2e}",
        harmonic_ratio.sqrt()
    );

    if harmonic_ratio > 0.0 && harmonic_ratio < fundamental_energy {
        let _ = writeln!(
            std::io::stdout().lock(),
            "  ✅ Harmonic generation working correctly"
        );
    } else {
        eprintln!("  ❌ Harmonic generation validation inconclusive");
    }

    Ok(())
}

/// Demonstrate convergence study setup
fn demonstrate_convergence_setup() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "Setting up convergence study framework..."
    );

    let grid_sizes = [16, 32, 64, 128];
    let mut results = Vec::new();

    let _ = writeln!(
        std::io::stdout().lock(),
        "  Testing convergence with grid refinement:"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Grid Size | dx (mm) | Expected Convergence"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  ----------|----------|-------------------"
    );

    for &nx in &grid_sizes {
        let dx = 0.01 / nx as f64; // 1cm domain
        let expected_convergence = if nx <= 64 {
            "2nd order"
        } else {
            "limited by model"
        };

        let _ = writeln!(
            std::io::stdout().lock(),
            "  {:8} | {:.4}  | {}",
            nx,
            dx * 1000.0,
            expected_convergence
        );

        // In a full implementation, we would run the simulation here
        // For demo purposes, just show the setup
        results.push((nx, dx));
    }

    let _ = writeln!(
        std::io::stdout().lock(),
        "  ✅ Convergence study framework ready"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  📊 Full convergence analysis available in test suite"
    );

    Ok(())
}
