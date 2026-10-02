//! Theorem Validation Demonstration
//!
//! This example demonstrates the systematic validation of mathematical theorems
//! implemented in Kwavers with quantitative error bounds and convergence proofs.

use kwavers_analysis::validation::theorem_validation::{TheoremValidation, TheoremValidator};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "🔬 Theorem Validation Demonstration"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "==================================="
    );

    let validator = TheoremValidator;

    // Run comprehensive theorem validation
    let _ = writeln!(
        std::io::stdout().lock(),
        "Running comprehensive theorem validation suite..."
    );
    let validations = validator.run_comprehensive_validation();

    let _ = writeln!(
        std::io::stdout().lock(),
        "Validated {} mathematical theorems",
        validations.len()
    );

    // Display results
    display_validation_results(&validations);

    // Generate detailed report
    let report = validator.generate_validation_report(&validations);
    let _ = writeln!(std::io::stdout().lock(), "\n{}", report);

    // Demonstrate individual theorem validations
    demonstrate_individual_validations()?;

    let _ = writeln!(
        std::io::stdout().lock(),
        "\n✅ Theorem validation demonstration completed!"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "   Demonstrated: Systematic theorem validation with quantitative error bounds"
    );

    Ok(())
}

fn display_validation_results(validations: &[TheoremValidation]) {
    let _ = writeln!(std::io::stdout().lock(), "\n📊 Validation Results Summary:");
    let _ = writeln!(std::io::stdout().lock(), "------------------------------");

    let total = validations.len();
    let passed = validations.iter().filter(|v| v.passed).count();
    let pass_rate = passed as f64 / total as f64 * 100.0;

    let _ = writeln!(std::io::stdout().lock(), "Total Theorems: {}", total);
    let _ = writeln!(
        std::io::stdout().lock(),
        "Passed: {} ({:.1}%)",
        passed,
        pass_rate
    );
    eprintln!("Failed: {} ({:.1}%)", total - passed, 100.0 - pass_rate);

    let _ = writeln!(std::io::stdout().lock(), "\n📋 Detailed Results:");
    let _ = writeln!(std::io::stdout().lock(), "-------------------");

    for (i, validation) in validations.iter().enumerate() {
        let status = if validation.passed { "✅" } else { "❌" };
        let confidence_pct = validation.confidence * 100.0;

        let _ = writeln!(
            std::io::stdout().lock(),
            "{}. {} {} - {:.1}% confidence",
            i + 1,
            validation.theorem,
            status,
            confidence_pct
        );

        if !validation.passed {
            eprintln!(
                "   Error: {:.2e} (bound: {:.2e})",
                validation.measured_error, validation.error_bound
            );
        }
    }
}

fn demonstrate_individual_validations() -> Result<(), Box<dyn std::error::Error>> {
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n🎯 Individual Theorem Demonstrations"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "===================================="
    );

    // Demonstrate Beer-Lambert law validation
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n1. Beer-Lambert Law Validation:"
    );
    let _ = writeln!(std::io::stdout().lock(), "-------------------------------");
    let distances: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let alpha: f64 = 0.1; // 1/m
    let initial_intensity: f64 = 1.0;

    // Generate theoretical intensities
    let theoretical_intensities: Vec<f64> = distances
        .iter()
        .map(|&d| initial_intensity * (-alpha * d).exp())
        .collect();

    // Add some measurement noise
    let measured_intensities: Vec<f64> = theoretical_intensities
        .iter()
        .enumerate()
        .map(|(i, &theoretical)| {
            let noise = if i == 0 {
                0.0
            } else {
                0.01 * (i as f64).sqrt()
            };
            theoretical * (1.0 + noise)
        })
        .collect();

    let result = TheoremValidator::validate_beer_lambert_law(
        initial_intensity,
        alpha,
        &distances,
        &measured_intensities,
    );

    let _ = writeln!(std::io::stdout().lock(), "Theorem: {}", result.theorem);
    eprintln!(
        "Status: {}",
        if result.passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    eprintln!("Max Error: {:.2e}", result.measured_error);
    let _ = writeln!(
        std::io::stdout().lock(),
        "Theoretical Bound: {:.2e}",
        result.error_bound
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Confidence: {:.1}%",
        result.confidence * 100.0
    );
    let _ = writeln!(std::io::stdout().lock(), "Details: {}", result.details);

    // Demonstrate CFL condition validation
    let _ = writeln!(std::io::stdout().lock(), "\n2. CFL Condition Validation:");
    let _ = writeln!(std::io::stdout().lock(), "----------------------------");
    let dt = 1e-7; // 0.1 μs
    let dx = 1e-4; // 0.1 mm
    let c = 1500.0; // m/s
    let dimensions = 3;

    let cfl_result = TheoremValidator::validate_cfl_condition(dt, dx, c, dimensions);

    let _ = writeln!(std::io::stdout().lock(), "Theorem: {}", cfl_result.theorem);
    eprintln!(
        "Status: {}",
        if cfl_result.passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "CFL Number: {:.3}",
        cfl_result.measured_error
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Stability Limit: {:.3}",
        cfl_result.error_bound
    );
    let _ = writeln!(std::io::stdout().lock(), "Details: {}", cfl_result.details);

    // Demonstrate PINN convergence validation
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n3. PINN Convergence Theorem Validation:"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "---------------------------------------"
    );
    let n_collocation = 2000;
    let network_width = 100;
    let measured_error = 0.005; // 0.5% error
    let solution_smoothness = 1.0;

    let pinn_result = TheoremValidator::validate_pinn_convergence(
        n_collocation,
        network_width,
        measured_error,
        solution_smoothness,
    );

    let _ = writeln!(std::io::stdout().lock(), "Theorem: {}", pinn_result.theorem);
    eprintln!(
        "Status: {}",
        if pinn_result.passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    eprintln!("Measured Error: {:.2e}", pinn_result.measured_error);
    let _ = writeln!(
        std::io::stdout().lock(),
        "Theoretical Bound: {:.2e}",
        pinn_result.error_bound
    );
    let _ = writeln!(std::io::stdout().lock(), "Details: {}", pinn_result.details);

    // Demonstrate MUSIC resolution validation
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n4. MUSIC Resolution Theorem Validation:"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "---------------------------------------"
    );
    let array_length = 0.05; // 5 cm
    let wavelength = 0.0003; // 0.3 mm (1 MHz in water)
    let snr_db = 25.0;
    let measured_resolution = 0.005; // 5 mrad

    let music_result = TheoremValidator::validate_music_resolution(
        array_length,
        wavelength,
        snr_db,
        measured_resolution,
    );

    let _ = writeln!(
        std::io::stdout().lock(),
        "Theorem: {}",
        music_result.theorem
    );
    eprintln!(
        "Status: {}",
        if music_result.passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Measured Resolution: {:.2e} rad",
        music_result.measured_error
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Theoretical Bound: {:.2e} rad",
        music_result.error_bound
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Details: {}",
        music_result.details
    );

    // Demonstrate coded excitation validation
    let _ = writeln!(
        std::io::stdout().lock(),
        "\n5. Coded Excitation SNR Theorem Validation:"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "-------------------------------------------"
    );
    let code_length = 512;
    let compression_ratio = 4.0;
    let measured_snr_improvement = 35.0; // Linear scale

    let ce_result = TheoremValidator::validate_coded_excitation_snr(
        code_length,
        compression_ratio,
        measured_snr_improvement,
    );

    let _ = writeln!(std::io::stdout().lock(), "Theorem: {}", ce_result.theorem);
    eprintln!(
        "Status: {}",
        if ce_result.passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Measured SNR: {:.1} dB",
        10.0 * ce_result.measured_error.log10()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Theoretical SNR: {:.1} dB",
        10.0 * ce_result.error_bound.log10()
    );
    let _ = writeln!(std::io::stdout().lock(), "Details: {}", ce_result.details);

    Ok(())
}
