//! Rigorous validation against literature benchmarks

// Re-export existing validation types for compatibility
pub use crate::solver::validation::{KWaveTestCase, KWaveValidator, ValidationReport};

/// Validation result struct (legacy compatibility)
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

impl ValidationResult {
    #[must_use]
    pub fn success() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
        }
    }

    #[must_use]
    pub fn failure(error: String) -> Self {
        Self {
            is_valid: false,
            errors: vec![error],
        }
    }

    #[must_use]
    pub fn from_errors(errors: Vec<String>) -> Self {
        Self {
            is_valid: errors.is_empty(),
            errors,
        }
    }
}

/// Trait for validatable components
pub trait Validatable {
    fn validate(&self) -> ValidationResult;
}

// Rigorous validation against literature benchmarks
//
// This module validates implementations against:
// - Taflove & Hagness (2005) for FDTD
// - Roden & Gedney (2000) for CPML
// - Szabo (1994) for fractional absorption
// - k-Wave benchmarks for accuracy

use crate::grid::Grid;

/// Physical constants for validation
pub mod constants {
    /// Speed of sound in water at 20°C (m/s) - NIST reference
    pub const SOUND_SPEED_WATER: f64 = 1482.0;

    /// Absorption in water at 1 `MHz` (dB/cm/MHz²) - Pinkerton (1949)
    pub const WATER_ABSORPTION: f64 = 0.0022;

    /// CFL limit for 3D FDTD - Taflove & Hagness (2005) Eq. 4.92
    pub const CFL_LIMIT_3D: f64 = 0.577350269; // 1/sqrt(3)
}

// Re-export constants for backward compatibility
pub use constants::SOUND_SPEED_WATER;

/// Physics validation results (extends `ValidationReport`)
#[derive(Debug)]
pub struct PhysicsValidation {
    pub test_name: String,
    pub passed: bool,
    pub error_l2: f64,
    pub error_linf: f64,
    pub reference: String,
}

/// Validate CFL condition implementation
pub fn validate_cfl_condition(grid: &Grid, dt: f64, c_max: f64) -> PhysicsValidation {
    let dx_min = grid.dx.min(grid.dy).min(grid.dz);
    let cfl_number = c_max * dt / dx_min;
    let theoretical_limit = constants::CFL_LIMIT_3D;

    PhysicsValidation {
        test_name: "CFL Stability Condition".to_string(),
        passed: cfl_number <= theoretical_limit,
        error_l2: (cfl_number - theoretical_limit).abs(),
        error_linf: cfl_number / theoretical_limit,
        reference: "Taflove & Hagness (2005) Eq. 4.92".to_string(),
    }
}

/// Validate dispersion relation for FDTD
pub fn validate_fdtd_dispersion(grid: &Grid, dt: f64, frequency: f64, c: f64) -> PhysicsValidation {
    // Numerical wavenumber from FDTD dispersion relation
    // Taflove & Hagness (2005) Eq. 4.110
    let k_exact = 2.0 * std::f64::consts::PI * frequency / c;
    let dx = grid.dx;

    // Simplified 1D dispersion relation for validation testing
    // Uses basic 1D approximation ω = ck for initial validation.
    // Full 3D dispersion analysis is performed in validation test suite.
    let omega = 2.0 * std::f64::consts::PI * frequency;
    let k_num = (2.0 / dx) * ((omega * dt / 2.0).sin() / (c * dt / dx)).asin();

    let error = (k_num - k_exact).abs() / k_exact;

    PhysicsValidation {
        test_name: "FDTD Dispersion Relation".to_string(),
        passed: error < 0.01, // 1% error threshold
        error_l2: error,
        error_linf: error,
        reference: "Taflove & Hagness (2005) Eq. 4.110".to_string(),
    }
}

/// Validate absorption implementation against Szabo model
pub fn validate_absorption_model(
    frequency: f64,
    measured_alpha: f64,
    medium: &dyn crate::medium::Medium,
    grid: &Grid,
) -> PhysicsValidation {
    // Szabo's power law: α = α₀|ω|^y
    // For water: y = 2, α₀ = 0.0022 dB/(cm·MHz²)

    let alpha_theory = constants::WATER_ABSORPTION * (frequency / 1e6).powi(2);
    let alpha_np_m = alpha_theory * 0.1151; // Convert dB/cm to Np/m

    // Get implementation's absorption
    use crate::medium::AcousticProperties;
    let alpha_impl =
        AcousticProperties::absorption_coefficient(medium, 0.0, 0.0, 0.0, grid, frequency);

    // Compare with measured value if provided
    let measurement_error = if measured_alpha > 0.0 {
        (alpha_impl - measured_alpha).abs() / measured_alpha
    } else {
        0.0
    };

    let theory_error = (alpha_impl - alpha_np_m).abs() / alpha_np_m;
    let total_error = theory_error.max(measurement_error);

    PhysicsValidation {
        test_name: "Power Law Absorption (Szabo)".to_string(),
        passed: total_error < 0.05, // 5% error threshold
        error_l2: theory_error,
        error_linf: total_error,
        reference: "Szabo (1994) J. Acoust. Soc. Am. 96(1)".to_string(),
    }
}

/// Validate CPML boundary implementation
#[must_use]
pub fn validate_cpml_reflection(reflection_coefficient: f64) -> PhysicsValidation {
    // Roden & Gedney (2000) achieved R < 10^-5 for normal incidence
    let target_reflection = 1e-5;

    PhysicsValidation {
        test_name: "CPML Reflection Coefficient".to_string(),
        passed: reflection_coefficient < target_reflection,
        error_l2: reflection_coefficient,
        error_linf: reflection_coefficient / target_reflection,
        reference: "Roden & Gedney (2000) Microwave Opt. Tech. Lett. 27(5)".to_string(),
    }
}

/// Run all validations
pub fn validate_all(
    grid: &Grid,
    dt: f64,
    medium: &dyn crate::medium::Medium,
) -> Vec<PhysicsValidation> {
    let mut reports = Vec::new();

    // CFL validation
    // Calculate max sound speed directly since we have a trait object
    let mut c_max = 0.0;
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                let speed = crate::medium::sound_speed_at(medium, x, y, z, grid);
                if speed > c_max {
                    c_max = speed;
                }
            }
        }
    }
    reports.push(validate_cfl_condition(grid, dt, c_max));

    // Dispersion validation
    reports.push(validate_fdtd_dispersion(grid, dt, 1e6, c_max));

    // Absorption validation
    reports.push(validate_absorption_model(1e6, 0.0, medium, grid));

    // Print summary
    println!("\n=== VALIDATION REPORT ===");
    for report in &reports {
        println!(
            "{}: {}",
            report.test_name,
            if report.passed {
                "PASSED ✓"
            } else {
                "FAILED ✗"
            }
        );
        if !report.passed {
            println!("  L2 Error: {:.2e}", report.error_l2);
            println!("  Reference: {}", report.reference);
        }
    }

    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfl_validation() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
        let dt = 1e-7;
        let c = 1500.0;

        let report = validate_cfl_condition(&grid, dt, c);
        assert!(
            report.passed,
            "CFL condition should pass for conservative timestep"
        );
    }
}
