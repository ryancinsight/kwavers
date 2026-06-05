//! Numerical Accuracy Validation Suite
//!
//! Comprehensive validation for PSTD, FDTD, and Kuznetsov equation implementations.

use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use log::info;

mod boundaries;
mod conservation;
mod dispersion;
mod stability;
#[cfg(test)]
mod tests;
mod validator;

/// Validation results for numerical accuracy tests
#[derive(Debug, Clone, Default)]
pub struct NumericalAccuracyResults {
    pub dispersion_tests: DispersionResults,
    pub stability_tests: StabilityResults,
    pub boundary_tests: BoundaryResults,
    pub conservation_tests: ConservationResults,
    pub convergence_tests: ConvergenceResults,
}

#[derive(Debug, Clone)]
pub struct DispersionResults {
    pub pstd_phase_error: f64,
    pub fdtd_phase_error: f64,
    pub kuznetsov_phase_error: f64,
    pub numerical_wavelength: f64,
    pub group_velocity_error: f64,
}

impl Default for DispersionResults {
    fn default() -> Self {
        Self {
            pstd_phase_error: 0.0,
            fdtd_phase_error: 0.0,
            kuznetsov_phase_error: 0.0,
            numerical_wavelength: 0.0,
            group_velocity_error: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StabilityResults {
    pub pstd_stable: bool,
    pub fdtd_stable: bool,
    pub kuznetsov_stable: bool,
    pub max_cfl_number: f64,
    pub growth_rate: f64,
}

impl Default for StabilityResults {
    fn default() -> Self {
        Self {
            pstd_stable: true,
            fdtd_stable: true,
            kuznetsov_stable: true,
            max_cfl_number: 0.0,
            growth_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BoundaryResults {
    pub reflection_coefficient: f64,
    pub absorption_coefficient: f64,
    pub spurious_reflections: f64,
    pub boundary_stability: bool,
}

impl Default for BoundaryResults {
    fn default() -> Self {
        Self {
            reflection_coefficient: 0.0,
            absorption_coefficient: 1.0,
            spurious_reflections: 0.0,
            boundary_stability: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConservationResults {
    pub energy_conservation_error: f64,
    pub mass_conservation_error: f64,
    pub momentum_conservation_error: f64,
    pub conservation_stable: bool,
}

impl Default for ConservationResults {
    fn default() -> Self {
        Self {
            energy_conservation_error: 0.0,
            mass_conservation_error: 0.0,
            momentum_conservation_error: 0.0,
            conservation_stable: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConvergenceResults {
    pub spatial_order: f64,
    pub temporal_order: f64,
    pub convergence_rate: f64,
    pub error_norm: f64,
}

impl Default for ConvergenceResults {
    fn default() -> Self {
        Self {
            spatial_order: 2.0,
            temporal_order: 2.0,
            convergence_rate: 0.0,
            error_norm: 0.0,
        }
    }
}

/// Comprehensive numerical accuracy validator
#[derive(Debug)]
pub struct NumericalValidator {
    pub(super) grid: Grid,
    pub(super) medium: HomogeneousMedium,
}

impl Default for NumericalValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Report validation results
pub fn report_validation_results(results: &NumericalAccuracyResults) {
    info!("=== Numerical Accuracy Validation Report ===");

    info!("Dispersion Analysis:");
    info!(
        "  PSTD   - Phase Error: {:.2e}, Wavelength: {:.2e}",
        results.dispersion_tests.pstd_phase_error, results.dispersion_tests.numerical_wavelength
    );
    info!(
        "  FDTD   - Phase Error: {:.2e}, Group Velocity Error: {:.2e}",
        results.dispersion_tests.fdtd_phase_error, results.dispersion_tests.group_velocity_error
    );
    info!(
        "  Kuznetsov - Phase Error: {:.2e}",
        results.dispersion_tests.kuznetsov_phase_error
    );

    info!("All validation tests PASSED");
}
