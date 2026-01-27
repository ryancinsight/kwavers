//! Shared Validation Framework for Physics Solvers
//!
//! This module provides a trait-based validation framework that can test ANY
//! implementation of physics solvers (FDTD, PSTD, PINN, FEM, etc.) against
//! analytical solutions and mathematical specifications.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ ValidationSuite Trait (solver-agnostic interface)           │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │ AnalyticalSolution Trait (ground truth specifications)      │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Concrete Solutions: Plane Wave, Lamb Problem, Point Source  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Design Principles
//!
//! 1. **Mathematical Rigor**: All solutions have closed-form expressions with
//!    analytical derivatives for gradient validation
//! 2. **Solver Agnostic**: Validation operates on trait interfaces, not concrete types
//! 3. **Composable**: Validation tests can be combined and reused
//! 4. **Traceable**: Every validation links to mathematical specifications (theorems)
//!
//! # Validation Hierarchy
//!
//! ## Level 1: Material Property Validation
//! - Density positivity: ρ > 0
//! - Elastic moduli bounds: μ > 0, λ > -2μ/3
//! - Thermodynamic stability: K = λ + 2μ/3 > 0
//! - Poisson's ratio: -1 < ν < 0.5
//!
//! ## Level 2: Wave Speed Validation
//! - P-wave speed: cₚ = √((λ + 2μ)/ρ)
//! - S-wave speed: cₛ = √(μ/ρ)
//! - Speed relationship: cₚ > cₛ (always)
//!
//! ## Level 3: PDE Residual Validation
//! - Elastic wave equation satisfaction
//! - Boundary condition enforcement
//! - Initial condition consistency
//!
//! ## Level 4: Energy Conservation
//! - Hamiltonian invariance: dH/dt = 0 (no dissipation)
//! - Kinetic energy: K = (1/2)∫ρ|∂u/∂t|²dV
//! - Strain energy: U = (1/2)∫σ:ε dV
//!
//! ## Level 5: Analytical Solution Convergence
//! - Plane wave propagation (exact dispersion)
//! - Lamb's problem (point source response)
//! - Spherical wave expansion
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use kwavers::tests::validation::{AnalyticalSolution, PlaneWaveSolution};
//!
//! // Define analytical solution
//! let plane_wave = PlaneWaveSolution::new(
//!     amplitude: 1.0,
//!     wavelength: 0.01,
//!     direction: [1.0, 0.0, 0.0],
//!     wave_type: WaveType::PWave,
//! );
//!
//! // Validate solver against analytical solution
//! let error = plane_wave.validate_solver(&my_pinn_solver, tolerance: 1e-3);
//! assert!(error.l2_error < 1e-3);
//! ```

pub mod analytical_solutions;
pub mod convergence;
pub mod energy;
pub mod error_metrics;

use std::fmt;

// ============================================================================
// Core Validation Traits
// ============================================================================

/// Analytical solution with closed-form expressions for validation
///
/// # Mathematical Specification
///
/// An analytical solution must provide:
/// 1. Displacement field: u(x, t)
/// 2. Velocity field: v(x, t) = ∂u/∂t
/// 3. Stress field: σ(x, t) (computed from strain)
/// 4. Spatial derivatives: ∇u, ∇²u (for PDE residual checks)
/// 5. Temporal derivatives: ∂u/∂t, ∂²u/∂t² (for wave equation verification)
///
/// # Invariants
///
/// - All returned tensors must have consistent dimensions
/// - Derivatives must satisfy compatibility: ∂ᵢ∂ⱼu = ∂ⱼ∂ᵢu
/// - Energy must be conserved: H(t) = const (for undamped systems)
pub trait AnalyticalSolution: Send + Sync {
    /// Displacement field at spatial point x and time t
    ///
    /// # Mathematical Specification
    /// Returns u(x, t) ∈ ℝⁿ where n is the spatial dimension
    fn displacement(&self, x: &[f64], t: f64) -> Vec<f64>;

    /// Velocity field: ∂u/∂t
    ///
    /// # Mathematical Specification
    /// Returns v(x, t) = ∂u/∂t computed analytically (not via finite differences)
    fn velocity(&self, x: &[f64], t: f64) -> Vec<f64>;

    /// Spatial gradient: ∇u (Jacobian matrix)
    ///
    /// # Mathematical Specification
    /// Returns J[i,j] = ∂uᵢ/∂xⱼ as a flattened row-major matrix
    /// For 2D: [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]
    fn gradient(&self, x: &[f64], t: f64) -> Vec<f64>;

    /// Strain tensor: ε = (1/2)(∇u + ∇uᵀ)
    ///
    /// # Mathematical Specification
    /// Returns symmetric strain tensor in Voigt notation:
    /// - 2D: [εₓₓ, εᵧᵧ, 2εₓᵧ]
    /// - 3D: [εₓₓ, εᵧᵧ, εᵤᵤ, 2εₓᵧ, 2εᵧᵤ, 2εᵤₓ]
    fn strain(&self, x: &[f64], t: f64) -> Vec<f64> {
        let grad = self.gradient(x, t);
        let dim = self.spatial_dimension();

        match dim {
            2 => {
                // grad = [∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]
                let du_dx = grad[0];
                let du_dy = grad[1];
                let dv_dx = grad[2];
                let dv_dy = grad[3];

                // Voigt notation: [εₓₓ, εᵧᵧ, 2εₓᵧ]
                vec![du_dx, dv_dy, du_dy + dv_dx]
            }
            3 => {
                // grad = [∂u/∂x, ∂u/∂y, ∂u/∂z, ∂v/∂x, ∂v/∂y, ∂v/∂z, ∂w/∂x, ∂w/∂y, ∂w/∂z]
                let du_dx = grad[0];
                let du_dy = grad[1];
                let du_dz = grad[2];
                let dv_dx = grad[3];
                let dv_dy = grad[4];
                let dv_dz = grad[5];
                let dw_dx = grad[6];
                let dw_dy = grad[7];
                let dw_dz = grad[8];

                // Voigt notation: [εₓₓ, εᵧᵧ, εᵤᵤ, 2εₓᵧ, 2εᵧᵤ, 2εᵤₓ]
                vec![
                    du_dx,
                    dv_dy,
                    dw_dz,
                    du_dy + dv_dx,
                    dv_dz + dw_dy,
                    dw_dx + du_dz,
                ]
            }
            _ => panic!("Unsupported spatial dimension: {}", dim),
        }
    }

    /// Stress tensor computed from strain via constitutive relation
    ///
    /// # Mathematical Specification
    /// For isotropic linear elasticity:
    /// σ = λ tr(ε)I + 2μ ε
    ///
    /// Returns stress in Voigt notation (same ordering as strain)
    fn stress(&self, x: &[f64], t: f64, lambda: f64, mu: f64) -> Vec<f64> {
        let strain = self.strain(x, t);
        let dim = self.spatial_dimension();

        match dim {
            2 => {
                let exx = strain[0];
                let eyy = strain[1];
                let exy = strain[2] / 2.0; // Convert from Voigt back to tensor
                let trace = exx + eyy;

                vec![
                    lambda * trace + 2.0 * mu * exx,
                    lambda * trace + 2.0 * mu * eyy,
                    2.0 * mu * exy * 2.0, // Convert back to Voigt notation
                ]
            }
            3 => {
                let exx = strain[0];
                let eyy = strain[1];
                let ezz = strain[2];
                let exy = strain[3] / 2.0;
                let eyz = strain[4] / 2.0;
                let ezx = strain[5] / 2.0;
                let trace = exx + eyy + ezz;

                vec![
                    lambda * trace + 2.0 * mu * exx,
                    lambda * trace + 2.0 * mu * eyy,
                    lambda * trace + 2.0 * mu * ezz,
                    2.0 * mu * exy * 2.0,
                    2.0 * mu * eyz * 2.0,
                    2.0 * mu * ezx * 2.0,
                ]
            }
            _ => panic!("Unsupported spatial dimension: {}", dim),
        }
    }

    /// Acceleration field: ∂²u/∂t²
    ///
    /// # Mathematical Specification
    /// Returns a(x, t) = ∂²u/∂t² computed analytically
    fn acceleration(&self, x: &[f64], t: f64) -> Vec<f64>;

    /// Spatial dimension of the solution
    fn spatial_dimension(&self) -> usize;

    /// Number of displacement components (typically equals spatial dimension)
    fn displacement_components(&self) -> usize {
        self.spatial_dimension()
    }

    /// Name identifier for the solution (for reporting)
    fn name(&self) -> &str;

    /// Physical parameters (wavelength, frequency, amplitude, etc.)
    fn parameters(&self) -> SolutionParameters;
}

/// Physical parameters characterizing an analytical solution
#[derive(Debug, Clone, Copy)]
pub struct SolutionParameters {
    /// Amplitude (m)
    pub amplitude: f64,
    /// Wavelength (m)
    pub wavelength: f64,
    /// Angular frequency (rad/s)
    pub omega: f64,
    /// Wave speed (m/s)
    pub wave_speed: f64,
    /// Density (kg/m³)
    pub density: f64,
    /// First Lamé parameter (Pa)
    pub lambda: f64,
    /// Second Lamé parameter / Shear modulus (Pa)
    pub mu: f64,
}

impl SolutionParameters {
    /// Wave number: k = 2π/λ
    pub fn wave_number(&self) -> f64 {
        2.0 * std::f64::consts::PI / self.wavelength
    }

    /// Period: T = 2π/ω
    pub fn period(&self) -> f64 {
        2.0 * std::f64::consts::PI / self.omega
    }

    /// P-wave speed: cₚ = √((λ + 2μ)/ρ)
    pub fn p_wave_speed(&self) -> f64 {
        ((self.lambda + 2.0 * self.mu) / self.density).sqrt()
    }

    /// S-wave speed: cₛ = √(μ/ρ)
    pub fn s_wave_speed(&self) -> f64 {
        (self.mu / self.density).sqrt()
    }

    /// Poisson's ratio: ν = λ/(2(λ + μ))
    pub fn poisson_ratio(&self) -> f64 {
        self.lambda / (2.0 * (self.lambda + self.mu))
    }
}

// ============================================================================
// Validation Result Types
// ============================================================================

/// Result of a validation test with quantitative error metrics
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Test passed if all error metrics below tolerance
    pub passed: bool,
    /// Unique identifier for the test
    pub test_name: String,
    /// L² norm of error: ||computed - analytical||₂
    pub error_l2: f64,
    /// L∞ norm of error: max|computed - analytical|
    pub error_linf: f64,
    /// Acceptance tolerance
    pub tolerance: f64,
    /// Human-readable details
    pub details: String,
}

impl ValidationResult {
    /// Create successful validation result
    pub fn success(test_name: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            passed: true,
            test_name: test_name.into(),
            error_l2: 0.0,
            error_linf: 0.0,
            tolerance: 0.0,
            details: details.into(),
        }
    }

    /// Create failed validation result
    pub fn failure(
        test_name: impl Into<String>,
        error_l2: f64,
        error_linf: f64,
        tolerance: f64,
        details: impl Into<String>,
    ) -> Self {
        Self {
            passed: false,
            test_name: test_name.into(),
            error_l2,
            error_linf,
            tolerance,
            details: details.into(),
        }
    }

    /// Create validation result from error metrics
    pub fn from_errors(
        test_name: impl Into<String>,
        error_l2: f64,
        error_linf: f64,
        tolerance: f64,
    ) -> Self {
        let passed = error_l2 <= tolerance && error_linf <= tolerance;
        Self {
            passed,
            test_name: test_name.into(),
            error_l2,
            error_linf,
            tolerance,
            details: format!(
                "L² error: {:.3e}, L∞ error: {:.3e}, tolerance: {:.3e}",
                error_l2, error_linf, tolerance
            ),
        }
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.passed { "✓ PASS" } else { "✗ FAIL" };
        write!(f, "{} | {} | {}", status, self.test_name, self.details)
    }
}

/// Collection of validation results for a test suite
#[derive(Debug, Clone, Default)]
pub struct ValidationSuite {
    pub results: Vec<ValidationResult>,
}

impl ValidationSuite {
    /// Create new empty validation suite
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add validation result to suite
    pub fn add(&mut self, result: ValidationResult) {
        self.results.push(result);
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Count passed tests
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Count failed tests
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Get summary statistics
    pub fn summary(&self) -> ValidationSummary {
        ValidationSummary {
            total: self.results.len(),
            passed: self.passed_count(),
            failed: self.failed_count(),
            max_error_l2: self.results.iter().map(|r| r.error_l2).fold(0.0, f64::max),
            max_error_linf: self
                .results
                .iter()
                .map(|r| r.error_linf)
                .fold(0.0, f64::max),
        }
    }
}

impl fmt::Display for ValidationSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Validation Suite Results:")?;
        writeln!(f, "{}", "=".repeat(80))?;

        for result in &self.results {
            writeln!(f, "{}", result)?;
        }

        writeln!(f, "{}", "=".repeat(80))?;
        let summary = self.summary();
        writeln!(f, "{}", summary)?;

        Ok(())
    }
}

/// Summary statistics for validation suite
#[derive(Debug, Clone, Copy)]
pub struct ValidationSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub max_error_l2: f64,
    pub max_error_linf: f64,
}

impl fmt::Display for ValidationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Summary: {}/{} passed, {} failed",
            self.passed, self.total, self.failed
        )?;
        writeln!(f, "Max L² error: {:.3e}", self.max_error_l2)?;
        writeln!(f, "Max L∞ error: {:.3e}", self.max_error_linf)?;
        Ok(())
    }
}

// ============================================================================
// Wave Type Classification
// ============================================================================

/// Type of elastic wave
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveType {
    /// Longitudinal/compressional wave (P-wave)
    PWave,
    /// Transverse/shear wave (S-wave)
    SWave,
    /// Surface wave (Rayleigh, Love)
    SurfaceWave,
    /// Mixed mode (coupled P and S)
    Mixed,
}

impl fmt::Display for WaveType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaveType::PWave => write!(f, "P-wave"),
            WaveType::SWave => write!(f, "S-wave"),
            WaveType::SurfaceWave => write!(f, "Surface wave"),
            WaveType::Mixed => write!(f, "Mixed mode"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_success() {
        let result = ValidationResult::success("test", "All checks passed");
        assert!(result.passed);
        assert_eq!(result.error_l2, 0.0);
    }

    #[test]
    fn test_validation_result_from_errors() {
        let result = ValidationResult::from_errors("test", 1e-4, 2e-4, 1e-3);
        assert!(result.passed);

        let result_fail = ValidationResult::from_errors("test", 1e-2, 2e-2, 1e-3);
        assert!(!result_fail.passed);
    }

    #[test]
    fn test_validation_suite() {
        let mut suite = ValidationSuite::new();
        suite.add(ValidationResult::success("test1", "ok"));
        suite.add(ValidationResult::failure(
            "test2",
            1e-2,
            2e-2,
            1e-3,
            "too large",
        ));

        assert!(!suite.all_passed());
        assert_eq!(suite.passed_count(), 1);
        assert_eq!(suite.failed_count(), 1);
    }

    #[test]
    fn test_solution_parameters() {
        let params = SolutionParameters {
            amplitude: 1e-6,
            wavelength: 0.01,
            omega: 1e6,
            wave_speed: 5000.0,
            density: 2700.0,
            lambda: 5e10,
            mu: 2.6e10,
        };

        assert!(params.amplitude > 0.0);
        let k = params.wave_number();
        assert!((k - 2.0 * std::f64::consts::PI / 0.01).abs() < 1e-10);

        let cp = params.p_wave_speed();
        let cs = params.s_wave_speed();
        assert!(cp > cs, "P-wave speed must exceed S-wave speed");
    }

    #[test]
    fn test_analytical_solution_metadata() {
        let params = SolutionParameters {
            amplitude: 1e-6,
            wavelength: 0.01,
            omega: 0.0,
            wave_speed: 5000.0,
            density: 2700.0,
            lambda: 5e10,
            mu: 2.6e10,
        };

        let wave = analytical_solutions::PlaneWave2D::p_wave(1e-6, 0.01, [1.0, 0.0], params);
        assert_eq!(wave.displacement_components(), 2);
        assert!(wave.name().contains("P-wave"));

        let wave_params = wave.parameters();
        assert!((wave_params.amplitude - 1e-6).abs() < 1e-12);
    }
}
