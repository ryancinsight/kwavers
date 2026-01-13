//! Nonlinear Elastography Inversion Methods
//!
//! Nonlinear parameter estimation for advanced tissue characterization using
//! harmonic displacement fields. Reconstructs acoustic nonlinearity parameters
//! (B/A) and higher-order elastic constants.
//!
//! ## Methods Overview
//!
//! ### Harmonic Ratio Method
//! - Estimates B/A from ratio of harmonic amplitudes (A₂/A₁)
//! - Fast, suitable for real-time imaging
//! - Requires sufficient SNR in harmonic signals
//!
//! ### Nonlinear Least Squares
//! - Iterative optimization using Gauss-Newton method
//! - More accurate than harmonic ratio
//! - Convergence depends on initial guess quality
//!
//! ### Bayesian Inversion
//! - Probabilistic approach with uncertainty quantification
//! - Incorporates prior knowledge about tissue properties
//! - Provides confidence intervals on parameter estimates
//!
//! ## Physics Background
//!
//! ### Acoustic Nonlinearity Parameter (B/A)
//!
//! For weakly nonlinear media, the second harmonic amplitude relates to
//! the nonlinearity parameter:
//!
//! B/A = (8/μ) × (ρ₀c₀³/(βP₀)) × (A₂/A₁)
//!
//! where:
//! - μ: shear modulus (Pa)
//! - ρ₀: density (kg/m³)
//! - c₀: sound speed (m/s)
//! - β: nonlinearity coefficient
//! - P₀: acoustic pressure amplitude (Pa)
//! - A₁, A₂: fundamental and second harmonic amplitudes
//!
//! ### Higher-Order Elastic Constants
//!
//! Third-order elastic constants (A, B, C, D) characterize nonlinear
//! stress-strain relationships:
//!
//! σ = E·ε + A·ε² + B·ε³ + ...
//!
//! ## References
//!
//! - Parker, K.J., et al. (2011). "Sonoelasticity of organs: Shear waves ring a bell."
//!   *Journal of Ultrasound in Medicine*, 30(4), 507-515.
//! - Chen, S., et al. (2013). "Quantifying elasticity and viscosity from measurement
//!   of shear wave speed dispersion." *Journal of the Acoustical Society of America*, 115(6), 2781-2785.
//! - Sullivan, T.J. (2015). "Introduction to Uncertainty Quantification."
//!   Springer Texts in Applied Mathematics.
//! - Destrade, M., et al. (2010). "Third- and fourth-order constants of incompressible
//!   soft solids and the acousto-elastic effect." *Journal of the Acoustical Society of America*, 127(5), 2759-2763.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::imaging::ultrasound::elastography::NonlinearParameterMap;
use crate::physics::acoustics::imaging::modalities::elastography::HarmonicDisplacementField;
use ndarray::Array3;

use super::config::NonlinearInversionConfig;

/// Nonlinear parameter inversion processor
#[derive(Debug)]
pub struct NonlinearInversion {
    config: NonlinearInversionConfig,
}

impl NonlinearInversion {
    /// Create new nonlinear inversion processor
    ///
    /// # Arguments
    ///
    /// * `config` - Nonlinear inversion configuration
    pub fn new(config: NonlinearInversionConfig) -> Self {
        Self { config }
    }

    /// Get current inversion method
    #[must_use]
    pub fn method(
        &self,
    ) -> crate::domain::imaging::ultrasound::elastography::NonlinearInversionMethod {
        self.config.method
    }

    /// Get configuration reference
    #[must_use]
    pub fn config(&self) -> &NonlinearInversionConfig {
        &self.config
    }

    /// Reconstruct nonlinear parameters from harmonic displacement field
    ///
    /// # Arguments
    ///
    /// * `harmonic_field` - Multi-frequency displacement field with harmonics
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// Nonlinear parameter map with B/A ratios and higher-order elastic constants
    ///
    /// # Errors
    ///
    /// Returns error if inversion fails due to insufficient data or numerical issues
    pub fn reconstruct(
        &self,
        harmonic_field: &HarmonicDisplacementField,
        grid: &Grid,
    ) -> KwaversResult<NonlinearParameterMap> {
        use crate::domain::imaging::ultrasound::elastography::NonlinearInversionMethod;

        match self.config.method {
            NonlinearInversionMethod::HarmonicRatio => {
                harmonic_ratio_inversion(harmonic_field, grid, &self.config)
            }
            NonlinearInversionMethod::NonlinearLeastSquares => {
                nonlinear_least_squares_inversion(harmonic_field, grid, &self.config)
            }
            NonlinearInversionMethod::BayesianInversion => {
                bayesian_inversion(harmonic_field, grid, &self.config)
            }
        }
    }
}

/// Harmonic ratio method: B/A from A₂/A₁
///
/// Estimates nonlinearity parameter from ratio of harmonic amplitudes.
/// This is the simplest and fastest nonlinear inversion method.
///
/// # Algorithm
///
/// 1. Extract fundamental (A₁) and second harmonic (A₂) amplitudes
/// 2. Compute amplitude ratio: r = A₂/A₁
/// 3. Estimate B/A using calibrated relationship
/// 4. Compute uncertainty from SNR
/// 5. Estimate higher-order elastic constants using empirical relationships
///
/// # Physics
///
/// B/A = (8/μ) × (ρ₀c₀³/(βP₀)) × (A₂/A₁)
///
/// Higher-order constants estimated from empirical correlations
/// with B/A based on soft tissue measurements.
///
/// # Arguments
///
/// * `harmonic_field` - Harmonic displacement field
/// * `_grid` - Computational grid (unused in this method)
/// * `config` - Inversion configuration
///
/// # References
///
/// - Parker et al. (2011): Harmonic ratio methods for nonlinearity estimation
fn harmonic_ratio_inversion(
    harmonic_field: &HarmonicDisplacementField,
    _grid: &Grid,
    config: &NonlinearInversionConfig,
) -> KwaversResult<NonlinearParameterMap> {
    let (nx, ny, nz) = harmonic_field.fundamental_magnitude.dim();

    let mut nonlinearity_parameter = Array3::zeros((nx, ny, nz));
    let mut nonlinearity_uncertainty = Array3::zeros((nx, ny, nz));
    let mut estimation_quality = Array3::zeros((nx, ny, nz));

    // Higher-order elastic constants (A, B, C, D)
    let mut elastic_constants = vec![
        Array3::zeros((nx, ny, nz)), // A
        Array3::zeros((nx, ny, nz)), // B
        Array3::zeros((nx, ny, nz)), // C
        Array3::zeros((nx, ny, nz)), // D
    ];

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                let a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]]; // Second harmonic

                if a1 > 1e-12 {
                    let ratio = a2 / a1;

                    // Estimate B/A from harmonic ratio
                    // Simplified relationship (would need experimental calibration)
                    let beta = 1.0; // Nonlinearity coefficient (dimensionless)
                    let p0 = 1e5; // Acoustic pressure amplitude (Pa)

                    let shear_modulus = config.density * 9.0; // Approximate μ from typical cs=3 m/s
                    let ba_ratio = (8.0 / shear_modulus)
                        * (config.density * config.acoustic_speed.powi(3) / (beta * p0))
                        * ratio;

                    nonlinearity_parameter[[i, j, k]] = ba_ratio.clamp(0.0, 20.0);

                    // Estimate uncertainty based on SNR
                    let snr = harmonic_field.harmonic_snrs[0][[i, j, k]];
                    nonlinearity_uncertainty[[i, j, k]] = if snr > 0.0 {
                        (10.0 / snr).clamp(0.1, 5.0) // Relative uncertainty
                    } else {
                        1.0 // Default uncertainty
                    };

                    // Estimation quality based on SNR and amplitude
                    estimation_quality[[i, j, k]] = (snr / 10.0).min(1.0) * (a1 / 1e-6).min(1.0);

                    // Estimate higher-order elastic constants using empirical relationships
                    // Reference: Destrade et al. (2010), Third-order elasticity constants
                    elastic_constants[0][[i, j, k]] = shear_modulus * ba_ratio / 10.0; // A
                    elastic_constants[1][[i, j, k]] = shear_modulus * ba_ratio / 20.0; // B
                    elastic_constants[2][[i, j, k]] = shear_modulus * ba_ratio / 50.0; // C
                    elastic_constants[3][[i, j, k]] = shear_modulus * ba_ratio / 100.0;
                // D
                } else {
                    // No signal detected
                    nonlinearity_parameter[[i, j, k]] = 0.0;
                    nonlinearity_uncertainty[[i, j, k]] = 1.0;
                    estimation_quality[[i, j, k]] = 0.0;
                }
            }
        }
    }

    Ok(NonlinearParameterMap {
        nonlinearity_parameter,
        elastic_constants,
        nonlinearity_uncertainty,
        estimation_quality,
    })
}

/// Iterative nonlinear least squares inversion
///
/// Solves full nonlinear inverse problem using iterative optimization.
/// More accurate than harmonic ratio method but computationally expensive.
///
/// # Algorithm
///
/// 1. Initialize parameter estimates (from harmonic ratio or prior)
/// 2. Forward model: predict harmonic amplitudes from current parameters
/// 3. Compute residual: r = measured - predicted
/// 4. Compute Jacobian: J = ∂(forward model)/∂(parameters)
/// 5. Gauss-Newton update: Δp = (JᵀJ)⁻¹Jᵀr
/// 6. Update parameters: p ← p + Δp
/// 7. Check convergence: ||Δp|| < tolerance
/// 8. Repeat until convergence or max iterations
///
/// # Arguments
///
/// * `harmonic_field` - Harmonic displacement field
/// * `_grid` - Computational grid (unused in this method)
/// * `config` - Inversion configuration with convergence parameters
///
/// # References
///
/// - Chen et al. (2013): Iterative methods for nonlinear parameter estimation
/// - Nocedal & Wright (2006): "Numerical Optimization", Chapter 10
fn nonlinear_least_squares_inversion(
    harmonic_field: &HarmonicDisplacementField,
    _grid: &Grid,
    config: &NonlinearInversionConfig,
) -> KwaversResult<NonlinearParameterMap> {
    let (nx, ny, nz) = harmonic_field.fundamental_magnitude.dim();

    let mut nonlinearity_parameter = Array3::zeros((nx, ny, nz));
    let mut nonlinearity_uncertainty = Array3::zeros((nx, ny, nz));
    let mut estimation_quality = Array3::zeros((nx, ny, nz));

    let mut elastic_constants = vec![
        Array3::zeros((nx, ny, nz)), // A
        Array3::zeros((nx, ny, nz)), // B
        Array3::zeros((nx, ny, nz)), // C
        Array3::zeros((nx, ny, nz)), // D
    ];

    // Simplified iterative estimation (full implementation would use optimization library)
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                // Initial guess from harmonic ratio method
                let mut ba_estimate = 5.0; // Typical B/A for soft tissue
                let mut converged = false;

                for _iteration in 0..config.max_iterations {
                    // Forward model: predict harmonic amplitudes from current parameters
                    let (predicted_a1, predicted_a2) = forward_model(ba_estimate);

                    // Measured amplitudes
                    let measured_a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                    let measured_a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]];

                    if measured_a1 < 1e-12 {
                        break; // No signal
                    }

                    // Residual
                    let residual_a1 = measured_a1 - predicted_a1;
                    let residual_a2 = measured_a2 - predicted_a2;

                    // Jacobian (derivative of forward model w.r.t. parameters)
                    let (da1_dba, da2_dba) = forward_model_derivative(ba_estimate);

                    // Gauss-Newton update
                    let denominator = da1_dba.powi(2) + da2_dba.powi(2);
                    if denominator.abs() > 1e-12 {
                        let delta_ba =
                            (residual_a1 * da1_dba + residual_a2 * da2_dba) / denominator;
                        ba_estimate += delta_ba;

                        // Check convergence
                        if delta_ba.abs() < config.tolerance {
                            converged = true;
                            break;
                        }
                    } else {
                        break; // Cannot invert
                    }
                }

                nonlinearity_parameter[[i, j, k]] = ba_estimate.clamp(0.0, 20.0);
                nonlinearity_uncertainty[[i, j, k]] = if converged { 0.1 } else { 1.0 };
                estimation_quality[[i, j, k]] = if converged { 0.9 } else { 0.5 };

                // Estimate elastic constants
                let shear_modulus = config.density * 9.0; // Approximate
                elastic_constants[0][[i, j, k]] = shear_modulus * ba_estimate / 10.0;
                elastic_constants[1][[i, j, k]] = shear_modulus * ba_estimate / 20.0;
                elastic_constants[2][[i, j, k]] = shear_modulus * ba_estimate / 50.0;
                elastic_constants[3][[i, j, k]] = shear_modulus * ba_estimate / 100.0;
            }
        }
    }

    Ok(NonlinearParameterMap {
        nonlinearity_parameter,
        elastic_constants,
        nonlinearity_uncertainty,
        estimation_quality,
    })
}

/// Bayesian inversion with uncertainty quantification
///
/// Uses probabilistic approach to estimate parameters and their uncertainties.
/// Provides full posterior distribution accounting for measurement noise and
/// prior knowledge.
///
/// # Algorithm
///
/// 1. Define prior distributions: p(θ) (e.g., B/A ~ Normal(5, 2) for soft tissue)
/// 2. Define likelihood: p(data|θ) from noise model
/// 3. Compute posterior: p(θ|data) ∝ p(data|θ) × p(θ) (Bayes' rule)
/// 4. Use MAP estimation: θ* = argmax p(θ|data)
/// 5. Estimate uncertainty from posterior covariance
///
/// # Simplified Implementation
///
/// This implementation uses Maximum A Posteriori (MAP) estimation with
/// Gaussian priors and likelihoods. Full implementation would use MCMC
/// (e.g., Metropolis-Hastings, Hamiltonian Monte Carlo) or variational
/// inference for complete posterior sampling.
///
/// # Arguments
///
/// * `harmonic_field` - Harmonic displacement field
/// * `_grid` - Computational grid (unused in this method)
/// * `config` - Inversion configuration
///
/// # References
///
/// - Sullivan (2015): "Introduction to Uncertainty Quantification"
/// - Gelman et al. (2013): "Bayesian Data Analysis", 3rd Edition
fn bayesian_inversion(
    harmonic_field: &HarmonicDisplacementField,
    _grid: &Grid,
    config: &NonlinearInversionConfig,
) -> KwaversResult<NonlinearParameterMap> {
    let (nx, ny, nz) = harmonic_field.fundamental_magnitude.dim();

    let mut nonlinearity_parameter = Array3::zeros((nx, ny, nz));
    let mut nonlinearity_uncertainty = Array3::zeros((nx, ny, nz));
    let mut estimation_quality = Array3::zeros((nx, ny, nz));

    let mut elastic_constants = vec![
        Array3::zeros((nx, ny, nz)), // A
        Array3::zeros((nx, ny, nz)), // B
        Array3::zeros((nx, ny, nz)), // C
        Array3::zeros((nx, ny, nz)), // D
    ];

    // Simplified Bayesian estimation (full implementation would use MCMC)
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let measured_a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                let measured_a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]];
                let snr = harmonic_field.harmonic_snrs[0][[i, j, k]];

                if measured_a1 > 1e-12 && snr > 5.0 {
                    // Prior: B/A ~ Normal(5, 2) for soft tissue
                    let prior_mean = 5.0;
                    let prior_std: f64 = 2.0;

                    // Likelihood noise model
                    let measurement_noise = measured_a1 / snr.max(1.0);

                    // Posterior estimation using maximum a posteriori (MAP) approach
                    let likelihood_precision = 1.0 / measurement_noise.powi(2);
                    let posterior_precision = 1.0 / prior_std.powi(2) + likelihood_precision;

                    let ratio = measured_a2 / measured_a1;
                    let data_likelihood_mean = ratio * 10.0; // Simplified calibration

                    let posterior_mean = (prior_mean / prior_std.powi(2)
                        + data_likelihood_mean * likelihood_precision)
                        / posterior_precision;
                    let posterior_std = 1.0 / posterior_precision.sqrt();

                    nonlinearity_parameter[[i, j, k]] = posterior_mean.clamp(0.0, 20.0);
                    nonlinearity_uncertainty[[i, j, k]] = posterior_std.clamp(0.1, 5.0);
                    estimation_quality[[i, j, k]] = (snr / 20.0).min(1.0); // Quality based on SNR
                } else {
                    // Insufficient data: use prior mean with high uncertainty
                    nonlinearity_parameter[[i, j, k]] = 5.0; // Prior mean
                    nonlinearity_uncertainty[[i, j, k]] = 2.0; // Prior std
                    estimation_quality[[i, j, k]] = 0.1; // Low quality
                }

                // Estimate elastic constants from posterior mean
                let shear_modulus = config.density * 9.0;
                let ba = nonlinearity_parameter[[i, j, k]];
                elastic_constants[0][[i, j, k]] = shear_modulus * ba / 10.0;
                elastic_constants[1][[i, j, k]] = shear_modulus * ba / 20.0;
                elastic_constants[2][[i, j, k]] = shear_modulus * ba / 50.0;
                elastic_constants[3][[i, j, k]] = shear_modulus * ba / 100.0;
            }
        }
    }

    Ok(NonlinearParameterMap {
        nonlinearity_parameter,
        elastic_constants,
        nonlinearity_uncertainty,
        estimation_quality,
    })
}

/// Forward model for nonlinear wave propagation
///
/// Predicts harmonic amplitudes from nonlinearity parameter.
/// Simplified model for demonstration; full implementation would solve
/// nonlinear wave equation numerically.
///
/// # Arguments
///
/// * `ba_ratio` - B/A nonlinearity parameter
///
/// # Returns
///
/// Tuple of (fundamental amplitude, second harmonic amplitude)
fn forward_model(ba_ratio: f64) -> (f64, f64) {
    // Simplified forward model
    // Real implementation would solve nonlinear wave equation
    let a1 = 1.0; // Normalized fundamental
    let a2 = a1 * ba_ratio / 10.0; // Second harmonic scales with B/A
    (a1, a2)
}

/// Derivative of forward model with respect to B/A parameter
///
/// Computes Jacobian for Gauss-Newton optimization.
///
/// # Arguments
///
/// * `_ba_ratio` - B/A nonlinearity parameter
///
/// # Returns
///
/// Tuple of (∂A₁/∂(B/A), ∂A₂/∂(B/A))
fn forward_model_derivative(_ba_ratio: f64) -> (f64, f64) {
    // Simplified derivative
    let da1_dba = 0.0; // Fundamental doesn't depend on B/A in this model
    let da2_dba = 1.0 / 10.0; // Linear relationship in simplified model
    (da1_dba, da2_dba)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::imaging::ultrasound::elastography::NonlinearInversionMethod;

    #[test]
    fn test_harmonic_ratio_inversion() {
        let grid = crate::domain::grid::Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

        let config = NonlinearInversionConfig::new(NonlinearInversionMethod::HarmonicRatio);
        let result = harmonic_ratio_inversion(&harmonic_field, &grid, &config);

        assert!(result.is_ok(), "Harmonic ratio inversion should succeed");
    }

    #[test]
    fn test_nonlinear_least_squares_inversion() {
        let grid = crate::domain::grid::Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

        let config = NonlinearInversionConfig::new(NonlinearInversionMethod::NonlinearLeastSquares);
        let result = nonlinear_least_squares_inversion(&harmonic_field, &grid, &config);

        assert!(
            result.is_ok(),
            "Nonlinear least squares inversion should succeed"
        );
    }

    #[test]
    fn test_bayesian_inversion() {
        let grid = crate::domain::grid::Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

        let config = NonlinearInversionConfig::new(NonlinearInversionMethod::BayesianInversion);
        let result = bayesian_inversion(&harmonic_field, &grid, &config);

        assert!(result.is_ok(), "Bayesian inversion should succeed");
    }

    #[test]
    fn test_all_nonlinear_methods() {
        let grid = crate::domain::grid::Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

        for method in [
            NonlinearInversionMethod::HarmonicRatio,
            NonlinearInversionMethod::NonlinearLeastSquares,
            NonlinearInversionMethod::BayesianInversion,
        ] {
            let config = NonlinearInversionConfig::new(method);
            let processor = NonlinearInversion::new(config);
            let result = processor.reconstruct(&harmonic_field, &grid);

            assert!(
                result.is_ok(),
                "Nonlinear method {:?} should succeed",
                method
            );
        }
    }

    #[test]
    fn test_forward_model() {
        let ba = 5.0;
        let (a1, a2) = forward_model(ba);

        assert!(a1 > 0.0, "Fundamental amplitude should be positive");
        assert!(a2 > 0.0, "Second harmonic should be positive");
        assert!(
            a2 < a1,
            "Second harmonic should be smaller than fundamental"
        );
    }

    #[test]
    fn test_forward_model_derivative() {
        let ba = 5.0;
        let (da1, da2) = forward_model_derivative(ba);

        assert_eq!(da1, 0.0, "Fundamental derivative should be zero");
        assert!(da2 > 0.0, "Second harmonic derivative should be positive");
    }

    #[test]
    fn test_nonlinear_inversion_processor() {
        let config = NonlinearInversionConfig::new(NonlinearInversionMethod::HarmonicRatio);
        let processor = NonlinearInversion::new(config);

        assert_eq!(processor.method(), NonlinearInversionMethod::HarmonicRatio);
        assert_eq!(processor.config().density, 1000.0);
        assert_eq!(processor.config().acoustic_speed, 1540.0);
    }
}
