//! Nonlinear Elastography Inversion Methods
//!
//! Nonlinear parameter estimation for advanced tissue characterization using
//! harmonic shear-wave displacement fields. Reconstructs the shear-wave
//! acoustic nonlinearity parameter β_s and the acoustic pressure-wave
//! nonlinearity parameter B/A.
//!
//! ## Methods Overview
//!
//! ### Harmonic Ratio Method
//! - Estimates β_s and B/A from ratio of second-harmonic to fundamental shear
//!   displacement amplitudes.
//! - Fast, suitable for real-time imaging.
//! - Requires SNR > ~10 dB in the second-harmonic displacement.
//!
//! ### Nonlinear Least Squares
//! - Iterative Gauss-Newton optimisation fitting measured (A₁, A₂) to the
//!   forward shear-wave harmonic model.
//! - More robust to noise than the direct ratio.
//! - Convergence depends on initial-guess quality (seeded with harmonic ratio
//!   result).
//!
//! ### Bayesian Inversion
//! - Maximum A Posteriori (MAP) estimation with Gaussian prior
//!   B/A ~ N(7, 2²) for soft tissue.
//! - Provides posterior standard deviation as uncertainty quantification.
//! - Full MCMC posterior sampling reserved for future work.
//!
//! ---
//!
//! ## Physics Background
//!
//! ### Shear-Wave Second-Harmonic Generation
//!
//! For a shear wave of angular frequency ω propagating along z in a
//! nonlinear elastic medium with shear modulus μ, the second-harmonic
//! displacement amplitude A₂ accumulated over propagation distance z is
//! (Rénier et al. 2008, JASA 124(5) 2856, Eq. 7):
//!
//! ```text
//! A₂(z) = β_s k_s A₁² z / 2
//! ```
//!
//! where:
//! - A₁   : fundamental shear displacement amplitude [m]
//! - A₂   : second-harmonic shear displacement amplitude [m]
//! - k_s  = ω / c_s : shear wavenumber [rad m⁻¹]
//! - c_s  : shear wave speed [m s⁻¹]
//! - β_s  : shear-wave acoustic nonlinearity parameter (dimensionless)
//! - z    : propagation (accumulation) distance [m]
//!
//! Solving for β_s:
//!
//! ```text
//! β_s = 2 A₂ c_s / (ω A₁² z)      [Rénier 2008, Eq. 8]
//! ```
//!
//! ### Relation to Landau-Lifshitz Third-Order Constant
//!
//! The Landau-Lifshitz third-order elastic constant A_L relates to β_s via
//! (Destrade & Ogden 2010, JASA 127(5) 2759, Eq. 3.8):
//!
//! ```text
//! A_L = μ (4 β_s − 3)              [Pa]
//! ```
//!
//! ### Reported B/A
//!
//! Following the acoustic convention B/A = 2(β − 1), the shear-wave analogue is:
//!
//! ```text
//! B/A ≡ 2(β_s − 1)                [dimensionless, ~0–20 for soft tissue]
//! ```
//!
//! For reference tissue values (Rénier 2008, Table 1):
//! - Gelatin phantom (2%): β_s ≈ 1.8, B/A ≈ 1.6
//! - Pork muscle:          β_s ≈ 3.5, B/A ≈ 5.0
//! - Liver (ex vivo):      β_s ≈ 4.0, B/A ≈ 6.0
//!
//! ### Higher-Order Elastic Constants
//!
//! From A_L, the Murnaghan third-order constants (A, B, C, D notation of
//! Destrade & Ogden 2010) are approximated for nearly incompressible isotropic
//! media:
//! - A  = A_L
//! - B  ≈ A_L / 3   (empirical ratio for soft tissue; Destrade 2010 §4)
//! - C  ≈ A_L / 9
//! - D  ≈ A_L / 27
//!
//! ---
//!
//! ## References
//!
//! - Rénier M, Gennisson J-L, Barrière C, Royer D, Fink M (2008).
//!   "Fourth-order shear elastic constant assessment in quasi-incompressible
//!   soft solids." *JASA* **124**(5), 2856–2864. doi:10.1121/1.2982433
//! - Destrade M, Ogden R W (2010). "On the third- and fourth-order constants
//!   of incompressible isotropic elasticity." *JASA* **128**(6), 3334–3343.
//!   doi:10.1121/1.3505102
//! - Parker K J et al. (2011). "Sonoelasticity of organs: Shear waves ring a
//!   bell." *J Ultrasound Med* **30**(4), 507–515.
//! - Nightingale K R et al. (2011). "Derivation and analysis of viscoelastic
//!   properties in human liver." *IEEE TUFFC* **62**(11), 1946–1957.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::imaging::ultrasound::elastography::NonlinearParameterMap;
use crate::physics::acoustics::imaging::modalities::elastography::HarmonicDisplacementField;
use ndarray::Array3;
use std::f64::consts::PI;

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

// ─── shared pre-computations ──────────────────────────────────────────────────

/// Derive the shear modulus [Pa] from config density and shear wave speed.
///
/// μ = ρ c_s²  (Hooke's law for shear waves in elastic media)
#[inline]
fn shear_modulus(config: &NonlinearInversionConfig) -> f64 {
    let c_s = config.shear_wave_speed.max(1e-3); // guard against zero
    config.density * c_s * c_s
}

/// Compute β_s from measured displacement amplitudes using Rénier (2008) Eq. 8.
///
/// ```text
/// β_s = 2 A₂ c_s / (ω A₁² z)
/// ```
///
/// Returns `None` if A₁ is below the noise floor (< 1e-12 m).
#[inline]
fn beta_s_from_amplitudes(a1: f64, a2: f64, config: &NonlinearInversionConfig) -> Option<f64> {
    if a1 < 1e-12 {
        return None;
    }
    let omega = 2.0 * PI * config.excitation_frequency;
    let c_s = config.shear_wave_speed.max(1e-3);
    let z = config.propagation_distance.max(1e-6);
    // β_s = 2 A₂ c_s / (ω A₁² z)  [Rénier 2008, Eq. 8]
    Some(2.0 * a2 * c_s / (omega * a1 * a1 * z))
}

/// Convert β_s to the reported B/A value.
///
/// ```text
/// B/A = 2(β_s − 1)   [acoustic convention; linear medium → B/A = 0]
/// ```
#[inline]
fn ba_from_beta_s(beta_s: f64) -> f64 {
    2.0 * (beta_s - 1.0)
}

/// Compute the Landau-Lifshitz constant A_L [Pa] from μ and β_s.
///
/// ```text
/// A_L = μ (4 β_s − 3)   [Destrade & Ogden 2010, Eq. 3.8]
/// ```
#[inline]
fn a_landau(mu: f64, beta_s: f64) -> f64 {
    mu * (4.0 * beta_s - 3.0)
}

// ─── Harmonic ratio method ────────────────────────────────────────────────────

/// Harmonic ratio method: B/A from A₂/A₁
///
/// Estimates the shear-wave nonlinearity parameter β_s directly from the
/// amplitude ratio of second-harmonic to fundamental shear displacement
/// (Rénier et al. 2008), then converts to the reported B/A via B/A = 2(β_s−1).
///
/// # Algorithm
///
/// 1. For each voxel, read A₁ (fundamental) and A₂ (second-harmonic) displacement
///    amplitudes from `harmonic_field`.
/// 2. If A₁ ≥ 1e-12 m, compute β_s = 2 A₂ c_s / (ω A₁² z).
/// 3. Clamp B/A = 2(β_s−1) to [0, 20] (physiological range).
/// 4. Compute Landau constant A_L = μ(4β_s − 3) and derive A, B, C, D.
/// 5. SNR-based estimation uncertainty: σ ≈ (10/SNR) clamped to [0.1, 5.0].
///
/// # References
///
/// - Rénier et al. (2008), JASA 124(5) 2856, Eq. 7–8.
fn harmonic_ratio_inversion(
    harmonic_field: &HarmonicDisplacementField,
    _grid: &Grid,
    config: &NonlinearInversionConfig,
) -> KwaversResult<NonlinearParameterMap> {
    let (nx, ny, nz) = harmonic_field.fundamental_magnitude.dim();

    let mut nonlinearity_parameter = Array3::zeros((nx, ny, nz));
    let mut nonlinearity_uncertainty = Array3::zeros((nx, ny, nz));
    let mut estimation_quality = Array3::zeros((nx, ny, nz));

    // Higher-order elastic constants (A_L, B ≈ A_L/3, C ≈ A_L/9, D ≈ A_L/27)
    let mut elastic_constants = vec![
        Array3::zeros((nx, ny, nz)), // A = A_L
        Array3::zeros((nx, ny, nz)), // B ≈ A_L / 3
        Array3::zeros((nx, ny, nz)), // C ≈ A_L / 9
        Array3::zeros((nx, ny, nz)), // D ≈ A_L / 27
    ];

    let mu = shear_modulus(config); // μ = ρ c_s²  [Pa]

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                let a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]];

                if let Some(beta) = beta_s_from_amplitudes(a1, a2, config) {
                    let ba_ratio = ba_from_beta_s(beta).clamp(0.0, 20.0);
                    let al = a_landau(mu, beta);

                    nonlinearity_parameter[[i, j, k]] = ba_ratio;

                    // Estimation quality and uncertainty from harmonic SNR
                    let snr = harmonic_field.harmonic_snrs[0][[i, j, k]];
                    nonlinearity_uncertainty[[i, j, k]] = (10.0 / snr.max(1e-3)).clamp(0.1, 5.0);
                    estimation_quality[[i, j, k]] = (snr / 10.0).min(1.0) * (a1 / 1e-6).min(1.0);

                    // Third-order elastic constants (Destrade & Ogden 2010 §4)
                    elastic_constants[0][[i, j, k]] = al; // A
                    elastic_constants[1][[i, j, k]] = al / 3.0; // B
                    elastic_constants[2][[i, j, k]] = al / 9.0; // C
                    elastic_constants[3][[i, j, k]] = al / 27.0; // D
                } else {
                    // No signal detected: report zero with maximum uncertainty
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

// ─── Nonlinear least squares ──────────────────────────────────────────────────

/// Iterative nonlinear least squares inversion (Gauss-Newton)
///
/// Fits the observed (A₁, A₂) pair at each voxel to the shear-wave forward
/// model by iteratively minimising ‖(A₁_pred − A₁_obs, A₂_pred − A₂_obs)‖².
///
/// # Algorithm
///
/// 1. Initialise B/A estimate from the direct harmonic ratio (§harmonic_ratio_inversion).
/// 2. Forward model: predict (A₁_pred, A₂_pred) from current B/A.
/// 3. Residual: r = (A₁_obs − A₁_pred, A₂_obs − A₂_pred).
/// 4. Jacobian column J = (∂A₁_pred/∂(B/A), ∂A₂_pred/∂(B/A)).
/// 5. Gauss-Newton update: Δ(B/A) = (Jᵀ r) / (Jᵀ J).
/// 6. Repeat until |Δ(B/A)| < tolerance or max_iterations reached.
///
/// # References
///
/// - Nocedal & Wright (2006). *Numerical Optimization*, §10.3 (Gauss-Newton).
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

    let mu = shear_modulus(config);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let measured_a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                let measured_a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]];

                if measured_a1 < 1e-12 {
                    nonlinearity_parameter[[i, j, k]] = 0.0;
                    nonlinearity_uncertainty[[i, j, k]] = 1.0;
                    estimation_quality[[i, j, k]] = 0.0;
                    continue;
                }

                // Seed from harmonic ratio (warm start for Gauss-Newton)
                let mut ba_estimate = beta_s_from_amplitudes(measured_a1, measured_a2, config)
                    .map(ba_from_beta_s)
                    .unwrap_or(5.0) // default: typical soft tissue
                    .clamp(0.0, 20.0);

                let mut converged = false;
                for _iteration in 0..config.max_iterations {
                    // Forward model: predict (A₁_pred, A₂_pred) from current B/A
                    let (predicted_a1, predicted_a2) =
                        forward_model(ba_estimate, measured_a1, config);

                    // Residuals
                    let residual_a1 = measured_a1 - predicted_a1;
                    let residual_a2 = measured_a2 - predicted_a2;

                    // Jacobian: ∂(A₁, A₂)/∂(B/A)
                    let (da1_dba, da2_dba) =
                        forward_model_derivative(ba_estimate, measured_a1, config);

                    // Gauss-Newton update: Δ = Jᵀr / JᵀJ
                    let jt_j = da1_dba * da1_dba + da2_dba * da2_dba;
                    if jt_j.abs() < f64::EPSILON {
                        break;
                    }
                    let delta_ba = (residual_a1 * da1_dba + residual_a2 * da2_dba) / jt_j;
                    ba_estimate = (ba_estimate + delta_ba).clamp(0.0, 20.0);

                    if delta_ba.abs() < config.tolerance {
                        converged = true;
                        break;
                    }
                }

                nonlinearity_parameter[[i, j, k]] = ba_estimate;
                nonlinearity_uncertainty[[i, j, k]] = if converged { 0.1 } else { 1.0 };
                estimation_quality[[i, j, k]] = if converged { 0.9 } else { 0.5 };

                // Elastic constants from converged estimate
                let beta = ba_estimate / 2.0 + 1.0; // β_s = B/A/2 + 1
                let al = a_landau(mu, beta);
                elastic_constants[0][[i, j, k]] = al;
                elastic_constants[1][[i, j, k]] = al / 3.0;
                elastic_constants[2][[i, j, k]] = al / 9.0;
                elastic_constants[3][[i, j, k]] = al / 27.0;
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

// ─── Bayesian inversion ───────────────────────────────────────────────────────

/// Bayesian inversion with uncertainty quantification (MAP estimate)
///
/// Uses a Gaussian prior B/A ~ N(μ_prior, σ_prior²) over soft-tissue values
/// and a Gaussian likelihood centred on the harmonic-ratio B/A observation
/// with precision derived from the measured SNR.
///
/// MAP update rule for conjugate Gaussian model:
///
/// ```text
/// precision_post = 1/σ_prior² + precision_likelihood
/// mean_post      = (μ_prior/σ_prior² + ba_obs × precision_likelihood) / precision_post
/// σ_post         = 1 / √precision_post
/// ```
///
/// # References
///
/// - Gelman A et al. (2013). *Bayesian Data Analysis*, 3rd ed. §2.4 (conjugate Gaussian).
/// - Sullivan T J (2015). *Introduction to Uncertainty Quantification.* §3.1.
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
        Array3::zeros((nx, ny, nz)),
        Array3::zeros((nx, ny, nz)),
        Array3::zeros((nx, ny, nz)),
        Array3::zeros((nx, ny, nz)),
    ];

    let mu = shear_modulus(config);

    // Gaussian prior: B/A ~ N(7, 2²) (centred on liver/soft tissue; Rénier 2008 Table 1)
    let prior_mean: f64 = 7.0;
    let prior_std: f64 = 2.0;
    let prior_precision = 1.0 / (prior_std * prior_std);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let measured_a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                let measured_a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]];
                let snr = harmonic_field.harmonic_snrs[0][[i, j, k]];

                let (ba_post, sigma_post, quality) = if measured_a1 > 1e-12 && snr > 5.0 {
                    // Harmonic-ratio observation as the likelihood mean
                    let beta =
                        beta_s_from_amplitudes(measured_a1, measured_a2, config).unwrap_or(1.0);
                    let ba_obs = ba_from_beta_s(beta).clamp(0.0, 20.0);

                    // Likelihood precision: σ_likelihood² ≈ (A₁/SNR)² normalised to B/A scale
                    // A₁ acts as the amplitude scale; the SNR gives fractional uncertainty.
                    let sigma_likelihood = (measured_a1 / snr.max(1.0)).max(1e-12) * 10.0;
                    let likelihood_precision = 1.0 / (sigma_likelihood * sigma_likelihood);

                    let precision_post = prior_precision + likelihood_precision;
                    let mean_post = (prior_mean * prior_precision + ba_obs * likelihood_precision)
                        / precision_post;
                    let std_post = (1.0 / precision_post).sqrt();
                    let quality = (snr / 20.0).min(1.0);

                    (
                        mean_post.clamp(0.0, 20.0),
                        std_post.clamp(0.1, 5.0),
                        quality,
                    )
                } else {
                    // Insufficient data: return to prior
                    (prior_mean, prior_std, 0.1)
                };

                nonlinearity_parameter[[i, j, k]] = ba_post;
                nonlinearity_uncertainty[[i, j, k]] = sigma_post;
                estimation_quality[[i, j, k]] = quality;

                // Elastic constants from MAP estimate
                let beta = ba_post / 2.0 + 1.0; // β_s = B/A/2 + 1
                let al = a_landau(mu, beta);
                elastic_constants[0][[i, j, k]] = al;
                elastic_constants[1][[i, j, k]] = al / 3.0;
                elastic_constants[2][[i, j, k]] = al / 9.0;
                elastic_constants[3][[i, j, k]] = al / 27.0;
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

// ─── Forward model (NLS support) ─────────────────────────────────────────────

/// Forward model: predict (A₁_pred, A₂_pred) given B/A and observed A₁.
///
/// Uses the shear-wave harmonic generation model (Rénier 2008, Eq. 7):
///
/// ```text
/// A₁_pred = a1_obs           (fundamental not altered by B/A in quasilinear model)
/// A₂_pred = β_s k_s a1² z / 2
/// ```
///
/// where β_s = B/A/2 + 1, k_s = ω/c_s, z = propagation_distance.
fn forward_model(ba_ratio: f64, a1_obs: f64, config: &NonlinearInversionConfig) -> (f64, f64) {
    let omega = 2.0 * PI * config.excitation_frequency;
    let c_s = config.shear_wave_speed.max(1e-3);
    let k_s = omega / c_s;
    let z = config.propagation_distance.max(1e-6);
    let beta = ba_ratio / 2.0 + 1.0; // β_s = B/A/2 + 1

    // A₂_pred = β_s k_s A₁² z / 2  [Rénier 2008, Eq. 7]
    let a2_pred = (beta * k_s * a1_obs * a1_obs * z / 2.0).max(0.0);
    (a1_obs, a2_pred)
}

/// Jacobian of the forward model with respect to B/A.
///
/// ```text
/// ∂A₁_pred / ∂(B/A) = 0
/// ∂A₂_pred / ∂(B/A) = k_s A₁² z / 4      (from ∂β_s/∂(B/A) = 1/2)
/// ```
fn forward_model_derivative(
    _ba_ratio: f64,
    a1_obs: f64,
    config: &NonlinearInversionConfig,
) -> (f64, f64) {
    let omega = 2.0 * PI * config.excitation_frequency;
    let c_s = config.shear_wave_speed.max(1e-3);
    let k_s = omega / c_s;
    let z = config.propagation_distance.max(1e-6);

    // ∂β_s/∂(B/A) = 1/2, so ∂A₂_pred/∂(B/A) = k_s A₁² z / 4
    let da2_dba = k_s * a1_obs * a1_obs * z / 4.0;
    (0.0_f64, da2_dba)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::imaging::ultrasound::elastography::NonlinearInversionMethod;

    fn test_config() -> NonlinearInversionConfig {
        NonlinearInversionConfig::new(NonlinearInversionMethod::HarmonicRatio)
            .with_shear_properties(3.0, 100.0, 0.05) // c_s=3 m/s, f=100 Hz, z=5 cm
    }

    // ── round-trip: synthesise A₂ from known B/A, recover it ─────────────────

    /// β_s → A₂ → β_s round-trip must return the input β_s to within rtol=1e-10.
    ///
    /// Forward: A₂ = β_s k_s A₁² z / 2
    /// Inverse: β_s_rec = 2 A₂ c_s / (ω A₁² z)   [Rénier 2008, Eq. 8]
    #[test]
    fn test_beta_s_round_trip() {
        let config = test_config();
        let omega = 2.0 * PI * config.excitation_frequency;
        let c_s = config.shear_wave_speed;
        let k_s = omega / c_s;
        let z = config.propagation_distance;

        for &ba_target in &[1.0_f64, 5.0, 10.0, 15.0] {
            let beta_target = ba_target / 2.0 + 1.0;
            let a1: f64 = 1e-6; // 1 µm fundamental displacement
            let a2 = beta_target * k_s * a1 * a1 * z / 2.0;

            let beta_rec = beta_s_from_amplitudes(a1, a2, &config).unwrap();
            let rel_err = (beta_rec - beta_target).abs() / beta_target.abs().max(1e-12);
            assert!(
                rel_err < 1e-10,
                "Round-trip failed for B/A={ba_target}: got β_s={beta_rec:.6}, \
                 expected {beta_target:.6}, rel_err={rel_err:.2e}"
            );
        }
    }

    /// ba_from_beta_s at β_s=1.0 (linear medium) must give B/A=0.
    #[test]
    fn test_ba_zero_for_linear_medium() {
        let ba = ba_from_beta_s(1.0);
        assert!(
            ba.abs() < 1e-15,
            "Linear medium (β_s=1) must give B/A=0, got {ba}"
        );
    }

    /// Landau constant A_L must be negative for tissues with β_s < 3/4.
    /// For β_s > 3/4 (all physical tissues): A_L = μ(4β_s − 3) > 0 when β_s > 3/4.
    #[test]
    fn test_a_landau_sign() {
        let mu = 9000.0; // Pa (c_s=3 m/s, ρ=1000 kg/m³)
                         // β_s = 2 (B/A = 2): should give positive A_L
        assert!(a_landau(mu, 2.0) > 0.0, "A_L should be positive for β_s=2");
        // β_s = 0.5: should give negative A_L (unphysical but valid maths)
        assert!(
            a_landau(mu, 0.5) < 0.0,
            "A_L should be negative for β_s=0.5"
        );
    }

    // ── tissue reference values (Rénier 2008 Table 1) ────────────────────────

    /// For a gelatin phantom: β_s ≈ 1.8, A₁=1µm, f=100 Hz, z=5 cm, c_s=3 m/s.
    /// Synthesise A₂ and verify recovered B/A ≈ 2×(1.8−1) = 1.6 within 0.1%.
    #[test]
    fn test_gelatin_phantom_reference_value() {
        let config = test_config();
        let omega = 2.0 * PI * config.excitation_frequency;
        let c_s = config.shear_wave_speed;
        let k_s = omega / c_s;
        let z = config.propagation_distance;

        let beta_ref = 1.8_f64; // gelatin 2% (Rénier 2008)
        let a1 = 1e-6_f64;
        let a2 = beta_ref * k_s * a1 * a1 * z / 2.0;

        let beta_rec = beta_s_from_amplitudes(a1, a2, &config).unwrap();
        let ba_rec = ba_from_beta_s(beta_rec);
        let ba_expected = ba_from_beta_s(beta_ref); // = 1.6

        let err = (ba_rec - ba_expected).abs();
        assert!(
            err < 1e-3 * ba_expected.abs().max(1.0),
            "Gelatin phantom B/A: got {ba_rec:.4}, expected {ba_expected:.4}, err={err:.2e}"
        );
    }

    // ── forward model self-consistency ───────────────────────────────────────

    /// forward_model + beta_s_from_amplitudes must be inverses of each other.
    #[test]
    fn test_forward_model_invertible() {
        let config = test_config();
        let a1 = 2e-6; // 2 µm
        let ba_in = 6.0_f64; // liver-like

        let (_a1_pred, a2_pred) = forward_model(ba_in, a1, &config);
        let beta_rec = beta_s_from_amplitudes(a1, a2_pred, &config).unwrap();
        let ba_rec = ba_from_beta_s(beta_rec).clamp(0.0, 20.0);

        let err = (ba_rec - ba_in).abs();
        assert!(
            err < 1e-9,
            "forward_model/beta_s_from_amplitudes inverse: got {ba_rec:.6}, \
             expected {ba_in:.6}, err={err:.2e}"
        );
    }

    /// Jacobian: numerical derivative must agree with analytical derivative to 0.1%.
    #[test]
    fn test_forward_model_jacobian_numerical() {
        let config = test_config();
        let a1 = 1e-6_f64;
        let ba0 = 5.0_f64;
        let h = 1e-4;

        let (_, a2_pos) = forward_model(ba0 + h, a1, &config);
        let (_, a2_neg) = forward_model(ba0 - h, a1, &config);
        let da2_numerical = (a2_pos - a2_neg) / (2.0 * h);

        let (_, da2_analytical) = forward_model_derivative(ba0, a1, &config);

        let rel_err = (da2_numerical - da2_analytical).abs() / da2_analytical.abs().max(1e-30);
        assert!(
            rel_err < 1e-3,
            "Jacobian rel_err={rel_err:.2e}: numerical={da2_numerical:.4e}, \
             analytical={da2_analytical:.4e}"
        );
    }

    // ── inversion pipeline tests ─────────────────────────────────────────────

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
    fn test_nonlinear_inversion_processor() {
        let config = NonlinearInversionConfig::new(NonlinearInversionMethod::HarmonicRatio);
        let processor = NonlinearInversion::new(config);

        assert_eq!(processor.method(), NonlinearInversionMethod::HarmonicRatio);
        assert_eq!(processor.config().density, 1000.0);
        assert_eq!(processor.config().acoustic_speed, 1540.0);
        assert_eq!(processor.config().shear_wave_speed, 3.0);
        assert_eq!(processor.config().excitation_frequency, 100.0);
        assert_eq!(processor.config().propagation_distance, 0.05);
    }

    /// Shear modulus helper: μ = ρ c_s² for default config (ρ=1000, c_s=3) → 9000 Pa.
    #[test]
    fn test_shear_modulus_default() {
        let config = NonlinearInversionConfig::default();
        let mu = shear_modulus(&config);
        let expected = 1000.0 * 3.0 * 3.0; // 9000 Pa
        assert!(
            (mu - expected).abs() < 1e-10,
            "shear_modulus should be {expected}, got {mu}"
        );
    }
}
