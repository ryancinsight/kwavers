//! Bayesian MAP inversion with Gaussian prior for soft-tissue B/A.
//!
//! Prior: B/A ~ N(7, 2²) (Rénier 2008 Table 1).
//! Posterior update: conjugate Gaussian (Gelman et al. 2013, §2.4).

use super::super::config::NonlinearInversionConfig;
use super::helpers::{a_landau, ba_from_beta_s, beta_s_from_amplitudes, shear_modulus};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::NonlinearParameterMap;
use kwavers_physics::acoustics::imaging::modalities::elastography::HarmonicDisplacementField;
use leto::Array3;

/// Bayesian MAP inversion with uncertainty quantification.
///
/// Prior: B/A ~ N(7, 2²). Posterior via conjugate Gaussian update.
///
/// # References
///
/// - Gelman et al. (2013). *Bayesian Data Analysis*, 3rd ed. §2.4.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(super) fn bayesian_inversion(
    harmonic_field: &HarmonicDisplacementField,
    _grid: &Grid,
    config: &NonlinearInversionConfig,
) -> KwaversResult<NonlinearParameterMap> {
    let [nx, ny, nz] = harmonic_field.fundamental_magnitude.shape();

    let mut nonlinearity_parameter = Array3::zeros([nx, ny, nz]);
    let mut nonlinearity_uncertainty = Array3::zeros([nx, ny, nz]);
    let mut estimation_quality = Array3::zeros([nx, ny, nz]);

    let mut elastic_constants = vec![
        Array3::zeros([nx, ny, nz]),
        Array3::zeros([nx, ny, nz]),
        Array3::zeros([nx, ny, nz]),
        Array3::zeros([nx, ny, nz]),
    ];

    let mu = shear_modulus(config);

    // Gaussian prior: B/A ~ N(7, 2²) for soft tissue (Rénier 2008 Table 1)
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
                    let beta =
                        beta_s_from_amplitudes(measured_a1, measured_a2, config).unwrap_or(1.0);
                    let ba_obs = ba_from_beta_s(beta).clamp(0.0, 20.0);

                    let sigma_likelihood = (measured_a1 / snr.max(1.0)).max(1e-12) * 10.0;
                    let likelihood_precision = 1.0 / (sigma_likelihood * sigma_likelihood);

                    let precision_post = prior_precision + likelihood_precision;
                    let mean_post = prior_mean
                        .mul_add(prior_precision, ba_obs * likelihood_precision)
                        / precision_post;
                    let std_post = (1.0 / precision_post).sqrt();
                    let quality = (snr / 20.0).min(1.0);

                    (
                        mean_post.clamp(0.0, 20.0),
                        std_post.clamp(0.1, 5.0),
                        quality,
                    )
                } else {
                    (prior_mean, prior_std, 0.1)
                };

                nonlinearity_parameter[[i, j, k]] = ba_post;
                nonlinearity_uncertainty[[i, j, k]] = sigma_post;
                estimation_quality[[i, j, k]] = quality;

                let beta = ba_post / 2.0 + 1.0;
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
