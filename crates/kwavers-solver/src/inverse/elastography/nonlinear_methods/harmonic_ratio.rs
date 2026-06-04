//! Harmonic ratio method: B/A from A₂/A₁ (Rénier et al. 2008).

use super::super::config::NonlinearInversionConfig;
use super::helpers::{a_landau, ba_from_beta_s, beta_s_from_amplitudes, shear_modulus};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_domain::imaging::ultrasound::elastography::NonlinearParameterMap;
use kwavers_physics::acoustics::imaging::modalities::elastography::HarmonicDisplacementField;
use ndarray::Array3;

/// Harmonic ratio method: B/A from A₂/A₁.
///
/// # Algorithm
///
/// 1. For each voxel, read A₁ (fundamental) and A₂ (second-harmonic).
/// 2. If A₁ ≥ 1e-12 m, compute β_s = 2 A₂ c_s / (ω A₁² z).
/// 3. Clamp B/A = 2(β_s−1) to [0, 20] (physiological range).
/// 4. Compute Landau constant A_L = μ(4β_s − 3) and derive A, B, C, D.
/// 5. SNR-based estimation uncertainty: σ ≈ (10/SNR) clamped to [0.1, 5.0].
///
/// # References
///
/// - Rénier et al. (2008), JASA 124(5) 2856, Eq. 7–8.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(super) fn harmonic_ratio_inversion(
    harmonic_field: &HarmonicDisplacementField,
    _grid: &Grid,
    config: &NonlinearInversionConfig,
) -> KwaversResult<NonlinearParameterMap> {
    let (nx, ny, nz) = harmonic_field.fundamental_magnitude.dim();

    let mut nonlinearity_parameter = Array3::zeros((nx, ny, nz));
    let mut nonlinearity_uncertainty = Array3::zeros((nx, ny, nz));
    let mut estimation_quality = Array3::zeros((nx, ny, nz));

    let mut elastic_constants = vec![
        Array3::zeros((nx, ny, nz)), // A = A_L
        Array3::zeros((nx, ny, nz)), // B ≈ A_L / 3
        Array3::zeros((nx, ny, nz)), // C ≈ A_L / 9
        Array3::zeros((nx, ny, nz)), // D ≈ A_L / 27
    ];

    let mu = shear_modulus(config);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let a1 = harmonic_field.fundamental_magnitude[[i, j, k]];
                let a2 = harmonic_field.harmonic_magnitudes[0][[i, j, k]];

                if let Some(beta) = beta_s_from_amplitudes(a1, a2, config) {
                    let ba_ratio = ba_from_beta_s(beta).clamp(0.0, 20.0);
                    let al = a_landau(mu, beta);

                    nonlinearity_parameter[[i, j, k]] = ba_ratio;

                    let snr = harmonic_field.harmonic_snrs[0][[i, j, k]];
                    nonlinearity_uncertainty[[i, j, k]] = (10.0 / snr.max(1e-3)).clamp(0.1, 5.0);
                    estimation_quality[[i, j, k]] = (snr / 10.0).min(1.0) * (a1 / 1e-6).min(1.0);

                    elastic_constants[0][[i, j, k]] = al;
                    elastic_constants[1][[i, j, k]] = al / 3.0;
                    elastic_constants[2][[i, j, k]] = al / 9.0;
                    elastic_constants[3][[i, j, k]] = al / 27.0;
                } else {
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
