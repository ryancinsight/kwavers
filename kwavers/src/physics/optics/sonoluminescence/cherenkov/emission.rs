use super::model::CherenkovModel;
use crate::core::constants::fundamental::SPEED_OF_LIGHT;
use ndarray::{Array3, Zip};

/// Piezo-optic coefficient: Δn per unit compression excess.
///
/// For water, the strain-optic coefficient p₁₂ ≈ 0.31 and the
/// Lorentz-Lorenz relation gives dn/dε ≈ (n²-1)(n²+2)/(6n) × p₁₂ ≈ 0.02
/// at n ≈ 1.33.
///
/// Reference: Balakin et al., J. Opt. Soc. Am. B, 2016.
pub(crate) const COMPRESSION_REFRACTIVE_COEFFICIENT: f64 = 0.02;

/// Thermo-optic coefficient dn/dT for water [K⁻¹].
///
/// Negative sign: heating reduces density and refractive index.
/// Measured value: approximately −1 × 10⁻⁵ K⁻¹ near room temperature.
///
/// Reference: Kim, J. Opt. Soc. Korea, 2012.
pub(crate) const THERMAL_REFRACTIVE_COEFFICIENT: f64 = 1e-5;

/// Reference temperature for thermo-optic shift [K]
pub(crate) const REFERENCE_TEMPERATURE: f64 = 300.0;

/// Compute Cherenkov emission field over a grid from physical fields
///
/// ## Algorithm
///
/// At each voxel, the local refractive index is adjusted for
/// compression (piezo-optic effect) and temperature (thermo-optic
/// effect). Emission occurs only where particle velocity exceeds
/// the local phase velocity c/n. The Frank–Tamm formula then gives
/// spectral yield ∝ (1 − 1/(n²β²)).
///
/// ## References
///
/// - Frank, I. & Tamm, I. (1937). Dokl. Akad. Nauk SSSR, 14, 109–114.
#[must_use]
pub fn calculate_cherenkov_emission(
    velocity_field: &Array3<f64>,
    charge_density_field: &Array3<f64>,
    temperature_field: &Array3<f64>,
    compression_field: &Array3<f64>,
    model: &CherenkovModel,
) -> Array3<f64> {
    let shape = velocity_field.raw_dim();
    assert_eq!(shape, charge_density_field.raw_dim());
    assert_eq!(shape, temperature_field.raw_dim());
    assert_eq!(shape, compression_field.raw_dim());

    let mut emission = Array3::zeros(shape);
    let n_base = model.refractive_index_base;
    let coherence = model.coherence_factor;

    Zip::from(&mut emission)
        .and(velocity_field)
        .and(charge_density_field)
        .and(temperature_field)
        .and(compression_field)
        .for_each(|e, &v, &charge_density, &temp, &comp| {
            let charge_density = charge_density.max(0.0);
            let temp = temp.max(0.0);
            let comp = comp.max(0.0);

            // Local refractive index: piezo-optic + thermo-optic correction
            let increased_n = n_base * (1.0 + COMPRESSION_REFRACTIVE_COEFFICIENT * (comp - 1.0));
            let n_local = (increased_n - THERMAL_REFRACTIVE_COEFFICIENT * (temp - REFERENCE_TEMPERATURE)).max(1.0);

            let critical = SPEED_OF_LIGHT / n_local;
            if v <= critical {
                *e = 0.0;
                return;
            }

            let beta = v / SPEED_OF_LIGHT;
            let threshold_term = (1.0 - 1.0 / (n_local.powi(2) * beta.powi(2))).max(0.0);
            // Frank–Tamm emission per cell
            *e = coherence * charge_density * threshold_term;
        });
    emission
}
