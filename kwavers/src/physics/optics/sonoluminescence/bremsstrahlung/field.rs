//! Field assembly for bremsstrahlung power density.

use ndarray::{Array3, Zip};

use super::model::BremsstrahlungModel;

/// Calculate total bremsstrahlung emission field [W m^-3].
#[must_use]
pub fn calculate_bremsstrahlung_emission(
    temperature_field: &Array3<f64>,
    electron_density_field: &Array3<f64>,
    ion_density_field: &Array3<f64>,
    model: &BremsstrahlungModel,
) -> Array3<f64> {
    let mut emission_field = Array3::zeros(temperature_field.dim());

    Zip::from(&mut emission_field)
        .and(temperature_field)
        .and(electron_density_field)
        .and(ion_density_field)
        .for_each(|out, &temp, &n_electron, &n_ion| {
            if n_electron > 0.0 && n_ion > 0.0 && temp > 0.0 {
                *out = model.total_power(temp, n_electron, n_ion, 1.0);
            }
        });

    emission_field
}
