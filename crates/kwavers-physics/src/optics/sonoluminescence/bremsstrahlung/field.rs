//! Field assembly for bremsstrahlung power density.

use leto::Array3;

use super::model::BremsstrahlungModel;

/// Calculate total bremsstrahlung emission field [W m^-3].
#[must_use]
pub fn calculate_bremsstrahlung_emission(
    temperature_field: &Array3<f64>,
    electron_density_field: &Array3<f64>,
    ion_density_field: &Array3<f64>,
    model: &BremsstrahlungModel,
) -> Array3<f64> {
    let mut emission_field = Array3::zeros(temperature_field.shape());

    leto_ops::zip_mut_with(
        emission_field.view_mut(),
        (
            &temperature_field.view(),
            &electron_density_field.view(),
            &ion_density_field.view(),
        ),
        |out, (&temp, &n_electron, &n_ion)| {
            if n_electron > 0.0 && n_ion > 0.0 && temp > 0.0 {
                *out = model.total_power(temp, n_electron, n_ion, 1.0);
            }
        },
    )
    .expect("invariant: bremsstrahlung input and emission fields have matching shapes");

    emission_field
}
