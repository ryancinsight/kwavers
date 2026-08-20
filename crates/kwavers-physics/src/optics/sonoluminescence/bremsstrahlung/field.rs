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

    crate::parallel::zip_mut_three_refs(
        emission_field.view_mut(),
        temperature_field.view(),
        electron_density_field.view(),
        ion_density_field.view(),
        |out, &temp, &n_electron, &n_ion| {
            *out = bremsstrahlung_power_density(temp, n_electron, n_ion, model);
        },
    );

    emission_field
}

/// Calculate bremsstrahlung power density for one cell in `W/m³`.
#[must_use]
pub fn bremsstrahlung_power_density(
    temperature: f64,
    electron_density: f64,
    ion_density: f64,
    model: &BremsstrahlungModel,
) -> f64 {
    model.total_power(temperature, electron_density, ion_density, 1.0)
}
