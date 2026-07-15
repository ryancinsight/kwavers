//! Bremsstrahlung emission model.

use leto::Array1;

use super::constants::c_ff_per_sr;
use super::gaunt::gaunt_factor_thermal;
use super::plasma::PlasmaState;
use kwavers_core::constants::fundamental::{
    BOLTZMANN as BOLTZMANN_CONSTANT, PLANCK as PLANCK_CONSTANT, SPEED_OF_LIGHT,
};
use kwavers_core::constants::numerical::FOUR_PI;

/// Bremsstrahlung free-free radiation model for sonoluminescence.
#[derive(Debug, Clone)]
pub struct BremsstrahlungModel {
    /// Average ion charge number.
    pub z_ion: f64,
    /// Use temperature/frequency-dependent Gaunt factor.
    pub use_thermal_gaunt_factor: bool,
    /// Fixed Gaunt factor used when thermal Gaunt factors are disabled.
    pub fixed_gaunt_factor: f64,
}

impl Default for BremsstrahlungModel {
    fn default() -> Self {
        Self {
            z_ion: 1.0,
            use_thermal_gaunt_factor: true,
            fixed_gaunt_factor: 1.2,
        }
    }
}

impl BremsstrahlungModel {
    /// Bremsstrahlung emission coefficient per unit volume, frequency, and steradian.
    #[must_use]
    pub fn emission_coefficient(
        &self,
        frequency: f64,
        temperature: f64,
        n_electron: f64,
        n_ion: f64,
    ) -> f64 {
        if temperature <= 0.0 || frequency <= 0.0 || n_electron <= 0.0 || n_ion <= 0.0 {
            return 0.0;
        }

        let g_ff = if self.use_thermal_gaunt_factor {
            gaunt_factor_thermal(frequency, temperature)
        } else {
            self.fixed_gaunt_factor
        };
        let h_nu = PLANCK_CONSTANT * frequency;
        let exp_factor = (-h_nu / (BOLTZMANN_CONSTANT * temperature)).exp();

        c_ff_per_sr() * self.z_ion.powi(2) * g_ff * n_electron * n_ion / temperature.sqrt()
            * exp_factor
    }

    /// Spectral radiance at wavelength [W m^-2 sr^-1 m^-1].
    ///
    /// # Theorem
    ///
    /// The frequency-to-wavelength density conversion is
    /// `L_lambda = j_nu |dnu/dlambda| ell = j_nu c ell / lambda^2`.
    #[must_use]
    pub fn spectral_radiance(
        &self,
        wavelength: f64,
        temperature: f64,
        n_electron: f64,
        n_ion: f64,
        path_length: f64,
    ) -> f64 {
        if wavelength <= 0.0 || path_length <= 0.0 {
            return 0.0;
        }
        let frequency = SPEED_OF_LIGHT / wavelength;
        let j_nu = self.emission_coefficient(frequency, temperature, n_electron, n_ion);
        j_nu * SPEED_OF_LIGHT / wavelength.powi(2) * path_length
    }

    /// Total bremsstrahlung power in `volume`.
    #[must_use]
    pub fn total_power(&self, temperature: f64, n_electron: f64, n_ion: f64, volume: f64) -> f64 {
        if temperature <= 0.0 || n_electron <= 0.0 || n_ion <= 0.0 || volume <= 0.0 {
            return 0.0;
        }

        // Integrate j_ν ∝ T^{-1/2}·exp(-hν/kT) over ν: ∫exp(-hν/kT)dν = kT/h,
        // so total power ∝ T^{-1/2} · kT/h = (k/h)·T^{1/2}.
        let c_total =
            c_ff_per_sr() * FOUR_PI * BOLTZMANN_CONSTANT * temperature.sqrt() / PLANCK_CONSTANT;
        c_total * self.z_ion.powi(2) * self.fixed_gaunt_factor * n_electron * n_ion * volume
    }

    /// Compute single-stage Saha ionization fraction.
    #[must_use]
    pub fn saha_ionization(&self, temperature: f64, pressure: f64, ionization_energy: f64) -> f64 {
        let state = PlasmaState::from_single_stage(temperature, pressure, ionization_energy, 1.0);
        state.ionization_fraction
    }

    /// Compute full emission spectrum over wavelength samples.
    #[must_use]
    pub fn emission_spectrum(
        &self,
        temperature: f64,
        n_electron: f64,
        n_ion: f64,
        path_length: f64,
        wavelengths: &Array1<f64>,
    ) -> Array1<f64> {
        wavelengths.mapv(|lambda| {
            self.spectral_radiance(lambda, temperature, n_electron, n_ion, path_length)
        })
    }

    /// Compute self-consistent emission from temperature and pressure.
    #[must_use]
    pub fn emission_from_temperature_pressure(
        &self,
        frequency: f64,
        temperature: f64,
        pressure: f64,
        ionization_energy: f64,
    ) -> f64 {
        let state = PlasmaState::from_single_stage(temperature, pressure, ionization_energy, 1.0);
        self.emission_coefficient(
            frequency,
            temperature,
            state.electron_density,
            state.ion_density_z2,
        )
    }
}
