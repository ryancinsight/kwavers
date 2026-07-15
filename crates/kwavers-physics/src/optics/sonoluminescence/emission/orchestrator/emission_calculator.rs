//! Sonoluminescence Emission Calculator
//!
//! Calculates total light emission from blackbody, bremsstrahlung,
//! and Cherenkov radiation mechanisms at each spatial point.

use leto::Array3;

use kwavers_core::constants::fundamental::{BOLTZMANN, ELEMENTARY_CHARGE};

use crate::optics::sonoluminescence::blackbody::{calculate_blackbody_emission, BlackbodyModel};
use crate::optics::sonoluminescence::bremsstrahlung::{
    calculate_bremsstrahlung_emission, BremsstrahlungModel,
};
use crate::optics::sonoluminescence::cherenkov::{calculate_cherenkov_emission, CherenkovModel};
use crate::optics::sonoluminescence::spectral::{
    EmissionSpectrum, SpectralAnalyzer, SpectralRange,
};

use crate::optics::sonoluminescence::emission::spectrum::{EmissionParameters, SpectralField};

/// Main sonoluminescence emission calculator
#[derive(Debug)]
pub struct SonoluminescenceEmission {
    /// Emission parameters
    pub params: EmissionParameters,
    /// Blackbody model
    pub blackbody: BlackbodyModel,
    /// Bremsstrahlung model
    pub bremsstrahlung: BremsstrahlungModel,
    /// Cherenkov model
    pub cherenkov: CherenkovModel,
    /// Spectral analyzer
    pub analyzer: SpectralAnalyzer,
    /// Total emission field (W/m³)
    pub emission_field: Array3<f64>,
    /// Spectral emission field (Struct-of-Arrays)
    pub spectral_field: Option<SpectralField>,
}

impl SonoluminescenceEmission {
    /// Create new emission calculator
    #[must_use]
    pub fn new(grid_shape: [usize; 3], params: EmissionParameters) -> Self {
        let analyzer = SpectralAnalyzer::new(SpectralRange::default());
        let spectral_field = Some(SpectralField::new(grid_shape, analyzer.range.wavelengths()));

        Self {
            params: params.clone(),
            blackbody: BlackbodyModel::default(),
            bremsstrahlung: BremsstrahlungModel::default(),
            cherenkov: CherenkovModel::new(
                params.cherenkov_refractive_index,
                params.cherenkov_coherence_factor,
            ),
            analyzer,
            emission_field: Array3::zeros(grid_shape),
            spectral_field,
        }
    }

    /// Calculate total light emission from bubble fields
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_emission(
        &mut self,
        temperature_field: &Array3<f64>,
        _pressure_field: &Array3<f64>,
        radius_field: &Array3<f64>,
        velocity_field: &Array3<f64>,
        charge_density_field: &Array3<f64>,
        compression_field: &Array3<f64>,
        _time: f64,
    ) {
        self.emission_field.fill(0.0);

        // Blackbody contribution
        if self.params.use_blackbody {
            let bb_emission =
                calculate_blackbody_emission(temperature_field, radius_field, &self.blackbody);
            for (dst, src) in self.emission_field.iter_mut().zip(bb_emission.iter()) {
                *dst += *src;
            }
        }

        // Bremsstrahlung contribution
        if self.params.use_bremsstrahlung {
            let mut electron_density_field = charge_density_field.clone();
            for v in electron_density_field.iter_mut() {
                *v /= ELEMENTARY_CHARGE;
            }
            let ion_density_field = electron_density_field.clone();

            let br_emission = calculate_bremsstrahlung_emission(
                temperature_field,
                &electron_density_field,
                &ion_density_field,
                &self.bremsstrahlung,
            );
            for (dst, src) in self.emission_field.iter_mut().zip(br_emission.iter()) {
                *dst += *src;
            }
        }

        // Cherenkov contribution
        if self.params.use_cherenkov {
            let ch_emission = calculate_cherenkov_emission(
                velocity_field,
                charge_density_field,
                temperature_field,
                compression_field,
                &self.cherenkov,
            );
            for (dst, src) in self.emission_field.iter_mut().zip(ch_emission.iter()) {
                *dst += *src;
            }
        }

        // Apply minimum temperature cutoff
        for ([i, j, k], emission) in self.emission_field.indexed_iter_mut().unwrap() {
            if temperature_field[[i, j, k]] < self.params.min_temperature {
                *emission = 0.0;
            } else {
                *emission *= self.params.opacity_factor;
            }
        }
    }

    /// Calculate spectral emission at a specific point
    #[must_use]
    pub fn calculate_spectrum_at_point(
        &self,
        temperature: f64,
        pressure: f64,
        radius: f64,
        velocity: f64,
        charge_density: f64,
        compression: f64,
    ) -> EmissionSpectrum {
        let wavelengths = self.analyzer.range.wavelengths();
        let mut intensities = leto::Array1::zeros([wavelengths.len()]);

        if temperature < self.params.min_temperature || radius <= 0.0 {
            return EmissionSpectrum::new(wavelengths, intensities, 0.0);
        }

        // Blackbody contribution
        if self.params.use_blackbody {
            let bb_spectrum = self.blackbody.emission_spectrum(temperature, &wavelengths);
            for (dst, src) in intensities.iter_mut().zip(bb_spectrum.iter()) {
                *dst += *src;
            }
        }

        // Bremsstrahlung contribution
        if self.params.use_bremsstrahlung && temperature > 5000.0 {
            let x_ion = self.bremsstrahlung.saha_ionization(
                temperature,
                pressure,
                self.params.ionization_energy,
            );

            let n_total = pressure / (BOLTZMANN * temperature);
            let n_electron = x_ion * n_total;
            let n_ion = n_electron;

            let br_spectrum = self.bremsstrahlung.emission_spectrum(
                temperature,
                n_electron,
                n_ion,
                2.0 * radius,
                &wavelengths,
            );
            for (dst, src) in intensities.iter_mut().zip(br_spectrum.iter()) {
                *dst += *src;
            }
        }

        // Cherenkov contribution
        if self.params.use_cherenkov && velocity > 0.0 && charge_density > 0.0 {
            let mut local_model = self.cherenkov.clone();
            local_model.update_refractive_index(compression, temperature);

            if local_model.exceeds_threshold(velocity) {
                let charge_per_particle = 1.0;
                let ch_spectrum =
                    local_model.emission_spectrum(velocity, charge_per_particle, &wavelengths);
                let path_length = 2.0 * radius;
                let scale_factor = charge_density * path_length;
                for (dst, src) in intensities.iter_mut().zip(ch_spectrum.iter()) {
                    *dst += src * scale_factor;
                }
            }
        }

        for v in intensities.iter_mut() {
            *v *= self.params.opacity_factor;
        }
        EmissionSpectrum::new(wavelengths, intensities, 0.0)
    }

    /// Calculate full spectral field
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_spectral_field(
        &mut self,
        temperature_field: &Array3<f64>,
        pressure_field: &Array3<f64>,
        radius_field: &Array3<f64>,
        velocity_field: &Array3<f64>,
        charge_density_field: &Array3<f64>,
        compression_field: &Array3<f64>,
        time: f64,
    ) {
        let shape = temperature_field.shape();
        let wavelengths = self.analyzer.range.wavelengths();
        let mut spectral_field = SpectralField::new(shape, wavelengths);

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let mut spectrum = self.calculate_spectrum_at_point(
                        temperature_field[[i, j, k]],
                        pressure_field[[i, j, k]],
                        radius_field[[i, j, k]],
                        velocity_field[[i, j, k]],
                        charge_density_field[[i, j, k]],
                        compression_field[[i, j, k]],
                    );
                    spectrum.time = time;
                    spectrum.position = Some((i, j, k));
                    for (idx, &intensity) in spectrum.intensities.iter().enumerate() {
                        spectral_field.intensities[[i, j, k, idx]] = intensity;
                    }
                }
            }
        }

        spectral_field.update_derived_quantities();
        self.spectral_field = Some(spectral_field);
    }
}
