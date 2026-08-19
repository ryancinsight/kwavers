//! Sonoluminescence Emission Calculator
//!
//! Calculates dimensioned blackbody and bremsstrahlung emission fields and
//! arbitrary-unit spectra for the supported radiation mechanisms.

use leto::Array3;

use aequitas::systems::si::quantities::VolumetricPowerDensity;

use kwavers_core::constants::fundamental::{BOLTZMANN, ELEMENTARY_CHARGE};

use crate::optics::sonoluminescence::blackbody::{blackbody_power_density, BlackbodyModel};
use crate::optics::sonoluminescence::bremsstrahlung::{
    bremsstrahlung_power_density, BremsstrahlungModel,
};
use crate::optics::sonoluminescence::cherenkov::CherenkovModel;
use crate::optics::sonoluminescence::spectral::{
    EmissionSpectrum, SpectralAnalyzer, SpectralRange,
};

use crate::optics::sonoluminescence::emission::orchestrator::components::EmissionComponents;
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

    /// Calculate dimensioned light emission from bubble fields.
    ///
    /// The output contains blackbody and bremsstrahlung power density only.
    /// Cherenkov threshold yield remains on the arbitrary-unit spectral path.
    pub fn calculate_emission(
        &mut self,
        temperature_field: &Array3<f64>,
        radius_field: &Array3<f64>,
        charge_density_field: &Array3<f64>,
    ) {
        let params = &self.params;
        let blackbody = &self.blackbody;
        let bremsstrahlung = &self.bremsstrahlung;

        crate::parallel::zip_mut_three_refs(
            self.emission_field.view_mut(),
            temperature_field.view(),
            radius_field.view(),
            charge_density_field.view(),
            |out, &temperature, &radius, &charge_density| {
                if temperature < params.min_temperature {
                    *out = 0.0;
                    return;
                }
                let components = components_at_point(
                    temperature,
                    radius,
                    charge_density,
                    params,
                    blackbody,
                    bremsstrahlung,
                );
                *out = params.opacity_factor * components.total().into_base();
            },
        );
    }

    /// Calculate dimensioned emission components for one spatial cell.
    #[must_use]
    pub fn components_at_point(
        &self,
        temperature: f64,
        radius: f64,
        charge_density: f64,
    ) -> EmissionComponents {
        components_at_point(
            temperature,
            radius,
            charge_density,
            &self.params,
            &self.blackbody,
            &self.bremsstrahlung,
        )
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

fn components_at_point(
    temperature: f64,
    radius: f64,
    charge_density: f64,
    params: &EmissionParameters,
    blackbody: &BlackbodyModel,
    bremsstrahlung: &BremsstrahlungModel,
) -> EmissionComponents {
    let blackbody = params
        .use_blackbody
        .then(|| blackbody_power_density(temperature, radius, blackbody))
        .unwrap_or(0.0);
    let electron_density = charge_density / ELEMENTARY_CHARGE;
    let bremsstrahlung = params
        .use_bremsstrahlung
        .then(|| {
            bremsstrahlung_power_density(
                temperature,
                electron_density,
                electron_density,
                bremsstrahlung,
            )
        })
        .unwrap_or(0.0);

    EmissionComponents::new(
        VolumetricPowerDensity::from_base(blackbody),
        VolumetricPowerDensity::from_base(bremsstrahlung),
    )
}
