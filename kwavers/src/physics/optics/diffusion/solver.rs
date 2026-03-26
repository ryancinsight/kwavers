//! Light diffusion solver implementation

use super::properties::OpticalProperties;
use crate::core::constants::optical::{DEFAULT_POLARIZATION_FACTOR, LAPLACIAN_CENTER_COEFF};
use crate::domain::field::indices::LIGHT_IDX;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::physics::acoustics::traits::LightDiffusionModelTrait;
use crate::physics::optics::polarization::LinearPolarization;
use crate::physics::optics::PolarizationModel as PolarizationModelTrait;
use crate::physics::wave_propagation::scattering::ScatteringCalculator;
use log::debug;
use ndarray::{Array3, Array4, Axis};
use std::time::Instant;

/// Time-domain light diffusion solver
#[derive(Debug)]
pub struct LightDiffusion {
    /// Photon fluence rate field [photons/(m²·s)]
    pub fluence_rate: Array4<f64>,
    /// Emission spectrum field [photons/(m³·s·Hz)]
    pub emission_spectrum: Array3<f64>,
    /// Physical optical properties of the medium
    optical_properties: OpticalProperties,
    /// Polarization model (optional)
    _polarization: Option<Box<dyn PolarizationModelTrait>>,
    /// Scattering calculator (optional)
    _scattering: Option<ScatteringCalculator>,
    /// Feature flags
    _enable_polarization: bool,
    _enable_scattering: bool,
    // Performance metrics
    update_time: f64,
    fft_time: f64,
    diffusion_time: f64,
    effect_time: f64,
    call_count: usize,
}

impl LightDiffusion {
    /// Create a new light diffusion solver with physical optical properties
    ///
    /// # Arguments
    /// * `grid` - Computational grid
    /// * `optical_properties` - Physical optical properties of the medium
    /// * `enable_polarization` - Enable polarization effects
    /// * `enable_scattering` - Enable scattering calculations
    ///
    /// # Mathematical Foundation
    /// The photon diffusion equation is derived from the radiative transfer equation
    /// under the diffusion approximation (P1 approximation):
    ///
    /// ∂φ/∂t = ∇·(D∇φ) - μₐφ + S
    ///
    /// where:
    /// - φ is the photon fluence rate [photons/(m²·s)]
    /// - D = 1/(3(μₐ + μₛ')) is the diffusion coefficient [m]
    /// - μₐ is the absorption coefficient [m⁻¹]
    /// - μₛ' is the reduced scattering coefficient [m⁻¹]
    /// - S is the source term [photons/(m³·s)]
    ///
    /// The diffusion approximation is valid when μₛ' ≫ μₐ (scattering dominates).
    pub fn new(
        grid: &Grid,
        optical_properties: OpticalProperties,
        enable_polarization: bool,
        enable_scattering: bool,
    ) -> Self {
        let (nx, ny, nz) = grid.dimensions();

        // Validate diffusion approximation
        if !optical_properties.diffusion_approximation_valid() {
            log::warn!(
                "Diffusion approximation may not be valid: μₛ'/μₐ = {:.1}, should be ≫ 10",
                optical_properties.reduced_scattering_coefficient
                    / optical_properties.absorption_coefficient.max(1e-10)
            );
        }

        Self {
            fluence_rate: Array4::zeros((1, nx, ny, nz)),
            emission_spectrum: Array3::zeros((nx, ny, nz)),
            optical_properties,
            _polarization: if enable_polarization {
                Some(Box::new(LinearPolarization::new(
                    DEFAULT_POLARIZATION_FACTOR,
                )))
            } else {
                None
            },
            _scattering: if enable_scattering {
                // Default optical frequency for scattering
                let frequency = 5e14; // ~600nm wavelength
                let wave_speed = 3e8; // Speed of light
                Some(ScatteringCalculator::new(frequency, wave_speed))
            } else {
                None
            },
            _enable_polarization: enable_polarization,
            _enable_scattering: enable_scattering,
            update_time: 0.0,
            fft_time: 0.0,
            diffusion_time: 0.0,
            effect_time: 0.0,
            call_count: 0,
        }
    }
}

impl LightDiffusionModelTrait for LightDiffusion {
    fn update_light(
        &mut self,
        fields: &mut Array4<f64>,
        _light_source: &Array3<f64>,
        grid: &Grid,
        _medium: &dyn Medium,
        dt: f64,
    ) {
        let start_time = Instant::now();

        // Update the light field in the fields array
        let mut light_field = fields.index_axis_mut(Axis(0), LIGHT_IDX);

        // Get dimensions
        let (nx, ny, nz) = light_field.dim();

        // Photon diffusion equation: ∂φ/∂t = ∇·(D∇φ) - μₐφ + S
        // where φ is photon fluence rate, D is diffusion coefficient, μₐ is absorption coefficient
        //
        // Physical diffusion coefficient from optical properties
        let diffusion_coefficient = self.optical_properties.diffusion_coefficient();
        let absorption_coeff = self.optical_properties.absorption_coefficient;

        // Convert units if necessary (grid is in meters, coefficients in m⁻¹)
        // diffusion_coefficient is in m²/s, absorption_coeff in m⁻¹

        // Create a temporary array to store the updated values
        let mut updated_field = light_field.to_owned();

        let dx2_inv = 1.0 / (grid.dx * grid.dx);
        let dy2_inv = 1.0 / (grid.dy * grid.dy);
        let dz2_inv = 1.0 / (grid.dz * grid.dz);

        // Apply diffusion equation with second-order central differences
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let center_val = light_field[[i, j, k]];

                    // Source term from sonoluminescence
                    let source_term = self.emission_spectrum[[i, j, k]];

                    let laplacian_phi = (light_field[[i + 1, j, k]]
                        + LAPLACIAN_CENTER_COEFF * center_val
                        + light_field[[i - 1, j, k]])
                        * dx2_inv
                        + (light_field[[i, j + 1, k]]
                            + LAPLACIAN_CENTER_COEFF * center_val
                            + light_field[[i, j - 1, k]])
                            * dy2_inv
                        + (light_field[[i, j, k + 1]]
                            + LAPLACIAN_CENTER_COEFF * center_val
                            + light_field[[i, j, k - 1]])
                            * dz2_inv;

                    // Update using diffusion equation: ∂φ/∂t = D∇²φ - μₐφ + S
                    let update = center_val
                        + dt * (diffusion_coefficient * laplacian_phi
                            - absorption_coeff * center_val
                            + source_term);

                    // Ensure non-negative values (physical constraint)
                    updated_field[[i, j, k]] = update.max(0.0);
                }
            }
        }

        // Update the light field
        light_field.assign(&updated_field);

        // Update fluence_rate to match
        self.fluence_rate.assign(fields);

        self.update_time = start_time.elapsed().as_secs_f64();
        self.call_count += 1;
    }

    fn emission_spectrum(&self) -> &Array3<f64> {
        &self.emission_spectrum
    }

    fn fluence_rate(&self) -> &Array4<f64> {
        &self.fluence_rate
    }

    fn report_performance(&self) {
        debug!(
            "LightDiffusion performance: update_time={:.6e}s, fft_time={:.6e}s, diffusion_time={:.6e}s, effect_time={:.6e}s, calls={}",
            self.update_time, self.fft_time, self.diffusion_time, self.effect_time, self.call_count
        );
    }
}
