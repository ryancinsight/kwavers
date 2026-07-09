//! Light diffusion solver implementation

use super::properties::DiffusionOpticalProperties;
use crate::acoustics::traits::LightDiffusionModelTrait;
use crate::optics::polarization::LinearPolarization;
use crate::optics::PolarizationModel as PolarizationModelTrait;
use crate::wave_propagation::scattering::ScatteringCalculator;
use kwavers_core::constants::fundamental::SPEED_OF_LIGHT;
use kwavers_core::constants::optical::{
    DEFAULT_POLARIZATION_FACTOR, LAPLACIAN_CENTER_COEFF, VISIBLE_LIGHT_FREQUENCY_HZ,
};
use kwavers_field::indices::LIGHT_IDX;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::{Array3, Array4};
use log::debug;
use std::time::Instant;

/// Time-domain light diffusion solver
#[derive(Debug)]
pub struct LightDiffusion {
    /// Photon fluence rate field [photons/(m²·s)]
    pub fluence_rate: Array4<f64>,
    /// Emission spectrum field [photons/(m³·s·Hz)]
    pub emission_spectrum: Array3<f64>,
    /// Physical optical properties of the medium
    optical_properties: DiffusionOpticalProperties,
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
    /// The time-dependent photon diffusion equation derived from the
    /// radiative transfer equation under the P1 (diffusion) approximation
    /// is (Wang & Wu 2007, *Biomedical Optics*, Eq. 5.7; Arridge 1999):
    ///
    /// ```text
    /// (1/c) ∂φ/∂t = ∇·(D∇φ) - μₐφ + S
    /// ```
    ///
    /// equivalently `∂φ/∂t = c·∇·(D∇φ) - c·μₐ·φ + c·S`, where `c = c₀/n` is
    /// the speed of light in the medium with refractive index `n`. The
    /// factor of `c` is required for dimensional consistency: with
    /// `D` in metres and `μₐ` in m⁻¹, `D∇²φ` and `μₐφ` both have units of
    /// `[φ]/m`; multiplication by `c` (m/s) recovers `[φ]/s = ∂φ/∂t`.
    ///
    /// where:
    /// - φ is the photon fluence rate [photons/(m²·s)]
    /// - D = 1/(3(μₐ + μₛ')) is the diffusion coefficient (m)
    /// - μₐ is the absorption coefficient [m⁻¹]
    /// - μₛ' is the reduced scattering coefficient [m⁻¹]
    /// - S is the source term [photons/(m³·s)]
    ///
    /// The diffusion approximation is valid when μₛ' ≫ μₐ (scattering dominates).
    pub fn new(
        grid: &Grid,
        optical_properties: DiffusionOpticalProperties,
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
            fluence_rate: Array4::zeros([1, nx, ny, nz]),
            emission_spectrum: Array3::zeros([nx, ny, nz]),
            optical_properties,
            _polarization: if enable_polarization {
                Some(Box::new(LinearPolarization::new(
                    DEFAULT_POLARIZATION_FACTOR,
                )))
            } else {
                None
            },
            _scattering: if enable_scattering {
                // Default optical frequency for scattering (~600 nm green-yellow visible light)
                let frequency = VISIBLE_LIGHT_FREQUENCY_HZ;
                Some(ScatteringCalculator::new(frequency, SPEED_OF_LIGHT))
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
        let mut light_field = fields.index_axis_mut(0, LIGHT_IDX).unwrap();

        // Get dimensions
        let [nx, ny, nz] = light_field.shape();

        // Time-dependent photon diffusion equation under the P1 approximation:
        //
        //   (1/c) ∂φ/∂t = ∇·(D∇φ) − μₐ φ + S
        //
        // equivalently ∂φ/∂t = c (D ∇²φ − μₐ φ + S), where c = c₀/n is the
        // speed of light in the medium (n = refractive index of the tissue).
        // Without the factor of c, the RHS has units [φ]/m, not [φ]/s
        // (D is in m, μₐ in m⁻¹, ∇² in m⁻²), so the absolute time scale would
        // be off by a factor of c — a multi-orders-of-magnitude error.
        let diffusion_coefficient = self.optical_properties.diffusion_coefficient();
        let absorption_coeff = self.optical_properties.absorption_coefficient;
        let c_medium = SPEED_OF_LIGHT / self.optical_properties.refractive_index;

        // Create a temporary array to store the updated values
        let mut updated_field = light_field.to_contiguous();

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

                    let laplacian_phi = (LAPLACIAN_CENTER_COEFF
                        .mul_add(center_val, light_field[[i, j, k + 1]])
                        + light_field[[i, j, k - 1]])
                    .mul_add(
                        dz2_inv,
                        (LAPLACIAN_CENTER_COEFF.mul_add(center_val, light_field[[i + 1, j, k]])
                            + light_field[[i - 1, j, k]])
                        .mul_add(
                            dx2_inv,
                            (LAPLACIAN_CENTER_COEFF
                                .mul_add(center_val, light_field[[i, j + 1, k]])
                                + light_field[[i, j - 1, k]])
                                * dy2_inv,
                        ),
                    );

                    // Forward-Euler update of ∂φ/∂t = c·(D∇²φ − μₐφ + S).
                    // The c factor (speed of light in medium) converts the
                    // RHS from [φ]/m to [φ]/s for consistent integration in
                    // physical time.
                    let update = dt.mul_add(
                        c_medium
                            * (diffusion_coefficient
                                .mul_add(laplacian_phi, -(absorption_coeff * center_val))
                                + source_term),
                        center_val,
                    );

                    // Ensure non-negative values (physical constraint)
                    updated_field[[i, j, k]] = update.max(0.0);
                }
            }
        }

        // Update the light field
        light_field.assign(&updated_field);

        // Mirror the freshly updated light channel into the public
        // `fluence_rate` snapshot exposed via `LightDiffusionModelTrait`.
        // The snapshot is sized (1, nx, ny, nz); copying the full
        // multi-channel `fields` array would broadcast-fail at runtime.
        self.fluence_rate
            .index_axis_mut(0, 0)
            .unwrap()
            .assign(&light_field.to_contiguous());

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
