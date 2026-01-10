// physics/optics/diffusion/mod.rs
use crate::domain::field::indices::LIGHT_IDX;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::physics::optics::polarization::LinearPolarization;
use crate::physics::optics::PolarizationModel as PolarizationModelTrait;
use crate::physics::thermal::PennesSolver;
use crate::physics::wave_propagation::scattering::ScatteringCalculator;
use log::debug;
use ndarray::{Array3, Array4, Axis};

use crate::core::constants::optical::{DEFAULT_POLARIZATION_FACTOR, LAPLACIAN_CENTER_COEFF};
use crate::physics::traits::LightDiffusionModelTrait;

/// Physical optical properties of a medium for photon diffusion calculations
#[derive(Debug, Clone, Copy)]
pub struct OpticalProperties {
    /// Absorption coefficient μₐ [m⁻¹]
    pub absorption_coefficient: f64,
    /// Reduced scattering coefficient μₛ' [m⁻¹]
    /// The reduced scattering coefficient accounts for anisotropic scattering
    /// μₛ' = μₛ(1-g) where g is the anisotropy factor
    pub reduced_scattering_coefficient: f64,
    /// Refractive index n (dimensionless)
    pub refractive_index: f64,
}

impl OpticalProperties {
    /// Create optical properties for a typical biological tissue
    #[must_use]
    pub fn biological_tissue() -> Self {
        Self {
            absorption_coefficient: 10.0, // 10 cm⁻¹ = 100 m⁻¹ (typical NIR tissue absorption)
            reduced_scattering_coefficient: 1000.0, // 1000 cm⁻¹ = 10000 m⁻¹ (typical tissue scattering)
            refractive_index: 1.4,                  // Typical tissue refractive index
        }
    }

    /// Create optical properties for water
    #[must_use]
    pub fn water() -> Self {
        Self {
            absorption_coefficient: 0.1,          // Very low absorption in NIR
            reduced_scattering_coefficient: 0.01, // Low scattering in pure water
            refractive_index: 1.33,
        }
    }

    /// Calculate diffusion coefficient from optical properties
    ///
    /// In the diffusion approximation: D = 1/(3(μₐ + μₛ'))
    /// where μₐ is absorption coefficient, μₛ' is reduced scattering coefficient
    #[must_use]
    pub fn diffusion_coefficient(&self) -> f64 {
        1.0 / (3.0 * (self.absorption_coefficient + self.reduced_scattering_coefficient))
    }

    /// Calculate the transport coefficient μ_tr = μₐ + μₛ'
    #[must_use]
    pub fn transport_coefficient(&self) -> f64 {
        self.absorption_coefficient + self.reduced_scattering_coefficient
    }

    /// Calculate albedo ω = μₛ' / μ_tr (single scattering albedo)
    #[must_use]
    pub fn single_scatter_albedo(&self) -> f64 {
        let mu_tr = self.transport_coefficient();
        if mu_tr > 0.0 {
            self.reduced_scattering_coefficient / mu_tr
        } else {
            0.0
        }
    }

    /// Check validity of diffusion approximation
    ///
    /// The diffusion approximation is valid when:
    /// 1. Reduced scattering dominates absorption: μₛ' ≫ μₐ
    /// 2. Optical depth is large enough for diffusion to develop
    #[must_use]
    pub fn diffusion_approximation_valid(&self) -> bool {
        // Require scattering to be at least 10x absorption for good diffusion approximation
        self.reduced_scattering_coefficient >= 10.0 * self.absorption_coefficient
    }
}
use std::time::Instant;

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
    /// Thermal solver (optional)
    _thermal: Option<PennesSolver>,
    /// Feature flags
    _enable_polarization: bool,
    _enable_scattering: bool,
    _enable_thermal: bool,
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
    /// * `enable_thermal` - Enable thermal effects
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
        enable_thermal: bool,
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
            _thermal: if enable_thermal {
                use crate::physics::thermal::ThermalProperties;
                let properties = ThermalProperties {
                    k: 0.5,
                    rho: 1050.0,
                    c: 3600.0,
                    w_b: 0.5e-3,
                    c_b: 3800.0,
                    t_a: 37.0,
                    q_m: 420.0,
                };
                PennesSolver::new(
                    grid.nx, grid.ny, grid.nz, grid.dx, grid.dy, grid.dz, 0.001, properties,
                )
                .ok()
            } else {
                None
            },
            _enable_polarization: enable_polarization,
            _enable_scattering: enable_scattering,
            _enable_thermal: enable_thermal,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;

    #[test]
    fn test_optical_properties_diffusion_coefficient() {
        // Test biological tissue properties
        let tissue_props = OpticalProperties::biological_tissue();

        // Biological tissue should have high scattering, low absorption
        assert!(tissue_props.reduced_scattering_coefficient > tissue_props.absorption_coefficient);

        // Calculate diffusion coefficient: D = 1/(3(μₐ + μₛ'))
        let expected_d = 1.0
            / (3.0
                * (tissue_props.absorption_coefficient
                    + tissue_props.reduced_scattering_coefficient));
        let calculated_d = tissue_props.diffusion_coefficient();

        approx::assert_relative_eq!(calculated_d, expected_d, epsilon = 1e-10);

        // Diffusion coefficient should be small (typical tissue D ~ 0.0002 m²/s)
        assert!(calculated_d > 0.0 && calculated_d < 0.001);
    }

    #[test]
    fn test_optical_properties_transport_coefficient() {
        let props = OpticalProperties {
            absorption_coefficient: 10.0,
            reduced_scattering_coefficient: 100.0,
            refractive_index: 1.4,
        };

        let transport_coeff = props.transport_coefficient();
        assert_eq!(transport_coeff, 110.0); // μₐ + μₛ'

        let albedo = props.single_scatter_albedo();
        assert_eq!(albedo, 100.0 / 110.0); // μₛ' / μ_tr
    }

    #[test]
    fn test_diffusion_approximation_validity() {
        // Valid case: scattering dominates
        let valid_props = OpticalProperties {
            absorption_coefficient: 10.0,
            reduced_scattering_coefficient: 100.0, // 10x absorption
            refractive_index: 1.4,
        };
        assert!(valid_props.diffusion_approximation_valid());

        // Invalid case: absorption dominates
        let invalid_props = OpticalProperties {
            absorption_coefficient: 100.0,
            reduced_scattering_coefficient: 10.0, // 0.1x absorption
            refractive_index: 1.4,
        };
        assert!(!invalid_props.diffusion_approximation_valid());
    }

    #[test]
    fn test_light_diffusion_creation() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let optical_props = OpticalProperties::biological_tissue();

        let diffusion = LightDiffusion::new(
            &grid,
            optical_props,
            false, // no polarization
            false, // no scattering
            false, // no thermal
        );

        // Check that fluence rate has correct dimensions
        assert_eq!(diffusion.fluence_rate.shape(), &[1, 10, 10, 10]);

        // Check that emission spectrum has correct dimensions
        assert_eq!(diffusion.emission_spectrum.shape(), &[10, 10, 10]);
    }
}
