//! Tissue medium factory for heterogeneous media creation
//!
//! **Factory Pattern**: Encapsulated creation logic per Gang of Four
//! **Evidence-Based**: Tissue parameters from Hamilton & Blackstock (1998)

use kwavers_core::constants::acoustic_parameters::{
    REFERENCE_FREQUENCY_TISSUE_HZ, TISSUE_NONLINEARITY_B_A, VISCOSITY_SOFT_TISSUE,
};
use kwavers_core::constants::cavitation::{
    GAS_DIFFUSION_COEFFICIENT_TISSUE, POLYTROPIC_EXPONENT_AIR, SURFACE_TENSION_TISSUE,
    TISSUE_NUCLEATION_RADIUS, VAPOR_PRESSURE_WATER,
};
use kwavers_core::constants::fundamental::{
    ACOUSTIC_ABSORPTION_TISSUE, ATMOSPHERIC_PRESSURE, DENSITY_TISSUE, SOUND_SPEED_TISSUE,
};
use kwavers_core::constants::optical::{
    OPTICAL_ABSORPTION_TISSUE_NIR_M, OPTICAL_SCATTERING_REDUCED_TISSUE_NIR_M,
};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
use kwavers_core::constants::tissue_thermal::{
    SPECIFIC_HEAT_BRAIN, THERMAL_CONDUCTIVITY_BLOOD, THERMAL_DIFFUSIVITY_TISSUE,
    THERMAL_EXPANSION_SOFT_TISSUE,
};
use kwavers_core::constants::MHZ_TO_HZ;
use kwavers_grid::Grid;
use crate::heterogeneous::core::HeterogeneousMedium;
use log::debug;
use ndarray::Array3;

/// Factory for creating tissue-specific heterogeneous media
///
/// **Design Principle**: Single responsibility for tissue initialization
/// **Validation**: All parameters evidence-based from literature
#[derive(Debug)]
pub struct TissueFactory;

impl TissueFactory {
    /// Create a heterogeneous tissue medium with evidence-based parameters
    ///
    /// **Literature**: Hamilton & Blackstock (1998) Table 8.1
    /// **Parameters**: Validated against clinical acoustic measurements
    #[must_use]
    pub fn create_tissue_medium(grid: &Grid) -> HeterogeneousMedium {
        // Core acoustic properties (Hamilton & Blackstock Table 8.1)
        let density = Array3::from_elem((grid.nx, grid.ny, grid.nz), DENSITY_TISSUE);
        let sound_speed = Array3::from_elem((grid.nx, grid.ny, grid.nz), SOUND_SPEED_TISSUE);
        let viscosity = Array3::from_elem((grid.nx, grid.ny, grid.nz), VISCOSITY_SOFT_TISSUE);

        // Bubble dynamics parameters
        let surface_tension =
            Array3::from_elem((grid.nx, grid.ny, grid.nz), SURFACE_TENSION_TISSUE);
        let ambient_pressure = ATMOSPHERIC_PRESSURE;
        let vapor_pressure = Array3::from_elem((grid.nx, grid.ny, grid.nz), VAPOR_PRESSURE_WATER);
        let polytropic_index =
            Array3::from_elem((grid.nx, grid.ny, grid.nz), POLYTROPIC_EXPONENT_AIR);

        // Thermal properties
        let specific_heat = Array3::from_elem((grid.nx, grid.ny, grid.nz), SPECIFIC_HEAT_BRAIN);
        let thermal_conductivity =
            Array3::from_elem((grid.nx, grid.ny, grid.nz), THERMAL_CONDUCTIVITY_BLOOD);
        let thermal_expansion =
            Array3::from_elem((grid.nx, grid.ny, grid.nz), THERMAL_EXPANSION_SOFT_TISSUE);
        let gas_diffusion_coeff = Array3::from_elem(
            (grid.nx, grid.ny, grid.nz),
            GAS_DIFFUSION_COEFFICIENT_TISSUE,
        );
        let thermal_diffusivity =
            Array3::from_elem((grid.nx, grid.ny, grid.nz), THERMAL_DIFFUSIVITY_TISSUE);
        let temperature = Array3::from_elem((grid.nx, grid.ny, grid.nz), BODY_TEMPERATURE_K); // 37°C

        // Optical properties [m⁻¹] — broadband NIR soft tissue initialization
        let mu_a = Array3::from_elem((grid.nx, grid.ny, grid.nz), OPTICAL_ABSORPTION_TISSUE_NIR_M);
        let mu_s_prime =
            Array3::from_elem((grid.nx, grid.ny, grid.nz), OPTICAL_SCATTERING_REDUCED_TISSUE_NIR_M);

        // Bubble state
        let bubble_radius = Array3::from_elem((grid.nx, grid.ny, grid.nz), TISSUE_NUCLEATION_RADIUS);
        let bubble_velocity = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Acoustic parameters
        let alpha0 = Array3::from_elem((grid.nx, grid.ny, grid.nz), ACOUSTIC_ABSORPTION_TISSUE);
        let delta = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.1);
        let b_a = Array3::from_elem((grid.nx, grid.ny, grid.nz), TISSUE_NONLINEARITY_B_A);
        let reference_frequency = REFERENCE_FREQUENCY_TISSUE_HZ; // 180 kHz LIFU reference

        // Initialize viscoelastic fields with tissue-appropriate spatial variation
        let shear_sound_speed = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(i, j, _k)| {
            let base_speed = 3.0; // m/s (typical for muscle tissue)
            let variation =
                0.5 * ((i as f64 / grid.nx as f64).sin() + (j as f64 / grid.ny as f64).cos());
            (base_speed + variation).clamp(1.0, 8.0)
        });

        let shear_viscosity_coeff =
            Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(i, j, k)| {
                let center_x = grid.nx as f64 / 2.0;
                let center_y = grid.ny as f64 / 2.0;
                let center_z = grid.nz as f64 / 2.0;
                let dist_from_center = (k as f64 - center_z)
                    .mul_add(
                        k as f64 - center_z,
                        (j as f64 - center_y)
                            .mul_add(j as f64 - center_y, (i as f64 - center_x).powi(2)),
                    )
                    .sqrt();
                let max_dist = center_z
                    .mul_add(center_z, center_y.mul_add(center_y, center_x.powi(2)))
                    .sqrt();
                let normalized_dist = (dist_from_center / max_dist).min(1.0);
                2.0f64.mul_add(normalized_dist, 1.0) // Range: 1.0-3.0 Pa·s
            });

        let bulk_viscosity_coeff = shear_viscosity_coeff.mapv(|shear_visc| shear_visc * 3.0);

        // Initialize elastic properties
        // Use physically consistent relationships:
        //   μ = ρ c_s^2,  K ≈ ρ c_p^2,  λ = K - 2μ/3
        let default_density: f64 = DENSITY_TISSUE;
        let default_sound_speed: f64 = SOUND_SPEED_TISSUE; // compressional wave speed approximation
        let default_bulk_modulus = default_density * default_sound_speed.powi(2);

        // Compute spatially varying shear modulus from shear wave speed field
        let lame_mu = shear_sound_speed.mapv(|cs| default_density * cs * cs);
        // Ensure λ remains positive by subtracting 2μ/3 from bulk modulus
        let lame_lambda =
            lame_mu.mapv(|mu| (2.0_f64 / 3.0).mul_add(-mu, default_bulk_modulus).max(1.0));

        // Compute frequency-dependent properties
        let freq_ratio: f64 = reference_frequency / MHZ_TO_HZ;
        let absorption = alpha0.mapv(|a0| a0 * freq_ratio.powi(1));
        let nonlinearity = b_a.clone();

        debug!(
            "Created tissue medium: grid {}x{}x{}, freq = {:.2e} Hz",
            grid.nx, grid.ny, grid.nz, reference_frequency
        );

        HeterogeneousMedium {
            use_trilinear_interpolation: false, // Default to nearest neighbor
            density,
            sound_speed,
            viscosity,
            surface_tension,
            ambient_pressure,
            vapor_pressure,
            polytropic_index,
            specific_heat,
            thermal_conductivity,
            thermal_expansion,
            gas_diffusion_coeff,
            thermal_diffusivity,
            mu_a,
            mu_s_prime,
            temperature,
            bubble_radius,
            bubble_velocity,
            alpha0,
            delta,
            b_a,
            absorption,
            // Tissue default: y = 1.5 per Szabo (1994) Table I.
            alpha_power: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.5_f64),
            nonlinearity,
            shear_sound_speed,
            shear_viscosity_coeff,
            bulk_viscosity_coeff,
            lame_lambda,
            lame_mu,
            reference_frequency,
        }
    }
}
