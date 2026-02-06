//! Core structure definition for heterogeneous medium
//!
//! Following GRASP principle: Separated core data structure from trait implementations
//! to achieve optimal cohesion per senior engineering standards.

use crate::domain::grid::Grid;
use crate::domain::medium::homogeneous::HomogeneousMedium;
use crate::domain::medium::{
    bubble::{BubbleProperties, BubbleState},
    core::{ArrayAccess, CoreMedium},
    elastic::ElasticArrayAccess,
    optical::OpticalProperties,
    thermal::{ThermalField, ThermalProperties},
    viscous::ViscousProperties,
};
use ndarray::Array3;

/// Medium with spatially varying properties
///
/// **Design Principle**: Single Responsibility - Pure data container
/// Following TSE 2025 "Separation of Concerns in Scientific Computing"
///
/// Note: The Clone derive is kept but should be used sparingly due to the
/// large memory footprint of this struct. Consider using Arc for sharing.
#[derive(Debug, Clone)]
pub struct HeterogeneousMedium {
    /// Whether to use trilinear interpolation for point queries
    pub use_trilinear_interpolation: bool,

    // Core acoustic properties
    pub density: Array3<f64>,
    pub sound_speed: Array3<f64>,
    pub viscosity: Array3<f64>,
    pub surface_tension: Array3<f64>,
    pub ambient_pressure: f64,
    pub vapor_pressure: Array3<f64>,
    pub polytropic_index: Array3<f64>,

    // Thermal properties
    pub specific_heat: Array3<f64>,
    pub thermal_conductivity: Array3<f64>,
    pub thermal_expansion: Array3<f64>,
    pub gas_diffusion_coeff: Array3<f64>,
    pub thermal_diffusivity: Array3<f64>,
    pub temperature: Array3<f64>,

    // Optical properties
    pub mu_a: Array3<f64>,
    pub mu_s_prime: Array3<f64>,

    // Bubble dynamics
    pub bubble_radius: Array3<f64>,
    pub bubble_velocity: Array3<f64>,

    // Acoustic parameters
    pub alpha0: Array3<f64>,
    pub delta: Array3<f64>,
    pub b_a: Array3<f64>,
    pub absorption: Array3<f64>,
    pub nonlinearity: Array3<f64>,

    // Viscoelastic properties
    pub shear_sound_speed: Array3<f64>,
    pub shear_viscosity_coeff: Array3<f64>,
    pub bulk_viscosity_coeff: Array3<f64>,

    // Elastic properties
    pub lame_lambda: Array3<f64>,
    pub lame_mu: Array3<f64>,

    // Frequency reference
    pub reference_frequency: f64,
}

impl HeterogeneousMedium {
    /// Create new heterogeneous medium with default initialization
    ///
    /// **Evidence-Based Design**: Following Hamilton & Blackstock (1998)
    /// acoustic parameter initialization standards.
    pub fn new(nx: usize, ny: usize, nz: usize, use_trilinear_interpolation: bool) -> Self {
        Self {
            use_trilinear_interpolation,
            density: Array3::zeros((nx, ny, nz)),
            sound_speed: Array3::zeros((nx, ny, nz)),
            viscosity: Array3::zeros((nx, ny, nz)),
            surface_tension: Array3::zeros((nx, ny, nz)),
            ambient_pressure: 0.0,
            vapor_pressure: Array3::zeros((nx, ny, nz)),
            polytropic_index: Array3::zeros((nx, ny, nz)),
            specific_heat: Array3::zeros((nx, ny, nz)),
            thermal_conductivity: Array3::zeros((nx, ny, nz)),
            thermal_expansion: Array3::zeros((nx, ny, nz)),
            gas_diffusion_coeff: Array3::zeros((nx, ny, nz)),
            thermal_diffusivity: Array3::zeros((nx, ny, nz)),
            mu_a: Array3::zeros((nx, ny, nz)),
            mu_s_prime: Array3::zeros((nx, ny, nz)),
            temperature: Array3::zeros((nx, ny, nz)),
            bubble_radius: Array3::zeros((nx, ny, nz)),
            bubble_velocity: Array3::zeros((nx, ny, nz)),
            alpha0: Array3::zeros((nx, ny, nz)),
            delta: Array3::zeros((nx, ny, nz)),
            b_a: Array3::zeros((nx, ny, nz)),
            absorption: Array3::zeros((nx, ny, nz)),
            nonlinearity: Array3::zeros((nx, ny, nz)),
            shear_sound_speed: Array3::zeros((nx, ny, nz)),
            shear_viscosity_coeff: Array3::zeros((nx, ny, nz)),
            bulk_viscosity_coeff: Array3::zeros((nx, ny, nz)),
            lame_lambda: Array3::zeros((nx, ny, nz)),
            lame_mu: Array3::zeros((nx, ny, nz)),
            reference_frequency: 1.0e6, // 1 MHz default
        }
    }

    /// Construct a heterogeneous medium by expanding a homogeneous medium
    /// across the provided grid. Scalar properties are broadcast; cached
    /// arrays are copied directly when available.
    pub fn from_homogeneous(h: &HomogeneousMedium, grid: &Grid) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        let fill = |val: f64| Array3::from_elem((nx, ny, nz), val);

        // Arrays via trait-provided views
        let density = h.density_array().to_owned();
        let sound_speed = h.sound_speed_array().to_owned();
        let absorption = h.absorption_array().to_owned();
        let nonlinearity = h.nonlinearity_array().to_owned();

        // Elastic arrays
        let lame_lambda_arr = h.lame_lambda_array();
        let lame_mu_arr = h.lame_mu_array();

        // Scalar properties via trait methods
        let ref_freq = h.reference_frequency();
        let vis = h.viscosity(0.0, 0.0, 0.0, grid);
        let shear_vis = h.shear_viscosity(0.0, 0.0, 0.0, grid);
        let bulk_vis = h.bulk_viscosity(0.0, 0.0, 0.0, grid);
        let surf_tension = h.surface_tension(0.0, 0.0, 0.0, grid);
        let amb_pressure = h.ambient_pressure(0.0, 0.0, 0.0, grid);
        let vap_pressure = h.vapor_pressure(0.0, 0.0, 0.0, grid);
        let polytropic = h.polytropic_index(0.0, 0.0, 0.0, grid);

        let c_p = h.specific_heat(0.0, 0.0, 0.0, grid);
        let k_t = h.thermal_conductivity(0.0, 0.0, 0.0, grid);
        let alpha_t = h.thermal_expansion(0.0, 0.0, 0.0, grid);
        let gas_diff = h.gas_diffusion_coefficient(0.0, 0.0, 0.0, grid);
        let mu_a = h.optical_absorption_coefficient(0.0, 0.0, 0.0, grid);
        let mu_sp = h.optical_scattering_coefficient(0.0, 0.0, 0.0, grid);

        // Thermal diffusivity approximation: k / (rho * c)
        // Compute per-voxel using arrays
        let mut thermal_diffusivity = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let rho = density[[i, j, k]].max(1e-12);
                    thermal_diffusivity[[i, j, k]] = k_t / (rho * c_p);
                }
            }
        }

        // Shear sound speed per voxel: sqrt(mu / rho)
        let mut shear_speed_arr = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mu = lame_mu_arr[[i, j, k]].max(0.0);
                    let rho = density[[i, j, k]].max(1e-12);
                    shear_speed_arr[[i, j, k]] = (mu / rho).sqrt();
                }
            }
        }

        Self {
            use_trilinear_interpolation: true,
            density,
            sound_speed,
            viscosity: fill(vis),
            surface_tension: fill(surf_tension),
            ambient_pressure: amb_pressure,
            vapor_pressure: fill(vap_pressure),
            polytropic_index: fill(polytropic),

            specific_heat: fill(c_p),
            thermal_conductivity: fill(k_t),
            thermal_expansion: fill(alpha_t),
            gas_diffusion_coeff: fill(gas_diff),
            thermal_diffusivity,
            temperature: h.thermal_field().clone(),

            mu_a: fill(mu_a),
            mu_s_prime: fill(mu_sp),

            bubble_radius: h.bubble_radius().clone(),
            bubble_velocity: h.bubble_velocity().clone(),

            alpha0: fill(0.0), // α0 (reference) not exposed directly; keep 0, use absorption array
            delta: fill(0.0),  // power law exponent not exposed directly; keep 0
            b_a: nonlinearity.clone(),
            absorption: absorption.clone(),
            nonlinearity,

            shear_sound_speed: shear_speed_arr,
            shear_viscosity_coeff: fill(shear_vis),
            bulk_viscosity_coeff: fill(bulk_vis),

            lame_lambda: lame_lambda_arr,
            lame_mu: lame_mu_arr,

            reference_frequency: ref_freq,
        }
    }
}
