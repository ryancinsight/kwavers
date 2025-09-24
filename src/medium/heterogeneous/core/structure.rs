//! Core structure definition for heterogeneous medium
//!
//! Following GRASP principle: Separated core data structure from trait implementations
//! to achieve optimal cohesion per senior engineering standards.

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
    pub fn new(
        nx: usize, 
        ny: usize, 
        nz: usize,
        use_trilinear_interpolation: bool,
    ) -> Self {
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
}