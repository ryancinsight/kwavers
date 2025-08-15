// physics/scattering/optic/rayleigh.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};

/// Rayleigh optical scattering model for light propagation in tissue
/// 
/// This model implements Rayleigh scattering for small particles (λ >> particle size)
/// commonly found in biological tissues such as mitochondria, lipid droplets, etc.
#[derive(Debug, Clone)]
pub struct RayleighOpticalScatteringModel {
    /// Rayleigh scattering coefficient scaling factor
    rayleigh_scaling: f64,
    /// Reference wavelength in meters (typically 650 nm for red light)
    reference_wavelength: f64,
    /// Enable wavelength-dependent scattering (λ^-4 dependence)
    wavelength_dependent: bool,
    /// Particle density factor for tissue-specific scattering
    particle_density_factor: f64,
}

impl Default for RayleighOpticalScatteringModel {
    fn default() -> Self {
        Self::new()
    }
}

impl RayleighOpticalScatteringModel {
    /// Create a new Rayleigh optical scattering model with default parameters
    pub fn new() -> Self {
        debug!("Initializing RayleighOpticalScatteringModel with tissue-realistic parameters");
        Self {
            rayleigh_scaling: 1.0,
            reference_wavelength: 650e-9, // 650 nm (red light)
            wavelength_dependent: true,
            particle_density_factor: 1.0,
        }
    }
    
    /// Create with custom parameters for specific tissues
    pub fn with_parameters(
        rayleigh_scaling: f64,
        reference_wavelength: f64,
        wavelength_dependent: bool,
        particle_density_factor: f64,
    ) -> Self {
        debug!("Initializing custom RayleighOpticalScatteringModel");
        Self {
            rayleigh_scaling,
            reference_wavelength,
            wavelength_dependent,
            particle_density_factor,
        }
    }
    
    /// Calculate wavelength-dependent Rayleigh scattering coefficient
    /// Based on the λ^-4 dependence characteristic of Rayleigh scattering
    fn calculate_rayleigh_coefficient(&self, wavelength: f64, tissue_mu_s_prime: f64) -> f64 {
        if self.wavelength_dependent {
            // Rayleigh scattering: μs_rayleigh ∝ λ^-4
            let wavelength_ratio = self.reference_wavelength / wavelength;
            let rayleigh_contribution = wavelength_ratio.powi(4);
            
            // Tissue-specific Rayleigh scattering component
            // Typically 5-15% of total reduced scattering in soft tissue
            let rayleigh_fraction = 0.10; // 10% of total scattering from Rayleigh component
            
            tissue_mu_s_prime * rayleigh_fraction * rayleigh_contribution * self.rayleigh_scaling
        } else {
            // Simplified constant coefficient
            tissue_mu_s_prime * 0.10 * self.rayleigh_scaling
        }
    }
    
    /// Calculate anisotropy parameter for Rayleigh scattering
    /// Rayleigh scattering is nearly isotropic (g ≈ 0)
    fn rayleigh_anisotropy(&self) -> f64 {
        0.02 // Very small forward bias for Rayleigh scattering
    }
    
    /// Apply phase function correction for anisotropic scattering
    fn apply_phase_function_correction(&self, fluence: f64, scattering_coefficient: f64) -> f64 {
        let g = self.rayleigh_anisotropy();
        let transport_coefficient = scattering_coefficient * (1.0 - g);
        
        // Modified Beer-Lambert law with transport coefficient
        fluence * (-transport_coefficient).exp()
    }
}

impl super::OpticalScatteringModel for RayleighOpticalScatteringModel {
    fn apply_scattering(&mut self, fluence: &mut Array3<f64>, grid: &Grid, medium: &dyn Medium) {
        debug!("Applying physics-based Rayleigh optical scattering");
        
        // Assume red light wavelength for tissue applications
        let wavelength = 650e-9; // 650 nm
        
        Zip::indexed(fluence).for_each(|(i, j, k), f| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            
            // Get tissue optical properties
            // For Beer-Lambert attenuation, we need the full scattering coefficient μs, not μs'
            let tissue_mu_s_prime = medium.optical_scattering_coefficient(x, y, z, grid);
            
            // Convert reduced scattering coefficient back to full scattering coefficient
            // μs' = μs * (1 - g), so μs = μs' / (1 - g)
            // For Rayleigh scattering, anisotropy g ≈ 0.1-0.2 in biological tissues
            let rayleigh_anisotropy = self.rayleigh_anisotropy();
            let tissue_mu_s = tissue_mu_s_prime / (1.0 - rayleigh_anisotropy);
            
            // Calculate Rayleigh-specific scattering coefficient using full μs
            let rayleigh_mu_s = self.calculate_rayleigh_coefficient(wavelength, tissue_mu_s);
            
            // Apply particle density variations based on tissue type
            let density_factor = self.particle_density_factor;
            let effective_rayleigh_coefficient = rayleigh_mu_s * density_factor;
            
            // Calculate scattering path length (assuming step size is grid spacing)
            let path_length = (grid.dx.powi(2) + grid.dy.powi(2) + grid.dz.powi(2)).sqrt();
            
            // Apply Beer-Lambert attenuation: I = I₀ * exp(-μs * L)
            // This is the correct physics for primary beam attenuation
            let attenuation_factor = (-effective_rayleigh_coefficient * path_length).exp();
            let attenuated_fluence = *f * attenuation_factor;
            
            // Apply phase function correction for anisotropic scattering effects
            let corrected_fluence = self.apply_phase_function_correction(attenuated_fluence, effective_rayleigh_coefficient * path_length);
            
            // Update fluence with scattering losses
            *f = corrected_fluence.max(0.0);
            
            // Ensure numerical stability
            if f.is_nan() || f.is_infinite() {
                *f = 0.0;
            }
        });
        
        debug!("Rayleigh optical scattering applied successfully with correct Beer-Lambert physics");
    }
}