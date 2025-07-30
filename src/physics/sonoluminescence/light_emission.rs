//! Light emission models for sonoluminescence
//! 
//! Implements various light emission mechanisms:
//! - Blackbody radiation
//! - Bremsstrahlung (free-free transitions)
//! - Molecular emission lines
//! - Continuum emission

use ndarray::{Array1, Array3};
use std::f64::consts::PI;
use super::bubble_dynamics::BubbleState;

/// Physical constants
pub const PLANCK_CONSTANT: f64 = 6.62607015e-34; // J·s
pub const SPEED_OF_LIGHT: f64 = 2.99792458e8; // m/s
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23; // J/K
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8; // W/(m²·K⁴)
pub const ELECTRON_CHARGE: f64 = 1.602176634e-19; // C
pub const ELECTRON_MASS: f64 = 9.1093837015e-31; // kg
pub const AVOGADRO: f64 = 6.02214076e23; // mol⁻¹

/// Wavelength range for spectral calculations
#[derive(Debug, Clone)]
pub struct SpectralRange {
    pub lambda_min: f64,  // Minimum wavelength [m]
    pub lambda_max: f64,  // Maximum wavelength [m]
    pub n_points: usize,  // Number of spectral points
}

impl Default for SpectralRange {
    fn default() -> Self {
        Self {
            lambda_min: 200e-9,  // 200 nm (UV)
            lambda_max: 800e-9,  // 800 nm (near IR)
            n_points: 300,
        }
    }
}

/// Light emission model trait
pub trait LightEmissionModel {
    /// Calculate spectral radiance at given wavelength and temperature
    fn spectral_radiance(&self, lambda: f64, temperature: f64) -> f64;
    
    /// Calculate total radiated power
    fn total_power(&self, temperature: f64, emitting_volume: f64) -> f64;
    
    /// Get emission spectrum
    fn emission_spectrum(&self, temperature: f64, range: &SpectralRange) -> Array1<f64>;
}

/// Blackbody radiation model
#[derive(Debug, Clone)]
pub struct BlackbodyEmission {
    pub emissivity: f64,  // Emissivity factor (0-1)
}

impl Default for BlackbodyEmission {
    fn default() -> Self {
        Self {
            emissivity: 0.1,  // Partial emissivity for bubble interior
        }
    }
}

impl LightEmissionModel for BlackbodyEmission {
    fn spectral_radiance(&self, lambda: f64, temperature: f64) -> f64 {
        if temperature <= 0.0 || lambda <= 0.0 {
            return 0.0;
        }
        
        let hc_over_lambda = PLANCK_CONSTANT * SPEED_OF_LIGHT / lambda;
        let hc_over_lambda_kt = hc_over_lambda / (BOLTZMANN_CONSTANT * temperature);
        
        // Planck's law
        let radiance = if hc_over_lambda_kt > 50.0 {
            // Wien approximation for large x
            2.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT.powi(2) / lambda.powi(5) 
                * (-hc_over_lambda_kt).exp()
        } else {
            // Full Planck formula
            2.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT.powi(2) / lambda.powi(5) 
                / (hc_over_lambda_kt.exp() - 1.0)
        };
        
        self.emissivity * radiance
    }
    
    fn total_power(&self, temperature: f64, emitting_volume: f64) -> f64 {
        // Stefan-Boltzmann law for volume emission
        // Assuming optically thin plasma
        let power_density = self.emissivity * STEFAN_BOLTZMANN * temperature.powi(4);
        power_density * emitting_volume
    }
    
    fn emission_spectrum(&self, temperature: f64, range: &SpectralRange) -> Array1<f64> {
        let mut spectrum = Array1::zeros(range.n_points);
        let dlambda = (range.lambda_max - range.lambda_min) / (range.n_points - 1) as f64;
        
        for i in 0..range.n_points {
            let lambda = range.lambda_min + i as f64 * dlambda;
            spectrum[i] = self.spectral_radiance(lambda, temperature);
        }
        
        spectrum
    }
}

/// Bremsstrahlung (free-free) emission model
#[derive(Debug, Clone)]
pub struct BremsstrahlingEmission {
    pub ion_density: f64,      // Ion number density [m⁻³]
    pub electron_density: f64,  // Electron number density [m⁻³]
    pub ion_charge: f64,        // Average ion charge
    pub gaunt_factor: f64,      // Gaunt factor (quantum correction)
}

impl Default for BremsstrahlingEmission {
    fn default() -> Self {
        Self {
            ion_density: 1e25,       // Typical for weakly ionized plasma
            electron_density: 1e25,   // Quasi-neutral
            ion_charge: 1.0,         // Singly ionized
            gaunt_factor: 1.2,       // Typical value
        }
    }
}

impl LightEmissionModel for BremsstrahlingEmission {
    fn spectral_radiance(&self, lambda: f64, temperature: f64) -> f64 {
        if temperature <= 0.0 || lambda <= 0.0 {
            return 0.0;
        }
        
        let frequency = SPEED_OF_LIGHT / lambda;
        let h_nu = PLANCK_CONSTANT * frequency;
        
        // Bremsstrahlung emission coefficient
        let prefactor = 32.0 * PI * ELECTRON_CHARGE.powi(6) 
            / (3.0 * ELECTRON_MASS * SPEED_OF_LIGHT.powi(3))
            * (2.0 * PI / (3.0 * BOLTZMANN_CONSTANT * ELECTRON_MASS)).sqrt();
        
        let z_squared = self.ion_charge.powi(2);
        let exp_factor = (-h_nu / (BOLTZMANN_CONSTANT * temperature)).exp();
        
        let emission_coeff = prefactor * self.gaunt_factor * z_squared 
            * self.ion_density * self.electron_density 
            * temperature.powf(-0.5) * exp_factor;
        
        // Convert to spectral radiance
        emission_coeff * SPEED_OF_LIGHT / lambda.powi(2)
    }
    
    fn total_power(&self, temperature: f64, emitting_volume: f64) -> f64 {
        // Total bremsstrahlung power (integrated over all frequencies)
        let prefactor = 32.0 * PI * ELECTRON_CHARGE.powi(6) 
            / (3.0 * ELECTRON_MASS * SPEED_OF_LIGHT.powi(3) * PLANCK_CONSTANT)
            * (2.0 * PI * BOLTZMANN_CONSTANT / (3.0 * ELECTRON_MASS)).sqrt();
        
        let z_squared = self.ion_charge.powi(2);
        let power_density = prefactor * self.gaunt_factor * z_squared 
            * self.ion_density * self.electron_density * temperature.sqrt();
        
        power_density * emitting_volume
    }
    
    fn emission_spectrum(&self, temperature: f64, range: &SpectralRange) -> Array1<f64> {
        let mut spectrum = Array1::zeros(range.n_points);
        let dlambda = (range.lambda_max - range.lambda_min) / (range.n_points - 1) as f64;
        
        for i in 0..range.n_points {
            let lambda = range.lambda_min + i as f64 * dlambda;
            spectrum[i] = self.spectral_radiance(lambda, temperature);
        }
        
        spectrum
    }
}

/// Combined emission model for sonoluminescence
#[derive(Debug, Clone)]
pub struct SonoluminescenceEmission {
    pub blackbody: BlackbodyEmission,
    pub bremsstrahlung: BremsstrahlingEmission,
    pub use_blackbody: bool,
    pub use_bremsstrahlung: bool,
    pub opacity_correction: f64,  // Optical depth correction factor
}

impl Default for SonoluminescenceEmission {
    fn default() -> Self {
        Self {
            blackbody: BlackbodyEmission::default(),
            bremsstrahlung: BremsstrahlingEmission::default(),
            use_blackbody: true,
            use_bremsstrahlung: true,
            opacity_correction: 1.0,  // Optically thin
        }
    }
}

impl SonoluminescenceEmission {
    /// Calculate emission from a bubble based on its state
    pub fn calculate_bubble_emission(
        &self,
        state: &BubbleState,
        spectral_range: &SpectralRange,
    ) -> (f64, Array1<f64>) {
        let temperature = state.temperature;
        let radius = state.radius;
        let volume = 4.0 / 3.0 * PI * radius.powi(3);
        
        // Calculate emission spectrum
        let mut spectrum = Array1::zeros(spectral_range.n_points);
        
        if self.use_blackbody {
            let bb_spectrum = self.blackbody.emission_spectrum(temperature, spectral_range);
            spectrum = spectrum + bb_spectrum;
        }
        
        if self.use_bremsstrahlung && temperature > 5000.0 {
            // Only significant at high temperatures
            let br_spectrum = self.bremsstrahlung.emission_spectrum(temperature, spectral_range);
            spectrum = spectrum + br_spectrum;
        }
        
        // Apply opacity correction (for optically thick conditions)
        spectrum *= self.opacity_correction;
        
        // Calculate total power
        let total_power = if self.use_blackbody && self.use_bremsstrahlung {
            self.blackbody.total_power(temperature, volume) 
                + self.bremsstrahlung.total_power(temperature, volume)
        } else if self.use_blackbody {
            self.blackbody.total_power(temperature, volume)
        } else if self.use_bremsstrahlung {
            self.bremsstrahlung.total_power(temperature, volume)
        } else {
            0.0
        };
        
        (total_power * self.opacity_correction, spectrum)
    }
    
    /// Update plasma parameters based on temperature and ionization
    pub fn update_plasma_parameters(&mut self, temperature: f64, pressure: f64) {
        // Simple Saha equation for ionization fraction
        if temperature > 5000.0 {
            let ionization_energy = 13.6 * ELECTRON_CHARGE; // Hydrogen
            let saha_factor = 2.4e21 * temperature.powf(1.5) 
                * (-ionization_energy / (BOLTZMANN_CONSTANT * temperature)).exp();
            
            let n_total = pressure / (BOLTZMANN_CONSTANT * temperature);
            let ionization_fraction = (-1.0 + (1.0 + 4.0 * saha_factor / n_total).sqrt()) 
                / (2.0 * saha_factor / n_total);
            
            self.bremsstrahlung.electron_density = ionization_fraction * n_total;
            self.bremsstrahlung.ion_density = self.bremsstrahlung.electron_density;
        }
    }
}

/// Main light emission interface
pub struct LightEmission {
    model: SonoluminescenceEmission,
    spectral_range: SpectralRange,
    emission_field: Array3<f64>,
    spectral_field: Array3<Array1<f64>>,
}

impl LightEmission {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let model = SonoluminescenceEmission::default();
        let spectral_range = SpectralRange::default();
        let emission_field = Array3::zeros((nx, ny, nz));
        
        // Initialize spectral field
        let empty_spectrum = Array1::zeros(spectral_range.n_points);
        let spectral_field = Array3::from_elem((nx, ny, nz), empty_spectrum);
        
        Self {
            model,
            spectral_range,
            emission_field,
            spectral_field,
        }
    }
    
    /// Calculate light emission from bubble states
    pub fn calculate_emission(&mut self, bubble_states: &Array3<BubbleState>) {
        let shape = bubble_states.shape();
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let state = &bubble_states[[i, j, k]];
                    
                    // Update plasma parameters
                    self.model.update_plasma_parameters(
                        state.temperature, 
                        state.pressure_internal
                    );
                    
                    // Calculate emission
                    let (power, spectrum) = self.model.calculate_bubble_emission(
                        state, 
                        &self.spectral_range
                    );
                    
                    self.emission_field[[i, j, k]] = power;
                    self.spectral_field[[i, j, k]] = spectrum;
                }
            }
        }
    }
    
    /// Get total emission field
    pub fn get_emission_field(&self) -> &Array3<f64> {
        &self.emission_field
    }
    
    /// Get spectral emission at a point
    pub fn get_spectrum(&self, i: usize, j: usize, k: usize) -> &Array1<f64> {
        &self.spectral_field[[i, j, k]]
    }
    
    /// Get wavelength array
    pub fn get_wavelengths(&self) -> Array1<f64> {
        let mut wavelengths = Array1::zeros(self.spectral_range.n_points);
        let dlambda = (self.spectral_range.lambda_max - self.spectral_range.lambda_min) 
            / (self.spectral_range.n_points - 1) as f64;
        
        for i in 0..self.spectral_range.n_points {
            wavelengths[i] = self.spectral_range.lambda_min + i as f64 * dlambda;
        }
        
        wavelengths
    }
}