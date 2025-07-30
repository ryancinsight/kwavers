//! Bremsstrahlung radiation model for sonoluminescence
//!
//! Implements free-free emission from ionized gas in hot bubble

use ndarray::{Array1, Array3};
use std::f64::consts::PI;

/// Physical constants
pub const ELECTRON_CHARGE: f64 = 1.602176634e-19; // C
pub const ELECTRON_MASS: f64 = 9.1093837015e-31; // kg
pub const EPSILON_0: f64 = 8.8541878128e-12; // F/m
pub const PLANCK_CONSTANT: f64 = 6.62607015e-34; // J·s
pub const SPEED_OF_LIGHT: f64 = 2.99792458e8; // m/s
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23; // J/K

/// Bremsstrahlung radiation model
#[derive(Debug, Clone)]
pub struct BremsstrahlungModel {
    /// Average ion charge number
    pub z_ion: f64,
    /// Gaunt factor (quantum correction)
    pub gaunt_factor: f64,
    /// Minimum impact parameter cutoff
    pub cutoff_parameter: f64,
}

impl Default for BremsstrahlungModel {
    fn default() -> Self {
        Self {
            z_ion: 1.0, // Singly ionized
            gaunt_factor: 1.2, // Typical value
            cutoff_parameter: 1e-10, // 0.1 nm
        }
    }
}

impl BremsstrahlungModel {
    /// Calculate bremsstrahlung emission coefficient
    /// 
    /// # Arguments
    /// * `frequency` - Radiation frequency in Hz
    /// * `temperature` - Electron temperature in Kelvin
    /// * `n_electron` - Electron number density in m⁻³
    /// * `n_ion` - Ion number density in m⁻³
    /// 
    /// # Returns
    /// Emission coefficient in W/(m³·Hz·sr)
    pub fn emission_coefficient(
        &self,
        frequency: f64,
        temperature: f64,
        n_electron: f64,
        n_ion: f64,
    ) -> f64 {
        if temperature <= 0.0 || frequency <= 0.0 || n_electron <= 0.0 || n_ion <= 0.0 {
            return 0.0;
        }
        
        // Classical emission coefficient
        let prefactor = 32.0 * PI * ELECTRON_CHARGE.powi(6) 
            / (3.0 * ELECTRON_MASS * SPEED_OF_LIGHT.powi(3) * (4.0 * PI * EPSILON_0).powi(3));
        
        // Thermal velocity factor
        let thermal_factor = (2.0 * PI / (3.0 * BOLTZMANN_CONSTANT * ELECTRON_MASS * temperature)).sqrt();
        
        // Exponential cutoff
        let h_nu = PLANCK_CONSTANT * frequency;
        let exp_factor = (-h_nu / (BOLTZMANN_CONSTANT * temperature)).exp();
        
        // Total emission
        prefactor * self.z_ion.powi(2) * self.gaunt_factor * n_electron * n_ion 
            * thermal_factor * exp_factor
    }
    
    /// Calculate spectral radiance for bremsstrahlung
    /// 
    /// # Arguments
    /// * `wavelength` - Wavelength in meters
    /// * `temperature` - Temperature in Kelvin
    /// * `n_electron` - Electron density in m⁻³
    /// * `n_ion` - Ion density in m⁻³
    /// * `path_length` - Emission path length in meters
    /// 
    /// # Returns
    /// Spectral radiance in W/(m²·sr·m)
    pub fn spectral_radiance(
        &self,
        wavelength: f64,
        temperature: f64,
        n_electron: f64,
        n_ion: f64,
        path_length: f64,
    ) -> f64 {
        let frequency = SPEED_OF_LIGHT / wavelength;
        let emission_coeff = self.emission_coefficient(frequency, temperature, n_electron, n_ion);
        
        // Convert to spectral radiance
        emission_coeff * path_length * SPEED_OF_LIGHT / wavelength.powi(2)
    }
    
    /// Calculate total bremsstrahlung power
    /// 
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    /// * `n_electron` - Electron density in m⁻³
    /// * `n_ion` - Ion density in m⁻³
    /// * `volume` - Emitting volume in m³
    /// 
    /// # Returns
    /// Total power in Watts
    pub fn total_power(
        &self,
        temperature: f64,
        n_electron: f64,
        n_ion: f64,
        volume: f64,
    ) -> f64 {
        if temperature <= 0.0 || n_electron <= 0.0 || n_ion <= 0.0 || volume <= 0.0 {
            return 0.0;
        }
        
        // Total bremsstrahlung power (integrated over all frequencies)
        let prefactor = 32.0 * PI * ELECTRON_CHARGE.powi(6) 
            / (3.0 * ELECTRON_MASS * SPEED_OF_LIGHT.powi(3) * PLANCK_CONSTANT * (4.0 * PI * EPSILON_0).powi(3));
        
        let thermal_factor = (2.0 * PI * BOLTZMANN_CONSTANT * temperature / (3.0 * ELECTRON_MASS)).sqrt();
        
        prefactor * self.z_ion.powi(2) * self.gaunt_factor * n_electron * n_ion 
            * thermal_factor * volume
    }
    
    /// Calculate ionization fraction using Saha equation
    /// 
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    /// * `pressure` - Pressure in Pa
    /// * `ionization_energy` - Ionization energy in eV
    /// 
    /// # Returns
    /// Ionization fraction (0-1)
    pub fn saha_ionization(
        &self,
        temperature: f64,
        pressure: f64,
        ionization_energy: f64,
    ) -> f64 {
        if temperature <= 0.0 || pressure <= 0.0 {
            return 0.0;
        }
        
        // Convert ionization energy to Joules
        let e_ion = ionization_energy * ELECTRON_CHARGE;
        
        // Saha constant
        let saha_const = 2.4e21; // m⁻³·K⁻³/²
        
        // Total number density
        let n_total = pressure / (BOLTZMANN_CONSTANT * temperature);
        
        // Saha equation solution
        let saha_factor = saha_const * temperature.powf(1.5) 
            * (-e_ion / (BOLTZMANN_CONSTANT * temperature)).exp() / n_total;
        
        // Saha equation: x²/(1-x) = K, which gives x² + Kx - K = 0
        // Using quadratic formula: x = (-K + sqrt(K² + 4K)) / 2
        let k = saha_factor;
        (-k + (k * k + 4.0 * k).sqrt()) / 2.0
    }
    
    /// Calculate emission spectrum
    pub fn emission_spectrum(
        &self,
        temperature: f64,
        n_electron: f64,
        n_ion: f64,
        path_length: f64,
        wavelengths: &Array1<f64>,
    ) -> Array1<f64> {
        wavelengths.mapv(|lambda| {
            self.spectral_radiance(lambda, temperature, n_electron, n_ion, path_length)
        })
    }
}

/// Calculate bremsstrahlung emission field
pub fn calculate_bremsstrahlung_emission(
    temperature_field: &Array3<f64>,
    pressure_field: &Array3<f64>,
    bubble_radius_field: &Array3<f64>,
    model: &BremsstrahlungModel,
    ionization_energy: f64,
) -> Array3<f64> {
    let mut emission_field = Array3::zeros(temperature_field.dim());
    
    for ((i, j, k), &temp) in temperature_field.indexed_iter() {
        let pressure = pressure_field[[i, j, k]];
        let radius = bubble_radius_field[[i, j, k]];
        
        if radius > 0.0 && temp > 5000.0 { // Only significant at high temperatures
            // Calculate ionization fraction
            let x_ion = model.saha_ionization(temp, pressure, ionization_energy);
            
            // Total number density
            let n_total = pressure / (BOLTZMANN_CONSTANT * temp);
            let n_electron = x_ion * n_total;
            let n_ion = n_electron; // Quasi-neutrality
            
            // Bubble volume
            let volume = 4.0 / 3.0 * PI * radius.powi(3);
            
            // Total bremsstrahlung power
            let power = model.total_power(temp, n_electron, n_ion, volume);
            
            // Power density
            emission_field[[i, j, k]] = power / volume.max(1e-20);
        }
    }
    
    emission_field
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_saha_equation() {
        let model = BremsstrahlungModel::default();
        
        // Test hydrogen ionization at 10,000 K
        let temp = 10000.0;
        let pressure = 101325.0; // 1 atm
        let e_ion = 13.6; // eV for hydrogen
        
        let x_ion = model.saha_ionization(temp, pressure, e_ion);
        assert!(x_ion > 0.0 && x_ion < 1.0);
        
        // At very high temperature, should be fully ionized
        let x_ion_hot = model.saha_ionization(50000.0, pressure, e_ion);
        assert!(x_ion_hot > 0.9);
    }
    
    #[test]
    fn test_emission_coefficient() {
        let model = BremsstrahlungModel::default();
        
        let freq = 1e15; // Hz (optical)
        let temp = 20000.0; // K
        let n_e = 1e24; // m⁻³
        let n_i = 1e24; // m⁻³
        
        let emission = model.emission_coefficient(freq, temp, n_e, n_i);
        assert!(emission > 0.0);
        
        // Should decrease with frequency
        let emission_high = model.emission_coefficient(1e16, temp, n_e, n_i);
        assert!(emission_high < emission);
    }
}