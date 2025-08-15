//! Unified thermal physics module
//!
//! This module consolidates all thermal phenomena including:
//! - Heat conduction (Fourier's law)
//! - Bioheat transfer (Pennes equation)
//! - Optical heating (photon absorption)
//! - Acoustic heating (ultrasound absorption)
//! - Hyperbolic heat transfer (Cattaneo-Vernotte)
//!
//! # Literature References
//!
//! 1. **Pennes, H. H. (1948)**. "Analysis of tissue and arterial blood temperatures 
//!    in the resting human forearm." *Journal of Applied Physiology*, 1(2), 93-122.
//!
//! 2. **Sapareto, S. A., & Dewey, W. C. (1984)**. "Thermal dose determination in 
//!    cancer therapy." *International Journal of Radiation Oncology Biology Physics*, 
//!    10(6), 787-800.
//!
//! 3. **Cattaneo, C. (1958)**. "A form of heat conduction equation which eliminates 
//!    the paradox of instantaneous propagation." *Comptes Rendus*, 247, 431-433.
//!
//! 4. **Welch, A. J., & van Gemert, M. J. (2011)**. "Optical-Thermal Response of 
//!    Laser-Irradiated Tissue" (2nd ed.). Springer. ISBN: 978-90-481-8830-7

use crate::{
    error::{KwaversResult, KwaversError, PhysicsError},
    grid::Grid,
    medium::Medium,
};
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

/// Heat source types
#[derive(Debug, Clone)]
pub enum HeatSource {
    /// Optical absorption (laser/light)
    Optical {
        /// Fluence rate [W/m²]
        fluence: Array3<f64>,
        /// Absorption coefficient [1/m]
        absorption: Array3<f64>,
    },
    /// Acoustic absorption (ultrasound)
    Acoustic {
        /// Pressure amplitude [Pa]
        pressure: Array3<f64>,
        /// Absorption coefficient [Np/m]
        absorption: Array3<f64>,
        /// Frequency [Hz]
        frequency: f64,
    },
    /// Metabolic heat generation
    Metabolic {
        /// Metabolic rate [W/m³]
        rate: Array3<f64>,
    },
    /// External heat source
    External {
        /// Power density [W/m³]
        power_density: Array3<f64>,
    },
}

/// Unified thermal calculator
#[derive(Debug)]
pub struct ThermalCalculator {
    /// Temperature field [K]
    temperature: Array3<f64>,
    /// Heat flux components [W/m²]
    heat_flux: (Array3<f64>, Array3<f64>, Array3<f64>),
    /// Thermal dose (CEM43) [equivalent minutes at 43°C]
    thermal_dose: Array3<f64>,
    /// Configuration
    config: ThermalConfig,
}

/// Thermal physics configuration
#[derive(Debug, Clone)]
pub struct ThermalConfig {
    /// Enable bioheat equation
    pub bioheat: bool,
    /// Blood perfusion rate [1/s]
    pub perfusion_rate: f64,
    /// Blood temperature [K]
    pub blood_temperature: f64,
    /// Enable hyperbolic heat transfer
    pub hyperbolic: bool,
    /// Thermal relaxation time [s]
    pub relaxation_time: f64,
    /// Reference temperature for CEM43 [K]
    pub reference_temperature: f64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            bioheat: true,
            perfusion_rate: 0.5e-3, // 0.5 mL/mL/s typical tissue
            blood_temperature: 310.15, // 37°C
            hyperbolic: false,
            relaxation_time: 20.0, // 20s for biological tissue
            reference_temperature: 316.15, // 43°C
        }
    }
}

impl ThermalCalculator {
    /// Create a new thermal calculator
    pub fn new(grid: &Grid, initial_temperature: f64) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        
        Self {
            temperature: Array3::from_elem(shape, initial_temperature),
            heat_flux: (
                Array3::zeros(shape),
                Array3::zeros(shape),
                Array3::zeros(shape),
            ),
            thermal_dose: Array3::zeros(shape),
            config: ThermalConfig::default(),
        }
    }
    
    /// Configure the thermal calculator
    pub fn with_config(mut self, config: ThermalConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Calculate heat source from various mechanisms
    pub fn calculate_heat_source(
        &self,
        source: &HeatSource,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> Array3<f64> {
        let shape = (grid.nx, grid.ny, grid.nz);
        let mut q = Array3::zeros(shape);
        
        match source {
            HeatSource::Optical { fluence, absorption } => {
                // Q = μ_a * φ (optical absorption)
                Zip::from(&mut q)
                    .and(fluence)
                    .and(absorption)
                    .for_each(|q_val, &phi, &mu_a| {
                        *q_val = mu_a * phi;
                    });
            }
            HeatSource::Acoustic { pressure, absorption, frequency } => {
                // Q = 2αfp²/ρc (acoustic absorption)
                Zip::indexed(&mut q)
                    .and(pressure)
                    .and(absorption)
                    .for_each(|(i, j, k), q_val, &p, &alpha| {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        
                        let rho = medium.density(x, y, z, grid);
                        let c = medium.sound_speed(x, y, z, grid);
                        
                        *q_val = 2.0 * alpha * frequency * p * p / (rho * c);
                    });
            }
            HeatSource::Metabolic { rate } => {
                q.assign(rate);
            }
            HeatSource::External { power_density } => {
                q.assign(power_density);
            }
        }
        
        q
    }
    
    /// Update temperature using heat diffusion equation
    pub fn update_temperature(
        &mut self,
        heat_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        let shape = self.temperature.dim();
        
        // Calculate thermal properties at each point
        let mut thermal_diffusivity = Array3::zeros(shape);
        let mut specific_heat = Array3::zeros(shape);
        let mut density = Array3::zeros(shape);
        
        Zip::indexed(&mut thermal_diffusivity)
            .and(&mut specific_heat)
            .and(&mut density)
            .for_each(|(i, j, k), alpha, cp, rho| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                *alpha = medium.thermal_diffusivity(x, y, z, grid);
                *cp = medium.specific_heat(x, y, z, grid);
                *rho = medium.density(x, y, z, grid);
            });
        
        // Calculate Laplacian of temperature
        let laplacian = self.calculate_laplacian(&self.temperature.clone(), grid)?;
        
        // Update temperature
        if self.config.hyperbolic {
            // Hyperbolic heat equation (Cattaneo-Vernotte)
            // τ ∂²T/∂t² + ∂T/∂t = α∇²T + Q/(ρc)
            // Using first-order approximation for now
            let tau = self.config.relaxation_time;
            
            Zip::from(&mut self.temperature)
                .and(&laplacian)
                .and(&thermal_diffusivity)
                .and(heat_source)
                .and(&density)
                .and(&specific_heat)
                .for_each(|t, &lap, &alpha, &q, &rho, &cp| {
                    let dt_eff = dt / (1.0 + dt / tau);
                    *t += dt_eff * (alpha * lap + q / (rho * cp));
                });
        } else {
            // Parabolic heat equation (Fourier)
            // ∂T/∂t = α∇²T + Q/(ρc)
            Zip::from(&mut self.temperature)
                .and(&laplacian)
                .and(&thermal_diffusivity)
                .and(heat_source)
                .and(&density)
                .and(&specific_heat)
                .for_each(|t, &lap, &alpha, &q, &rho, &cp| {
                    *t += dt * (alpha * lap + q / (rho * cp));
                });
        }
        
        // Add bioheat terms if enabled
        if self.config.bioheat {
            self.add_bioheat_terms(grid, medium, dt)?;
        }
        
        // Update thermal dose
        self.update_thermal_dose(dt);
        
        Ok(())
    }
    
    /// Add Pennes bioheat equation terms
    fn add_bioheat_terms(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        let omega_b = self.config.perfusion_rate;
        let t_arterial = self.config.blood_temperature;
        
        Zip::indexed(&mut self.temperature)
            .for_each(|(i, j, k), t| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                let rho = medium.density(x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);
                
                // Blood properties (typical values)
                const RHO_BLOOD: f64 = 1050.0; // kg/m³
                const CP_BLOOD: f64 = 3617.0; // J/(kg·K)
                
                // Perfusion term: ω_b * ρ_b * c_b * (T_a - T) / (ρ * c_p)
                let perfusion_term = omega_b * RHO_BLOOD * CP_BLOOD * (t_arterial - *t) / (rho * cp);
                
                *t += dt * perfusion_term;
            });
        
        Ok(())
    }
    
    /// Calculate Laplacian using finite differences
    fn calculate_laplacian(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut laplacian = Array3::zeros((nx, ny, nz));
        
        let dx2_inv = 1.0 / (grid.dx * grid.dx);
        let dy2_inv = 1.0 / (grid.dy * grid.dy);
        let dz2_inv = 1.0 / (grid.dz * grid.dz);
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    laplacian[(i, j, k)] = 
                        (field[(i+1, j, k)] - 2.0 * field[(i, j, k)] + field[(i-1, j, k)]) * dx2_inv +
                        (field[(i, j+1, k)] - 2.0 * field[(i, j, k)] + field[(i, j-1, k)]) * dy2_inv +
                        (field[(i, j, k+1)] - 2.0 * field[(i, j, k)] + field[(i, j, k-1)]) * dz2_inv;
                }
            }
        }
        
        Ok(laplacian)
    }
    
    /// Update cumulative thermal dose (CEM43)
    fn update_thermal_dose(&mut self, dt: f64) {
        let t_ref = self.config.reference_temperature;
        const R: f64 = 0.5; // For T > 43°C
        const R_LOW: f64 = 0.25; // For T < 43°C
        
        Zip::from(&mut self.thermal_dose)
            .and(&self.temperature)
            .for_each(|dose, &t| {
                if t > t_ref {
                    *dose += dt * R.powf(t - t_ref) / 60.0; // Convert to minutes
                } else if t > 310.15 { // Above 37°C
                    *dose += dt * R_LOW.powf(t_ref - t) / 60.0;
                }
            });
    }
    
    /// Calculate heat flux using Fourier's law
    pub fn calculate_heat_flux(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = self.temperature.dim();
        
        // Calculate gradients
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    
                    let k_thermal = medium.thermal_conductivity(x, y, z, grid);
                    
                    // q = -k∇T
                    self.heat_flux.0[(i, j, k)] = -k_thermal * 
                        (self.temperature[(i+1, j, k)] - self.temperature[(i-1, j, k)]) / (2.0 * grid.dx);
                    self.heat_flux.1[(i, j, k)] = -k_thermal * 
                        (self.temperature[(i, j+1, k)] - self.temperature[(i, j-1, k)]) / (2.0 * grid.dy);
                    self.heat_flux.2[(i, j, k)] = -k_thermal * 
                        (self.temperature[(i, j, k+1)] - self.temperature[(i, j, k-1)]) / (2.0 * grid.dz);
                }
            }
        }
        
        Ok(())
    }
    
    /// Get temperature field
    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }
    
    /// Get thermal dose field
    pub fn thermal_dose(&self) -> &Array3<f64> {
        &self.thermal_dose
    }
    
    /// Get heat flux components
    pub fn heat_flux(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.heat_flux.0, &self.heat_flux.1, &self.heat_flux.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optical_heating() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let mut calc = ThermalCalculator::new(&grid, 310.15); // 37°C
        
        // Create optical source
        let fluence = Array3::from_elem((10, 10, 10), 1000.0); // 1000 W/m²
        let absorption = Array3::from_elem((10, 10, 10), 100.0); // 100 1/m
        
        let source = HeatSource::Optical { fluence, absorption };
        
        // Mock medium
        struct MockMedium;
        impl Medium for MockMedium {
            fn density(&self, _: f64, _: f64, _: f64, _: &Grid) -> f64 { 1000.0 }
            fn sound_speed(&self, _: f64, _: f64, _: f64, _: &Grid) -> f64 { 1500.0 }
            fn absorption_coefficient(&self, _: f64, _: f64, _: f64, _: &Grid) -> f64 { 0.1 }
            fn nonlinearity_coefficient(&self, _: f64, _: f64, _: f64, _: &Grid) -> f64 { 3.5 }
            fn thermal_conductivity(&self, _: f64, _: f64, _: f64, _: &Grid) -> f64 { 0.5 }
            fn specific_heat(&self, _: f64, _: f64, _: f64, _: &Grid) -> f64 { 4000.0 }
            fn thermal_diffusivity(&self, _: f64, _: f64, _: f64, _: &Grid) -> f64 { 1.25e-7 }
            fn optical_absorption_coefficient(&self, _: f64, _: f64, _: f64, _: &Grid) -> f64 { 100.0 }
            fn optical_scattering_coefficient(&self, _: f64, _: f64, _: f64, _: &Grid) -> f64 { 1000.0 }
        }
        
        let medium = MockMedium;
        let heat_source = calc.calculate_heat_source(&source, &grid, &medium);
        
        // Check heat source calculation
        assert_eq!(heat_source[(5, 5, 5)], 100.0 * 1000.0); // μ_a * φ
    }
    
    #[test]
    fn test_thermal_dose_calculation() {
        let grid = Grid::new(3, 3, 3, 0.001, 0.001, 0.001);
        let mut calc = ThermalCalculator::new(&grid, 310.15);
        
        // Set temperature above 43°C
        calc.temperature[(1, 1, 1)] = 317.15; // 44°C
        
        // Update thermal dose for 60 seconds
        calc.update_thermal_dose(60.0);
        
        // CEM43 = t * R^(T-43) = 60s * 0.5^(44-43) = 30s = 0.5 minutes
        assert!((calc.thermal_dose[(1, 1, 1)] - 0.5).abs() < 0.01);
    }
}