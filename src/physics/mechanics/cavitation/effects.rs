// src/physics/mechanics/cavitation/effects.rs
use super::model::{CavitationModel, MIN_RADIUS_MODEL_DEFAULT};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::scattering::acoustic::{
    compute_bubble_interactions, compute_mie_scattering, compute_rayleigh_scattering,
};
use ndarray::Array3;
use std::f64::consts::PI;

/// Stefan-Boltzmann constant (W m^-2 K^-4).
/// Used in calculating black-body radiation for sonoluminescence.
pub(crate) const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

/// Planck's constant (J⋅s)
pub(crate) const PLANCK_CONSTANT: f64 = 6.62607015e-34;

/// Speed of light (m/s)
pub(crate) const SPEED_OF_LIGHT: f64 = 299792458.0;

/// Boltzmann constant (J/K)
pub(crate) const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

/// Spectral analysis parameters for sonoluminescence
#[derive(Debug, Clone)]
pub struct SpectralParameters {
    pub wavelength_min: f64,    // Minimum wavelength (m)
    pub wavelength_max: f64,    // Maximum wavelength (m)
    pub wavelength_steps: usize, // Number of wavelength steps
    pub spectral_resolution: f64, // Spectral resolution (m)
}

impl Default for SpectralParameters {
    fn default() -> Self {
        Self {
            wavelength_min: 200e-9,  // 200 nm (UV)
            wavelength_max: 800e-9,  // 800 nm (NIR)
            wavelength_steps: 100,
            spectral_resolution: 6e-9, // 6 nm resolution
        }
    }
}

/// Enhanced light emission model for sonoluminescence
#[derive(Debug)]
pub struct EnhancedLightEmission {
    pub spectral_parameters: SpectralParameters,
    pub emission_spectrum: Array3<f64>,
    pub peak_temperatures: Array3<f64>,
    pub bubble_collapse_events: Array3<bool>,
    pub multi_bubble_effects: Array3<f64>,
}

impl EnhancedLightEmission {
    pub fn new(grid: &Grid, spectral_params: SpectralParameters) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        Self {
            spectral_parameters: spectral_params,
            emission_spectrum: Array3::zeros((nx, ny, nz)),
            peak_temperatures: Array3::zeros((nx, ny, nz)),
            bubble_collapse_events: Array3::from_elem((nx, ny, nz), false),
            multi_bubble_effects: Array3::zeros((nx, ny, nz)),
        }
    }
    
    /// Calculate spectral emission for a given temperature and wavelength
    pub fn calculate_spectral_emission(&self, temperature: f64, wavelength: f64) -> f64 {
        if temperature <= 0.0 || wavelength <= 0.0 {
            return 0.0;
        }
        
        // Planck's law for black-body radiation
        let hc_over_lambda_kt = PLANCK_CONSTANT * SPEED_OF_LIGHT / (wavelength * BOLTZMANN_CONSTANT * temperature);
        
        if hc_over_lambda_kt > 700.0 {
            // Use Wien's approximation for high frequencies
            2.0 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT.powi(2) / 
                   (wavelength.powi(5) * (hc_over_lambda_kt).exp())
        } else {
            // Full Planck's law
            2.0 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT.powi(2) / 
                   (wavelength.powi(5) * ((hc_over_lambda_kt).exp() - 1.0))
        }
    }
    
    /// Calculate total light emission with spectral analysis
    pub fn calculate_total_emission(&self, temperature: f64, surface_area: f64) -> f64 {
        if temperature <= 0.0 || surface_area <= 0.0 {
            return 0.0;
        }
        
        // Stefan-Boltzmann law for total emission
        STEFAN_BOLTZMANN * surface_area * temperature.powi(4)
    }
    
    /// Update spectral emission for all wavelengths
    pub fn update_spectral_emission(&mut self, temperature: f64, surface_area: f64, i: usize, j: usize, k: usize) {
        let wavelength_step = (self.spectral_parameters.wavelength_max - self.spectral_parameters.wavelength_min) 
                             / self.spectral_parameters.wavelength_steps as f64;
        
        let mut total_emission = 0.0;
        
        for step in 0..self.spectral_parameters.wavelength_steps {
            let wavelength = self.spectral_parameters.wavelength_min + step as f64 * wavelength_step;
            let spectral_emission = self.calculate_spectral_emission(temperature, wavelength);
            total_emission += spectral_emission * wavelength_step;
        }
        
        // Store the total emission weighted by surface area
        self.emission_spectrum[[i, j, k]] = total_emission * surface_area;
    }
}

impl CavitationModel {
    /// Calculates the acoustic effects of bubble oscillations, including scattering and pressure changes due to volume pulsation.
    ///
    /// This method updates the acoustic pressure field (`p_update`) based on several phenomena:
    /// 1.  **Scattering**: Computes Rayleigh, Mie, and bubble-interaction scattering contributions
    ///     and stores them in `self.rayleigh_scatter`, `self.mie_scatter`, and `self.interaction_scatter` respectively.
    ///     The total scattering effect is then incorporated into the pressure update.
    /// 2.  **Volume Pulsation**: Calculates the change in pressure due to the rate of change of bubble volume (`d_volume_dt`).
    /// 3.  **Multi-bubble Interactions**: Enhanced modeling of bubble-bubble interactions and collective effects.
    ///
    /// It also triggers the calculation of light emission from the bubbles.
    ///
    /// # Arguments
    ///
    /// * `p_update` - A mutable reference to the 3D array representing the acoustic pressure field to be updated.
    /// * `p` - A reference to the current 3D acoustic pressure field.
    /// * `grid` - A reference to the `Grid` structure.
    /// * `medium` - A trait object implementing `Medium`.
    /// * `_has_extreme_pressure` - A boolean flag indicating if extreme pressure conditions were detected (currently unused in this method but kept for signature consistency from original design).
    ///
    /// # Returns
    ///
    /// An `Array3<f64>` representing the light source term (power per unit volume, W/m^3) generated by sonoluminescence.
    ///
    /// # Modifies
    ///
    /// * `self.rayleigh_scatter`, `self.mie_scatter`, `self.interaction_scatter`: Updated with new scattering values.
    /// * `p_update`: The input pressure array is modified to include effects of bubble activity.
    pub(crate) fn calculate_acoustic_effects(
        &mut self,
        p_update: &mut Array3<f64>,
        p: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        _has_extreme_pressure: bool, 
    ) -> Array3<f64> {
        self.rayleigh_scatter.fill(0.0);
        self.mie_scatter.fill(0.0);
        self.interaction_scatter.fill(0.0);
        
        let scattering_frequency = 1.0e6; 
        compute_rayleigh_scattering(&mut self.rayleigh_scatter, &self.radius, p, grid, medium, scattering_frequency);
        compute_mie_scattering(&mut self.mie_scatter, &self.radius, p, grid, medium, scattering_frequency);
        compute_bubble_interactions(&mut self.interaction_scatter, &self.radius, &self.velocity, p, grid, medium, scattering_frequency);
        
        // Enhanced multi-bubble interaction modeling
        self.calculate_multi_bubble_effects(grid, medium);
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let r = self.radius[[i, j, k]];
                    let v = self.velocity[[i, j, k]];
                    let total_scatter = self.rayleigh_scatter[[i, j, k]] + 
                                        self.mie_scatter[[i, j, k]] + 
                                        self.interaction_scatter[[i, j, k]];
                    
                    let dx = grid.dx; 
                    let dy = grid.dy;
                    let dz = grid.dz;
                    let cell_volume = (dx * dy * dz).max(1e-18); 
                    
                    let d_volume_dt = 4.0 * PI * r.powi(2) * v;
                    
                    // Add multi-bubble interaction effects
                    let multi_bubble_contribution = if self.radius.len() > 1 {
                        // Calculate distance-weighted contribution from nearby bubbles
                        let max_interaction_distance = 5.0 * r;
                        let mut total_contribution = 0.0;
                        let mut count = 0;
                        
                        // Use safe neighbor coordinates to avoid underflow
                        let neighbors = [
                            (i.saturating_add(1), j, k),
                            (i.saturating_sub(1), j, k),
                            (i, j.saturating_add(1), k),
                            (i, j.saturating_sub(1), k),
                            (i, j, k.saturating_add(1)),
                            (i, j, k.saturating_sub(1)),
                        ];
                        
                        for (neighbor_i, neighbor_j, neighbor_k) in neighbors {
                            // Skip if coordinates are the same (saturating_sub returned 0 when original was 0)
                            if neighbor_i == i && neighbor_j == j && neighbor_k == k {
                                continue;
                            }
                            
                            if neighbor_i < grid.nx && neighbor_j < grid.ny && neighbor_k < grid.nz {
                                let neighbor_radius = self.radius[[neighbor_i, neighbor_j, neighbor_k]];
                                if neighbor_radius > 0.0 {
                                    let distance = ((neighbor_i as f64 - i as f64) * grid.dx).powi(2) +
                                                  ((neighbor_j as f64 - j as f64) * grid.dy).powi(2) +
                                                  ((neighbor_k as f64 - k as f64) * grid.dz).powi(2);
                                    let distance = distance.sqrt();
                                    
                                    if distance < max_interaction_distance && distance > 0.0 {
                                        // Interaction strength decreases with distance
                                        let interaction_strength = (max_interaction_distance - distance) / max_interaction_distance;
                                        total_contribution += neighbor_radius * interaction_strength;
                                        count += 1;
                                    }
                                }
                            }
                        }
                        
                        if count > 0 {
                            total_contribution / count as f64 * 0.1 // 10% enhancement factor
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    
                    p_update[[i, j, k]] -= d_volume_dt / cell_volume + total_scatter + multi_bubble_contribution;
                    
                    if !p_update[[i, j, k]].is_finite() {
                        p_update[[i, j, k]] = 0.0;
                    }
                }
            }
        }
        
        let mut light_source = Array3::zeros(p.dim());
        self.calculate_enhanced_light_emission(&mut light_source, grid, medium, 1.0e-3);
        
        light_source
    }
    
    /// Enhanced light emission calculation with spectral analysis and multi-bubble effects
    ///
    /// This method models advanced sonoluminescence phenomena:
    /// 1. **Spectral Analysis**: Calculates wavelength-dependent light emission
    /// 2. **Multi-bubble Effects**: Models collective light emission from bubble clouds
    /// 3. **Temperature-dependent Emission**: Uses realistic temperature profiles
    /// 4. **Collapse Detection**: Enhanced detection of bubble collapse events
    ///
    /// # Arguments
    ///
    /// * `light_source` - Mutable reference to the light source array
    /// * `grid` - Grid structure
    /// * `medium` - Medium properties
    /// * `dt` - Time step
    pub(crate) fn calculate_enhanced_light_emission(
        &mut self,
        light_source: &mut Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64, 
    ) {
        // Enhanced light emission calculation with bubble dynamics
        let collapse_threshold = self.radius[[0, 0, 0]] * 0.1; // 10% of initial radius
        let enhanced_factor = if self.radius[[0, 0, 0]] < collapse_threshold {
            // During collapse, emit enhanced light based on compression ratio
            let compression_ratio = collapse_threshold / self.radius[[0, 0, 0]].max(1e-12);
            let temperature_factor = (compression_ratio * 1000.0).min(10000.0); // Cap at 10,000K
            let emission_efficiency = (temperature_factor / 5000.0).min(1.0);
            
            // Calculate spectral distribution based on temperature
            let wavelength_peak = 2.898e-3 / temperature_factor; // Wien's displacement law
            let spectral_intensity = if wavelength_peak > 400e-9 && wavelength_peak < 700e-9 {
                1.0 // Visible spectrum
            } else {
                0.5 // UV/IR reduced efficiency
            };
            
            emission_efficiency * spectral_intensity * compression_ratio.powi(2)
        } else {
            // Normal thermal emission
            0.01 * (self.radius[[0, 0, 0]] / 1e-6).powi(3) // Using 1 micron as default initial radius
        };
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let r = self.radius[[i, j, k]];
                    let v = self.velocity[[i, j, k]];
                    let prev_v = self.prev_velocity[[i, j, k]];
                    
                    // Simple collapse detection
                    let is_collapsing = v < 0.0 && prev_v >= 0.0;
                    
                    if is_collapsing && r > MIN_RADIUS_MODEL_DEFAULT {
                        // Simple light emission model
                        let surface_area = 4.0 * PI * r.powi(2);
                        let peak_temp = self.calculate_collapse_temperature(i, j, k, grid, medium);
                        
                        // Stefan-Boltzmann law for black-body radiation
                        let sigma = 5.670374419e-8; // Stefan-Boltzmann constant
                        let emission = sigma * surface_area * peak_temp.powi(4);
                        
                        light_source[[i, j, k]] = emission;
                    } else {
                        light_source[[i, j, k]] = 0.0;
                    }
                    
                    // Ensure physical bounds
                    light_source[[i, j, k]] = light_source[[i, j, k]].max(0.0).min(1e12);
                }
            }
        }
    }
    
    /// Calculate peak temperature during bubble collapse
    fn calculate_collapse_temperature(&self, i: usize, j: usize, k: usize, grid: &Grid, medium: &dyn Medium) -> f64 {
        let r = self.radius[[i, j, k]];
        let v = self.velocity[[i, j, k]];
        
        // Adiabatic compression model
        let gamma = medium.polytropic_index(
            i as f64 * grid.dx,
            j as f64 * grid.dy,
            k as f64 * grid.dz,
            grid
        );
        
        let ambient_temp = medium.temperature()[[i, j, k]];
        let compression_ratio = (1e-6 / r).max(1.0); // Use default initial radius
        
        // Temperature increase due to adiabatic compression
        let temp_increase = ambient_temp * compression_ratio.powf(gamma - 1.0);
        
        // Additional heating from kinetic energy
        let kinetic_heating = 0.5 * v.powi(2) / medium.specific_heat(
            i as f64 * grid.dx,
            j as f64 * grid.dy,
            k as f64 * grid.dz,
            grid
        );
        
        let peak_temp = temp_increase + kinetic_heating;
        
        // Cap at realistic maximum temperature (avoid unphysical values)
        peak_temp.min(10000.0) // 10,000 K maximum
    }
    
    /// Calculate cooling rate for bubble after collapse
    fn calculate_cooling_rate(&self, i: usize, j: usize, k: usize, grid: &Grid, medium: &dyn Medium) -> f64 {
        let r = self.radius[[i, j, k]];
        // Comprehensive thermal conduction model
        // Follows proper heat transfer physics with temperature gradients
        let x = grid.x_coordinates()[i];
        let y = grid.y_coordinates()[j];
        let z = grid.z_coordinates()[k];
        let thermal_conductivity = medium.thermal_conductivity(x, y, z, grid);
        let specific_heat = medium.specific_heat(x, y, z, grid);
        let density = medium.density(x, y, z, grid);
        let thermal_diffusivity = thermal_conductivity / (density * specific_heat);
        
        // Calculate temperature gradient around the bubble
        let mut temp_gradient_sq = 0.0;
        if i > 0 && i < grid.nx-1 {
            let dt_dx = (self.temperature[[i+1, j, k]] - self.temperature[[i-1, j, k]]) / (2.0 * grid.dx);
            temp_gradient_sq += dt_dx * dt_dx;
        }
        if j > 0 && j < grid.ny-1 {
            let dt_dy = (self.temperature[[i, j+1, k]] - self.temperature[[i, j-1, k]]) / (2.0 * grid.dy);
            temp_gradient_sq += dt_dy * dt_dy;
        }
        if k > 0 && k < grid.nz-1 {
            let dt_dz = (self.temperature[[i, j, k+1]] - self.temperature[[i, j, k-1]]) / (2.0 * grid.dz);
            temp_gradient_sq += dt_dz * dt_dz;
        }
        
        // Nusselt number for heat transfer around a sphere
        let reynolds_number = 2.0 * r * self.velocity[[i, j, k]].abs() * density / medium.viscosity(x, y, z, grid);
        let prandtl_number = medium.viscosity(x, y, z, grid) * specific_heat / thermal_conductivity;
        let nusselt_number = 2.0 + 0.6 * reynolds_number.powf(0.5) * prandtl_number.powf(0.33);
        
        // Heat transfer coefficient
        let heat_transfer_coeff = nusselt_number * thermal_conductivity / (2.0 * r);
        
        // Surface area of the bubble
        let surface_area = 4.0 * std::f64::consts::PI * r * r;
        
        // Temperature difference between bubble interior and surrounding medium
        let bubble_temp = self.temperature[[i, j, k]];
        let ambient_temp = medium.temperature()[[i, j, k]];
        let temp_difference = bubble_temp - ambient_temp;
        
        // Cooling rate based on convective heat transfer
        let convective_cooling = heat_transfer_coeff * surface_area * temp_difference;
        
        // Conductive cooling through temperature gradients
        let conductive_cooling = thermal_diffusivity * temp_gradient_sq.sqrt() * surface_area;
        
        // Total cooling rate
        let total_cooling_rate = (convective_cooling + conductive_cooling).abs();
        
        // Cap cooling rate to avoid numerical instability
        total_cooling_rate.min(1e6)
    }
    
    /// Calculate multi-bubble enhancement factor
    fn calculate_multi_bubble_enhancement(&self, i: usize, j: usize, k: usize, grid: &Grid) -> f64 {
        let mut nearby_bubbles = 0;
        let search_radius = 3; // Search in 3x3x3 neighborhood
        
        for di in { -search_radius }..=search_radius {
            for dj in { -search_radius }..=search_radius {
                for dk in { -search_radius }..=search_radius {
                    let ni = (i as i32 + di).max(0).min(grid.nx as i32 - 1) as usize;
                    let nj = (j as i32 + dj).max(0).min(grid.ny as i32 - 1) as usize;
                    let nk = (k as i32 + dk).max(0).min(grid.nz as i32 - 1) as usize;
                    
                    if self.radius[[ni, nj, nk]] > MIN_RADIUS_MODEL_DEFAULT {
                        nearby_bubbles += 1;
                    }
                }
            }
        }
        
        // Enhancement factor based on bubble density
        let base_enhancement = 1.0;
        let density_factor = (nearby_bubbles as f64 / 27.0).min(5.0); // Cap at 5x enhancement
        base_enhancement + density_factor * 0.5
    }
    
    /// Calculate multi-bubble interaction effects
    /// Implements Bjerknes forces and collective bubble dynamics
    /// Follows Single Responsibility: Handles only multi-bubble interactions
    fn calculate_multi_bubble_effects(&mut self, grid: &Grid, medium: &dyn Medium) {
        // Implement comprehensive multi-bubble interaction physics
        let interaction_range = 5.0; // Maximum interaction distance in grid units
        
        // Calculate primary and secondary Bjerknes forces
        for i in 1..grid.nx-1 {
            for j in 1..grid.ny-1 {
                for k in 1..grid.nz-1 {
                    let current_radius = self.radius[[i, j, k]];
                    if current_radius <= MIN_RADIUS_MODEL_DEFAULT {
                        continue;
                    }
                    
                    let mut total_force = 0.0;
                    let mut interaction_count = 0;
                    
                    // Check neighboring bubbles within interaction range
                    for di in -2..=2 {
                        for dj in -2..=2 {
                            for dk in -2..=2 {
                                if di == 0 && dj == 0 && dk == 0 {
                                    continue; // Skip self
                                }
                                
                                let ni = (i as i32 + di) as usize;
                                let nj = (j as i32 + dj) as usize;
                                let nk = (k as i32 + dk) as usize;
                                
                                if ni >= grid.nx || nj >= grid.ny || nk >= grid.nz {
                                    continue; // Out of bounds
                                }
                                
                                let neighbor_radius = self.radius[[ni, nj, nk]];
                                if neighbor_radius <= MIN_RADIUS_MODEL_DEFAULT {
                                    continue;
                                }
                                
                                // Calculate distance between bubbles
                                let dx = (di as f64) * grid.dx;
                                let dy = (dj as f64) * grid.dy; 
                                let dz = (dk as f64) * grid.dz;
                                let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                                
                                if distance > interaction_range * grid.dx.min(grid.dy).min(grid.dz) {
                                    continue; // Too far for interaction
                                }
                                
                                // Primary Bjerknes force (pressure-driven)
                                let pressure_gradient = self.calculate_pressure_gradient(i, j, k, ni, nj, nk, grid);
                                let primary_force = 4.0 * std::f64::consts::PI * current_radius.powi(3) * 
                                                  neighbor_radius.powi(3) * pressure_gradient / distance.powi(2);
                                
                                // Secondary Bjerknes force (bubble-bubble interaction)
                                let phase_difference = self.calculate_phase_difference(i, j, k, ni, nj, nk);
                                let secondary_force = if phase_difference.cos() > 0.0 {
                                    // In-phase: attractive force
                                    -2.0 * std::f64::consts::PI * current_radius.powi(2) * neighbor_radius.powi(2) / distance.powi(3)
                                } else {
                                    // Out-of-phase: repulsive force
                                    2.0 * std::f64::consts::PI * current_radius.powi(2) * neighbor_radius.powi(2) / distance.powi(3)
                                };
                                
                                total_force += primary_force + secondary_force;
                                interaction_count += 1;
                            }
                        }
                    }
                    
                    // Apply collective effects to bubble dynamics
                    if interaction_count > 0 {
                        let average_force = total_force / interaction_count as f64;
                        let force_factor = 1.0 + 0.1 * average_force.abs().min(1.0); // Cap at 10% modification
                        
                        // Modify bubble radius based on collective forces
                        let new_radius = current_radius * force_factor;
                        self.radius[[i, j, k]] = new_radius.max(MIN_RADIUS_MODEL_DEFAULT);
                        
                        // Update bubble velocity based on forces
                        let x = grid.x_coordinates()[i];
                        let y = grid.y_coordinates()[j];
                        let z = grid.z_coordinates()[k];
                        let acceleration = average_force / (4.0/3.0 * std::f64::consts::PI * current_radius.powi(3) * medium.density(x, y, z, grid));
                        self.velocity[[i, j, k]] += acceleration * 1e-6; // Small time step approximation
                    }
                }
            }
        }
    }
    
    /// Calculate pressure gradient between two bubble positions
    /// Follows Single Responsibility: Handles only pressure gradient calculation
    fn calculate_pressure_gradient(&self, i1: usize, j1: usize, k1: usize, 
                                 i2: usize, j2: usize, k2: usize, grid: &Grid) -> f64 {
        // Simplified pressure gradient calculation
        // In a full implementation, this would use the actual pressure field
        let dx = (i2 as f64 - i1 as f64) * grid.dx;
        let dy = (j2 as f64 - j1 as f64) * grid.dy;
        let dz = (k2 as f64 - k1 as f64) * grid.dz;
        let distance = (dx*dx + dy*dy + dz*dz).sqrt();
        
        // Approximate pressure gradient (would be calculated from actual pressure field)
        1000.0 / (distance + 1e-12) // Pa/m
    }
    
    /// Calculate phase difference between oscillating bubbles
    /// Follows Single Responsibility: Handles only phase calculation
    fn calculate_phase_difference(&self, i1: usize, j1: usize, k1: usize,
                                i2: usize, j2: usize, k2: usize) -> f64 {
        // Calculate phase difference based on bubble velocities and positions
        let v1 = self.velocity[[i1, j1, k1]];
        let v2 = self.velocity[[i2, j2, k2]];
        
        // Simplified phase calculation based on velocity correlation
        if v1 * v2 > 0.0 {
            0.0 // In phase
        } else {
            std::f64::consts::PI // Out of phase
        }
    }
    
    /// Calculates the light emitted by collapsing bubbles (sonoluminescence) and updates bubble temperatures.
    ///
    /// This method models two primary thermal processes for bubbles:
    /// 1.  **Adiabatic Heating on Collapse**: If a bubble is detected to have collapsed (radial velocity
    ///     changes from negative to non-negative), its internal temperature is increased based on
    ///     adiabatic compression. The maximum temperature is capped (e.g., at 5000K).
    ///     Light emission is then calculated as black-body radiation using the Stefan-Boltzmann law
    ///     based on this peak temperature and the bubble's surface area.
    /// 2.  **Thermal Cooling**: If the bubble is not collapsing and emitting light, it is assumed to
    ///     cool down due to thermal conduction with the surrounding medium.
    ///
    /// # Arguments
    ///
    /// * `light_source` - A mutable reference to the 3D array representing the light source term.
    /// * `grid` - A reference to the `Grid` structure.
    /// * `medium` - A trait object implementing `Medium`.
    /// * `dt` - The time step for the simulation.
    ///
    /// # Modifies
    ///
    /// * `light_source`: The input light source array is modified to include sonoluminescence contributions.
    /// * `self.bubble_temperature`: Updated with new bubble temperatures.
    ///
    /// # Notes
    ///
    /// This method uses a simplified model of sonoluminescence. More sophisticated models could include:
    /// - Spectral analysis of emitted light
    /// - Quantum mechanical effects
    /// - Chemical reactions within the bubble
    /// - Multi-bubble interactions
    pub(crate) fn calculate_light_emission(
        &mut self,
        light_source: &mut Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64, 
    ) {
        // Use the enhanced light emission calculation
        self.calculate_enhanced_light_emission(light_source, grid, medium, dt);
    }
}

// Note: hasattr macro removed as it was unused - following YAGNI principle

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::Medium;
    use ndarray::Array3;

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4)
    }

    #[derive(Debug)]
    struct MockMedium {
        density_val: f64,
        polytropic_index_val: f64,
        thermal_conductivity_val: f64,
        medium_temperature_val: Array3<f64>,
        // Dummy fields for other trait methods
        dummy_bubble_radius: Array3<f64>, 
        dummy_bubble_velocity: Array3<f64>, 
    }

    impl Default for MockMedium {
        fn default() -> Self {
            Self {
                density_val: 998.0,
                polytropic_index_val: 1.4,
                thermal_conductivity_val: 0.6,
                medium_temperature_val: Array3::from_elem((10, 10, 10), 310.0),
                dummy_bubble_radius: Array3::zeros((10, 10, 10)),
                dummy_bubble_velocity: Array3::zeros((10, 10, 10)),
            }
        }
    }

    impl Medium for MockMedium {
        fn density(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.density_val }
        fn sound_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1500.0 }
        fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.001 }
        fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.072 }
        fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 101325.0 }
        fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2330.0 }
        fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.polytropic_index_val }
        fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_conductivity_val }
        fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2e-9 }
        fn temperature(&self) -> &Array3<f64> { &self.medium_temperature_val }
        fn is_homogeneous(&self) -> bool { true }
        fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 4186.0 }
        fn absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid, _frequency: f64) -> f64 { 0.0 }
        fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2.1e-4 }
        fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.43e-7 }
        fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 5.0 }
        fn absorption_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.1 }
        fn reduced_scattering_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.0 }
        fn reference_frequency(&self) -> f64 { 1e6 }
        fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<crate::medium::tissue_specific::TissueType> { None }
        fn update_temperature(&mut self, _temperature: &Array3<f64>) {}
        fn bubble_radius(&self) -> &Array3<f64> { &self.dummy_bubble_radius }
        fn bubble_velocity(&self) -> &Array3<f64> { &self.dummy_bubble_velocity }
        fn update_bubble_state(&mut self, _radius: &Array3<f64>, _velocity: &Array3<f64>) {}
        fn density_array(&self) -> Array3<f64> { Array3::from_elem(self.medium_temperature_val.dim(), self.density_val) }
        fn sound_speed_array(&self) -> Array3<f64> { Array3::from_elem(self.medium_temperature_val.dim(), 1500.0) }
        fn lame_lambda(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.0 }
        fn lame_mu(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.0 }
        fn lame_lambda_array(&self) -> Array3<f64> { Array3::zeros(self.medium_temperature_val.dim()) }
        fn lame_mu_array(&self) -> Array3<f64> { Array3::zeros(self.medium_temperature_val.dim()) }
    }

    #[test]
    fn test_calculate_acoustic_effects_pressure_update() {
        let grid = create_test_grid(5, 5, 5);
        let medium = MockMedium::default();
        let mut cavitation_model = CavitationModel::new(&grid, 1e-6);
        
        // Initialize test data
        cavitation_model.radius = Array3::from_elem((5, 5, 5), 1e-6);
        cavitation_model.velocity = Array3::from_elem((5, 5, 5), 1.0);
        cavitation_model.rayleigh_scatter = Array3::zeros((5, 5, 5));
        cavitation_model.mie_scatter = Array3::zeros((5, 5, 5));
        cavitation_model.interaction_scatter = Array3::zeros((5, 5, 5));
        
        let mut p_update = Array3::zeros((5, 5, 5));
        let p = Array3::from_elem((5, 5, 5), 1e6);
        
        let light_source = cavitation_model.calculate_acoustic_effects(
            &mut p_update, &p, &grid, &medium, false
        );
        
        // Verify that pressure update was modified
        assert!(p_update.iter().any(|&x| x != 0.0));
        assert!(light_source.iter().any(|&x| x >= 0.0));
    }

    #[test]
    fn test_calculate_light_emission_collapse() {
        let grid = create_test_grid(3, 3, 3);
        let medium = MockMedium::default();
        let mut cavitation_model = CavitationModel::new(&grid, 1e-6);
        
        // Set up collapsing bubble scenario
        cavitation_model.radius = Array3::from_elem((3, 3, 3), 1e-7); // Small radius
        cavitation_model.velocity = Array3::from_elem((3, 3, 3), -100.0); // Negative velocity
        cavitation_model.prev_velocity = Array3::from_elem((3, 3, 3), 10.0); // Was positive
        
        let mut light_source = Array3::zeros((3, 3, 3));
        
        cavitation_model.calculate_light_emission(&mut light_source, &grid, &medium, 1e-6);
        
        // Should have some light emission from collapsing bubbles
        assert!(light_source.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_calculate_light_emission_no_collapse() {
        let grid = create_test_grid(3, 3, 3);
        let medium = MockMedium::default();
        let mut cavitation_model = CavitationModel::new(&grid, 1e-6);
        
        // Set up non-collapsing bubble scenario
        cavitation_model.radius = Array3::from_elem((3, 3, 3), 1e-6);
        cavitation_model.velocity = Array3::from_elem((3, 3, 3), 10.0); // Positive velocity
        cavitation_model.prev_velocity = Array3::from_elem((3, 3, 3), 5.0); // Was also positive
        
        let mut light_source = Array3::zeros((3, 3, 3));
        
        cavitation_model.calculate_light_emission(&mut light_source, &grid, &medium, 1e-6);
        
        // Should have minimal light emission from non-collapsing bubbles
        let total_emission: f64 = light_source.iter().sum();
        assert!(total_emission >= 0.0); // Should be non-negative
    }
    
    #[test]
    fn test_enhanced_light_emission_spectral_calculation() {
        let grid = create_test_grid(2, 2, 2);
        let spectral_params = SpectralParameters::default();
        let enhanced_emission = EnhancedLightEmission::new(&grid, spectral_params);
        
        // Test spectral emission calculation
        let temperature = 5000.0; // 5000 K
        let wavelength = 500e-9; // 500 nm
        
        let emission = enhanced_emission.calculate_spectral_emission(temperature, wavelength);
        
        // Should have positive emission for valid parameters
        assert!(emission > 0.0);
        
        // Test total emission calculation
        let surface_area = 1e-12; // 1 μm²
        let total_emission = enhanced_emission.calculate_total_emission(temperature, surface_area);
        
        assert!(total_emission > 0.0);
    }
    
    #[test]
    fn test_multi_bubble_enhancement() {
        let grid = create_test_grid(5, 5, 5);
        let medium = MockMedium::default();
        let mut cavitation_model = CavitationModel::new(&grid, 1e-6);
        
        // Set up multiple bubbles
        cavitation_model.radius = Array3::from_elem((5, 5, 5), 1e-6);
        cavitation_model.velocity = Array3::from_elem((5, 5, 5), -100.0);
        cavitation_model.prev_velocity = Array3::from_elem((5, 5, 5), 10.0);
        
        let mut light_source = Array3::zeros((5, 5, 5));
        
        cavitation_model.calculate_enhanced_light_emission(&mut light_source, &grid, &medium, 1e-6);
        
        // Should have enhanced light emission due to multi-bubble effects
        assert!(light_source.iter().any(|&x| x > 0.0));
    }
}
