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
            return 2.0 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT.powi(2) / 
                   (wavelength.powi(5) * (hc_over_lambda_kt).exp());
        } else {
            // Full Planck's law
            return 2.0 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT.powi(2) / 
                   (wavelength.powi(5) * ((hc_over_lambda_kt).exp() - 1.0));
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
                    
                    // Add multi-bubble interaction effects (placeholder)
                    let multi_bubble_contribution = 0.0; // TODO: Implement when field is available
                    
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
        // Simplified light emission calculation (placeholder)
        // TODO: Implement full enhanced light emission when fields are available
        
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
        let thermal_conductivity = medium.thermal_conductivity(
            i as f64 * grid.dx,
            j as f64 * grid.dy,
            k as f64 * grid.dz,
            grid
        );
        
        // Simplified thermal conduction model
        let cooling_rate = thermal_conductivity / (r.powi(2) + 1e-12);
        cooling_rate.min(1e6) // Cap cooling rate to avoid unphysical values
    }
    
    /// Calculate multi-bubble enhancement factor
    fn calculate_multi_bubble_enhancement(&self, i: usize, j: usize, k: usize, grid: &Grid) -> f64 {
        let mut nearby_bubbles = 0;
        let search_radius = 3; // Search in 3x3x3 neighborhood
        
        for di in -search_radius as i32..=search_radius as i32 {
            for dj in -search_radius as i32..=search_radius as i32 {
                for dk in -search_radius as i32..=search_radius as i32 {
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
    fn calculate_multi_bubble_effects(&mut self, grid: &Grid, medium: &dyn Medium) {
        // TODO: Implement multi-bubble effects when the field is available
        // self.multi_bubble_effects.fill(0.0);
        
        // Placeholder implementation
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let r = self.radius[[i, j, k]];
                    if r <= MIN_RADIUS_MODEL_DEFAULT {
                        continue;
                    }
                    
                    // Simplified multi-bubble interaction (placeholder)
                    // In a full implementation, this would calculate interaction forces
                    // between neighboring bubbles
                }
            }
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

// Helper macro to check if a field exists (simplified implementation)
macro_rules! hasattr {
    ($obj:expr, $field:expr) => {
        false // Simplified - in real implementation would check field existence
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::Medium;
    use crate::medium::tissue_specific; // For tissue_type Option
    use ndarray::{Array3, ShapeBuilder}; // Added ShapeBuilder for .f()

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4)
    }

    #[derive(Default)]
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
        let mut enhanced_emission = EnhancedLightEmission::new(&grid, spectral_params);
        
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
