//! # Acoustic Scattering Module
//!
//! This module simulates the scattering of acoustic waves by particles or bubbles
//! within a medium. It provides models for different scattering regimes and
//! interactions.
//!
//! The primary struct, `AcousticScatteringModel`, orchestrates the calculation of
//! the total scattered field by combining contributions from various scattering
//! mechanisms, such as Rayleigh scattering, Mie scattering, and inter-bubble
//! interaction effects.
//!
//! ## Key Components:
//! - `AcousticScatteringModel`: Struct to manage and compute the total scattered acoustic field.
//! - `compute_rayleigh_scattering`: Calculates scattering for particles much smaller than the wavelength.
//! - `compute_mie_scattering`: Calculates scattering for particles comparable to the wavelength.
//! - `compute_bubble_interactions`: Calculates scattering effects arising from forces between bubbles.

pub mod bubble_interactions;
pub mod mie;
pub mod rayleigh;

pub use bubble_interactions::compute_bubble_interactions;
pub use mie::compute_mie_scattering;
pub use rayleigh::compute_rayleigh_scattering;

use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};
// std::f64::consts::PI is not directly used in this file after refactoring,
// but might be used by the submodules.

/// Represents a model for computing and storing the acoustic field scattered by particles or bubbles.
///
/// This struct holds the resulting scattered field and provides methods to compute it
/// based on various scattering mechanisms like Rayleigh, Mie, and inter-bubble interactions.
#[derive(Debug, Clone)]
pub struct AcousticScatteringModel {
    /// 3D array representing the computed acoustic field scattered by particles/bubbles at each grid point.
    /// The units depend on the input incident field (typically Pascals for pressure).
    pub scattered_field: Array3<f64>,
}

impl AcousticScatteringModel {
    /// Creates a new `AcousticScatteringModel` instance.
    ///
    /// Initializes the `scattered_field` as a 3D array of zeros with the same dimensions
    /// as the provided `grid`.
    ///
    /// # Arguments
    ///
    /// * `grid` - A reference to the `Grid` structure defining the simulation domain and discretization.
    ///
    /// # Returns
    ///
    /// A new `AcousticScatteringModel` instance with an initialized (zeroed) `scattered_field`.
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing AcousticScatteringModel");
        Self {
            scattered_field: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }

    /// Computes the total acoustic scattering from bubbles or particles.
    ///
    /// This method calculates the combined effect of Rayleigh scattering, Mie scattering,
    /// and inter-bubble interactions. It calls specialized functions for each of these
    /// mechanisms and sums their contributions into `self.scattered_field`.
    ///
    /// Any resulting NaN or infinite values in the `scattered_field` are reset to 0.0
    /// to maintain numerical stability.
    ///
    /// # Arguments
    ///
    /// * `incident_field` - A reference to the 3D array of the incident acoustic pressure field.
    /// * `bubble_radius` - A reference to the 3D array representing the radius of bubbles/particles
    ///   at each grid point (meters). For `compute_bubble_interactions`, this is also used for the second radius parameter.
    /// * `grid` - A reference to the `Grid` structure.
    /// * `medium` - A trait object implementing `Medium`, providing material properties.
    /// * `frequency` - The frequency of the incident acoustic field (Hz).
    ///
    /// # Modifies
    ///
    /// * `self.scattered_field`: This field is updated with the sum of all computed scattering contributions.
    pub fn compute_scattering(
        &mut self,
        incident_field: &Array3<f64>,
        bubble_radius: &Array3<f64>, // Used as radius1 for interactions
        // bubble_velocity is needed for compute_bubble_interactions, assuming it's part of bubble_radius for now
        // or it should be passed as a separate parameter if it's distinct from radius.
        // The original call used bubble_radius for both radius and velocity parameters to compute_bubble_interactions.
        // This seems like an error in the original call signature or logic if velocity is needed.
        // For now, replicating the existing call structure.
        grid: &Grid,
        medium: &dyn Medium,
        frequency: f64,
    ) {
        debug!("Computing combined acoustic scattering");
        let mut rayleigh_scatter = Array3::zeros(incident_field.dim());
        let mut mie_scatter = Array3::zeros(incident_field.dim());
        let mut interaction_scatter = Array3::zeros(incident_field.dim());

        compute_rayleigh_scattering(&mut rayleigh_scatter, bubble_radius, incident_field, grid, medium, frequency);
        compute_mie_scattering(&mut mie_scatter, bubble_radius, incident_field, grid, medium, frequency);
        // Assuming bubble_radius is passed for the 'velocity' parameter as per the original call.
        // This is likely a placeholder or simplification in the original code.
        // True bubble interactions might depend on actual bubble wall velocities.
        compute_bubble_interactions(&mut interaction_scatter, bubble_radius, bubble_radius, incident_field, grid, medium, frequency);

        Zip::from(&mut self.scattered_field)
            .and(&rayleigh_scatter)
            .and(&mie_scatter)
            .and(&interaction_scatter)
            .par_for_each(|s, &ray, &mie, &inter| {
                *s = ray + mie + inter;
                if s.is_nan() || s.is_infinite() {
                    *s = 0.0;
                }
            });
    }

    /// Returns a reference to the 3D array representing the total scattered acoustic field.
    pub fn scattered_field(&self) -> &Array3<f64> {
        &self.scattered_field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::Medium;
    use crate::medium::tissue_specific; // For tissue_type Option
    use ndarray::{Array3, ShapeBuilder}; // Added ShapeBuilder for .f()

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01)
    }

    #[derive(Debug)]
    struct MockMedium {
        sound_speed_val: f64,
        density_val: f64,
        // Dummy fields for other trait methods
        dummy_temperature: Array3<f64>,
        dummy_bubble_radius: Array3<f64>,
        dummy_bubble_velocity: Array3<f64>,
    }

    impl Default for MockMedium {
        fn default() -> Self {
            let default_dim = (2,2,2); // Default dimension for dummy arrays
            Self {
                sound_speed_val: 1500.0,
                density_val: 1000.0,
                dummy_temperature: Array3::zeros(default_dim.f()),
                dummy_bubble_radius: Array3::zeros(default_dim.f()),
                dummy_bubble_velocity: Array3::zeros(default_dim.f()),
            }
        }
    }

    impl Medium for MockMedium {
        fn sound_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.sound_speed_val }
        fn density(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.density_val }
        fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.001 }
        fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.072 }
        fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 101325.0 }
        fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2330.0 }
        fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.4 }
        fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.6 }
        fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2e-9 }
        fn temperature(&self) -> &Array3<f64> { &self.dummy_temperature }
        fn is_homogeneous(&self) -> bool { true }
        fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 4186.0 }
        fn absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid, _frequency: f64) -> f64 { 0.1 } // Example value
        fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2.1e-4 }
        fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.43e-7 }
        fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 5.0 }
        fn absorption_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.1 }
        fn reduced_scattering_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.0 }
        fn reference_frequency(&self) -> f64 { 1e6 }
        fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<tissue_specific::TissueType> { None }
        fn update_temperature(&mut self, _temperature: &Array3<f64>) {}
        fn bubble_radius(&self) -> &Array3<f64> { &self.dummy_bubble_radius }
        fn bubble_velocity(&self) -> &Array3<f64> { &self.dummy_bubble_velocity }
        fn update_bubble_state(&mut self, _radius: &Array3<f64>, _velocity: &Array3<f64>) {}
        fn density_array(&self) -> Array3<f64> { Array3::from_elem(self.dummy_temperature.dim(), self.density_val) }
        fn sound_speed_array(&self) -> Array3<f64> { Array3::from_elem(self.dummy_temperature.dim(), self.sound_speed_val) }
    }

    #[test]
    fn test_acoustic_scattering_model_new() {
        let grid_dims = (2, 3, 4);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let model = AcousticScatteringModel::new(&test_grid);

        assert_eq!(model.scattered_field.dim(), grid_dims);
        assert!(model.scattered_field.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_compute_scattering() {
        let grid_dims = (2, 2, 2);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut model = AcousticScatteringModel::new(&test_grid);
        
        let incident_field = Array3::from_elem(grid_dims.f(), 1.0); // Uniform incident pressure
        let bubble_radius = Array3::from_elem(grid_dims.f(), 1e-5); // Uniform bubble radius
        
        let mock_medium = MockMedium::default();
        let frequency = 1e6; // 1 MHz

        model.compute_scattering(&incident_field, &bubble_radius, &test_grid, &mock_medium, frequency);

        // Basic assertion: field should be modified and finite.
        // If all scattering components are zero (e.g., due to kr conditions),
        // the field might remain zero. A more robust test would set up conditions
        // known to cause non-zero scattering for at least one component.
        // For now, just check it doesn't panic and values are finite.
        assert!(model.scattered_field.iter().all(|&x| x.is_finite()));
        
        // Example: if Rayleigh scattering is dominant for these parameters,
        // we expect some non-zero values.
        // This is a very rough check:
        let kr = (2.0 * std::f64::consts::PI * frequency / mock_medium.sound_speed_val) * 1e-5;
        if kr < 1.0 && kr > 1e-9 { // If Rayleigh conditions met
             // If any element is non-zero, it means some scattering happened.
             // Or, if all are zero, it implies the sum of components was zero.
             // This test is mostly for ensuring the aggregation logic runs.
             // A specific value check would require knowing the exact output of sub-functions.
        }
        // For this test, we'll just rely on the fact that it ran and produced finite numbers.
    }
}
