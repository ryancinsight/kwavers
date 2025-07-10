// physics/scattering/acoustic/mod.rs
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
use rayon::prelude::*; // Required for par_for_each

#[derive(Debug, Clone)]
pub struct AcousticScatteringModel {
    scattered_field: Array3<f64>, // Changed to private (module private)
}

impl AcousticScatteringModel {
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing AcousticScatteringModel");
        Self {
            scattered_field: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }
}

use crate::physics::traits::AcousticScatteringModelTrait;

impl AcousticScatteringModelTrait for AcousticScatteringModel {
    fn compute_scattering(
        &mut self,
        incident_field: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        bubble_velocity: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        frequency: f64,
    ) {
        debug!("Computing combined acoustic scattering (via trait)");
        let mut rayleigh_scatter = Array3::zeros(incident_field.dim());
        let mut mie_scatter = Array3::zeros(incident_field.dim());
        let mut interaction_scatter = Array3::zeros(incident_field.dim());

        compute_rayleigh_scattering(&mut rayleigh_scatter, bubble_radius, incident_field, grid, medium, frequency);
        compute_mie_scattering(&mut mie_scatter, bubble_radius, incident_field, grid, medium, frequency);
        compute_bubble_interactions(&mut interaction_scatter, bubble_radius, bubble_velocity, incident_field, grid, medium, frequency);

        Zip::from(&mut self.scattered_field)
            .and(&rayleigh_scatter)
            .and(&mie_scatter)
            .and(&interaction_scatter)
            .par_for_each(|s, &ray, &mie, &inter| { // par_for_each needs rayon::prelude
                *s = ray + mie + inter;
                if s.is_nan() || s.is_infinite() {
                    *s = 0.0;
                }
            });
    }

    fn scattered_field(&self) -> &Array3<f64> {
        &self.scattered_field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::Medium;
    use crate::medium::tissue_specific;
    use ndarray::{Array3, ShapeBuilder};
    use crate::physics::traits::AcousticScatteringModelTrait; // For testing trait methods

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01)
    }

    #[derive(Debug)]
    struct MockMedium {
        sound_speed_val: f64,
        density_val: f64,
        dummy_temperature: Array3<f64>,
        dummy_bubble_radius: Array3<f64>,
        dummy_bubble_velocity: Array3<f64>,
    }

    impl Default for MockMedium {
        fn default() -> Self {
            let default_dim = (2,2,2);
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
        fn absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid, _frequency: f64) -> f64 { 0.1 }
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

        assert_eq!(model.scattered_field().dim(), grid_dims); // Use trait accessor
        assert!(model.scattered_field().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_compute_scattering_via_trait() { // Renamed to indicate trait usage
        let grid_dims = (2, 2, 2);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut model: Box<dyn AcousticScatteringModelTrait> = Box::new(AcousticScatteringModel::new(&test_grid)); // Use trait object
        
        let incident_field = Array3::from_elem(grid_dims.f(), 1.0);
        let bubble_radius_data = Array3::from_elem(grid_dims.f(), 1e-5);
        
        let mock_medium = MockMedium::default();
        let frequency = 1e6;

        // Call trait method
        model.compute_scattering(
            &incident_field,
            &bubble_radius_data, // Pass the owned array
            mock_medium.bubble_velocity(),
            &test_grid,
            &mock_medium,
            frequency
        );

        assert!(model.scattered_field().iter().all(|&x| x.is_finite()));
    }
}
