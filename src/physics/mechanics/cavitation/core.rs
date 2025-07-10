// src/physics/mechanics/cavitation/core.rs
// The CavitationModel::update_cavitation method has been moved to trait_impls.rs
// as part of the CavitationModelBehavior trait implementation.

// This file is now empty or can be removed if no other core logic for CavitationModel resides here.
// For now, leaving it empty. Helper methods like calculate_second_derivative, update_bubble_dynamics,
// calculate_acoustic_effects, calculate_light_emission are in dynamics.rs and effects.rs.

#[cfg(test)]
mod tests {
    // Original tests for update_cavitation might need to be adapted or moved.
    // If CavitationModel is constructed and the trait method is called,
    // the tests might still be relevant.

    // Preserving original tests from core.rs
    // They should still pass if CavitationModel is instantiated and its
    // (now trait) methods are called.
    use super::super::model::CavitationModel; // To get CavitationModel struct
    use crate::physics::traits::CavitationModelBehavior; // To use the trait method
    use crate::grid::Grid;
    use crate::medium::Medium;
    use crate::medium::tissue_specific;
    use ndarray::{Array3, ShapeBuilder};

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01)
    }

    #[derive(Debug)]
    struct MockMedium {
        density_val: f64,
        viscosity_val: f64,
        surface_tension_val: f64,
        ambient_pressure_val: f64,
        vapor_pressure_val: f64,
        polytropic_index_val: f64,
        thermal_conductivity_val: f64,
        gas_diffusion_coefficient_val: f64,
        medium_temperature_val: Array3<f64>,
        dummy_bubble_radius: Array3<f64>, 
        dummy_bubble_velocity: Array3<f64>,
    }

    impl Default for MockMedium {
        fn default() -> Self {
            let default_dim = (4,4,4);
            Self {
                density_val: 1000.0,
                viscosity_val: 0.001,
                surface_tension_val: 0.072,
                ambient_pressure_val: 101325.0,
                vapor_pressure_val: 2330.0,
                polytropic_index_val: 1.4,
                thermal_conductivity_val: 0.6,
                gas_diffusion_coefficient_val: 2e-9,
                medium_temperature_val: Array3::from_elem(default_dim.f(), 293.15),
                dummy_bubble_radius: Array3::zeros(default_dim.f()),
                dummy_bubble_velocity: Array3::zeros(default_dim.f()),
            }
        }
    }
    
    impl Medium for MockMedium {
        fn density(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.density_val }
        fn sound_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1500.0 }
        fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.viscosity_val }
        fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.surface_tension_val }
        fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.ambient_pressure_val }
        fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.vapor_pressure_val }
        fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.polytropic_index_val }
        fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_conductivity_val }
        fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.gas_diffusion_coefficient_val }
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
        fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<tissue_specific::TissueType> { None }
        fn update_temperature(&mut self, _temperature: &Array3<f64>) {}
        fn bubble_radius(&self) -> &Array3<f64> { &self.dummy_bubble_radius }
        fn bubble_velocity(&self) -> &Array3<f64> { &self.dummy_bubble_velocity }
        fn update_bubble_state(&mut self, _radius: &Array3<f64>, _velocity: &Array3<f64>) {}
        fn density_array(&self) -> Array3<f64> { Array3::from_elem(self.medium_temperature_val.dim(), self.density_val) }
        fn sound_speed_array(&self) -> Array3<f64> { Array3::from_elem(self.medium_temperature_val.dim(), 1500.0) }
    }

    #[test]
    fn test_update_cavitation_runs_without_panic_via_trait() {
        let grid_dims = (4, 4, 4); 
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut model = CavitationModel::new(&test_grid, 5e-6);

        let mut p_update = Array3::zeros(grid_dims.f());
        let p_acoustic = Array3::zeros(grid_dims.f());
        
        let mock_medium = MockMedium {
            medium_temperature_val: Array3::from_elem(grid_dims.f(), 293.15),
            dummy_bubble_radius: Array3::zeros(grid_dims.f()),
            dummy_bubble_velocity: Array3::zeros(grid_dims.f()),
            ..MockMedium::default()
        };

        let dt = 1e-7;
        let frequency = 1e6;

        // Call update_cavitation via the trait
        let _light_emission = CavitationModelBehavior::update_cavitation(
            &mut model,
            &mut p_update,
            &p_acoustic,
            &test_grid,
            dt,
            &mock_medium,
            frequency,
        );


        assert!(!p_update.iter().any(|&x| x.is_nan()), "p_update contains NaN values");
        assert!(!model.radius().iter().any(|&x| x.is_nan()), "model.radius contains NaN values");
        assert!(!model.velocity().iter().any(|&x| x.is_nan()), "model.velocity contains NaN values");
        assert!(!model.temperature().iter().any(|&x| x.is_nan()), "model.temperature contains NaN values");
    }
}
