// src/physics/mechanics/acoustic_wave/nonlinear/core.rs
// The NonlinearWave::update_wave method has been moved to trait_impls.rs
// as part of the AcousticWaveModel trait implementation.

// This file is now empty or can be removed if no other core logic resides here.
// For now, leaving it empty.

#[cfg(test)]
mod tests {
    // Original tests for update_wave might need to be adapted or moved
    // if they were testing the inherent method directly.
    // For now, keeping the test module structure if there were tests here.
    // If NonlinearWave is constructed and the trait method is called,
    // the tests might still be relevant.

    // Example of how tests might need to change:
    /*
    use super::super::config::NonlinearWave; // To get NonlinearWave struct
    use crate::physics::traits::AcousticWaveModel; // To use the trait method
    use crate::grid::Grid;
    use crate::medium::Medium;
    use crate::source::Source;
    use crate::signal::Signal;
    use ndarray::{Array3, Array4, ShapeBuilder};
    use num_complex::Complex;

    const TEST_N_FIELDS: usize = 4;

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01)
    }

    #[derive(Debug)]
    struct MockMedium { /* ... as defined before ... */ }
    impl Default for MockMedium { /* ... */ }
    impl Medium for MockMedium { /* ... */ }

    #[derive(Debug, Clone)]
    struct MockSignal;
    impl Signal for MockSignal { /* ... */ }

    #[derive(Debug)]
    struct MockSource { mock_signal: MockSignal, source_value: f64 }
    impl Default for MockSource { /* ... */ }
    impl Source for MockSource { /* ... */ }

    #[test]
    fn test_update_wave_runs_without_panic_minimal_grid_via_trait() {
        let grid_dim = 4;
        let test_grid = create_test_grid(grid_dim, grid_dim, grid_dim);
        let mut wave_solver = NonlinearWave::new(&test_grid); // Construct concrete type

        let mut fields_shape: [usize; 4] = [0; 4];
        fields_shape[0] = TEST_N_FIELDS;
        fields_shape[1] = test_grid.nx;
        fields_shape[2] = test_grid.ny;
        fields_shape[3] = test_grid.nz;

        let mut fields = Array4::<f64>::zeros(fields_shape.f());
        let prev_pressure = Array3::<f64>::zeros((test_grid.nx, test_grid.ny, test_grid.nz).f());

        let mock_medium = MockMedium::default();
        let mock_source = MockSource::default();

        let dt = 1e-5;
        let t = 0.0;

        // Call update_wave via the trait
        AcousticWaveModel::update_wave(&mut wave_solver, &mut fields, &prev_pressure, &mock_source, &test_grid, &mock_medium, dt, t);

        assert!(!fields.iter().any(|&x| x.is_nan()), "Output contains NaN values");
    }
    */
    // For now, the original tests from core.rs are preserved below.
    // They should still pass if NonlinearWave is instantiated and its
    // (now trait) methods are called.
    use super::super::config::NonlinearWave; // To get NonlinearWave struct
    use crate::physics::traits::AcousticWaveModel; // To use the trait method

    use crate::grid::Grid;
    use crate::medium::Medium;
    use crate::medium::tissue_specific;
    use crate::source::Source;
    use crate::signal::Signal;

    const TEST_N_FIELDS: usize = 4; 
    use ndarray::{Array3, Array4, ShapeBuilder, Ix3, Ix4, Array};
    use num_complex::Complex;


    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {

        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01)
    }


    #[derive(Debug)]
    struct MockMedium {
        sound_speed_val: f64,
        density_val: f64,
        nonlinearity_coeff_val: f64,
        viscosity_val: f64,
        absorption_coeff_val: f64,
        ref_freq_val: f64,

        dummy_temperature: Array3<f64>,
        dummy_bubble_radius: Array3<f64>,
        dummy_bubble_velocity: Array3<f64>,
    }

    impl Default for MockMedium {
        fn default() -> Self {
            Self {
                sound_speed_val: 1500.0,
                density_val: 1000.0,
                nonlinearity_coeff_val: 5.0,
                viscosity_val: 0.001,
                absorption_coeff_val: 0.1,
                ref_freq_val: 1e6,
                dummy_temperature: Array3::zeros((1,1,1).f()),
                dummy_bubble_radius: Array3::zeros((1,1,1).f()),
                dummy_bubble_velocity: Array3::zeros((1,1,1).f()),
            }
        }
    }

    impl Medium for MockMedium {
        fn sound_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.sound_speed_val }
        fn density(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.density_val }
        fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.nonlinearity_coeff_val }
        fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.viscosity_val }
        fn absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid, _freq: f64) -> f64 { self.absorption_coeff_val }
        fn reference_frequency(&self) -> f64 { self.ref_freq_val }


        fn is_homogeneous(&self) -> bool { true }
        fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.072 }
        fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 101325.0 }
        fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2330.0 }
        fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.4 }
        fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 4186.0 }
        fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.6 }
        fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2.1e-4 }
        fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 2e-9 }
        fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.43e-7 }
        fn absorption_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.1 }
        fn reduced_scattering_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 1.0 }
        fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<tissue_specific::TissueType> { None }

        fn update_temperature(&mut self, _temperature: &Array3<f64>) { /* dummy */ }
        fn temperature(&self) -> &Array3<f64> { &self.dummy_temperature }
        fn bubble_radius(&self) -> &Array3<f64> { &self.dummy_bubble_radius }
        fn bubble_velocity(&self) -> &Array3<f64> { &self.dummy_bubble_velocity }
        fn update_bubble_state(&mut self, _radius: &Array3<f64>, _velocity: &Array3<f64>) { /* dummy */ }
        fn density_array(&self) -> Array3<f64> { Array3::from_elem((self.dummy_temperature.dim()), self.density_val) }
        fn sound_speed_array(&self) -> Array3<f64> { Array3::from_elem((self.dummy_temperature.dim()), self.sound_speed_val) }
    }


    #[derive(Debug, Clone)]
    struct MockSignal;

    impl Signal for MockSignal {
        fn amplitude(&self, _t: f64) -> f64 { 1.0 }
        fn frequency(&self, _t: f64) -> f64 { 1e6 }
        fn phase(&self, _t: f64) -> f64 { 0.0 }
        fn clone_box(&self) -> Box<dyn Signal> { Box::new(self.clone()) }
    }


    #[derive(Debug)]
    struct MockSource {
        source_value: f64,
        mock_signal: MockSignal,
    }

    impl Default for MockSource {
        fn default() -> Self {
            Self { 
                source_value: 0.0,
                mock_signal: MockSignal {},
            }
        }
    }

    impl Source for MockSource {
        fn get_source_term(&self, _t: f64, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
            self.source_value
        }

        fn positions(&self) -> Vec<(f64, f64, f64)> { vec![(0.0, 0.0, 0.0)] }
        fn signal(&self) -> &dyn Signal { &self.mock_signal }
    }
    


    #[test]
    fn test_update_wave_runs_without_panic_minimal_grid() {
        let grid_dim = 4; 
        let test_grid = create_test_grid(grid_dim, grid_dim, grid_dim);
        let mut wave_solver = NonlinearWave::new(&test_grid);

        let mut fields_shape: [usize; 4] = [0; 4];
        fields_shape[0] = TEST_N_FIELDS; 
        fields_shape[1] = test_grid.nx;
        fields_shape[2] = test_grid.ny;
        fields_shape[3] = test_grid.nz;
        
        let mut fields = Array4::<f64>::zeros(fields_shape.f());
        let prev_pressure = Array3::<f64>::zeros((test_grid.nx, test_grid.ny, test_grid.nz).f());

        let mock_medium = MockMedium::default();
        let mock_source = MockSource::default();

        let dt = 1e-5; 
        let t = 0.0;   

        AcousticWaveModel::update_wave(&mut wave_solver, &mut fields, &prev_pressure, &mock_source, &test_grid, &mock_medium, dt, t);
        
        assert!(!fields.iter().any(|&x| x.is_nan()), "Output contains NaN values");
    }

    #[test]
    fn test_update_wave_empty_grid_does_not_panic() {

        let test_grid = create_test_grid(4, 4, 4); 
        let mut wave_solver = NonlinearWave::new(&test_grid);

        let mut fields_shape: [usize; 4] = [0; 4];
        fields_shape[0] = TEST_N_FIELDS; 
        fields_shape[1] = test_grid.nx; 
        fields_shape[2] = test_grid.ny; 
        fields_shape[3] = test_grid.nz; 
        
        let mut fields = Array4::<f64>::zeros(fields_shape.f()); 
        let prev_pressure = Array3::<f64>::zeros((test_grid.nx, test_grid.ny, test_grid.nz).f());

        let mock_medium = MockMedium::default();
        let mock_source = MockSource::default();
        let dt = 1e-5;
        let t = 0.0;

        AcousticWaveModel::update_wave(&mut wave_solver, &mut fields, &prev_pressure, &mock_source, &test_grid, &mock_medium, dt, t);

    }
}
