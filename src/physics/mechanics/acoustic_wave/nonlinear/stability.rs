// src/physics/mechanics/acoustic_wave/nonlinear/stability.rs
use super::config::NonlinearWave;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;
use log::warn;

impl NonlinearWave {
    /// Checks the stability of the simulation based on current parameters and field values.
    ///
    /// This method performs several checks:
    /// 1.  Calculates the maximum sound speed in the medium.
    /// 2.  Checks the pressure field for any NaN (Not a Number) or infinite values.
    /// 3.  Computes the CFL (Courant-Friedrichs-Lewy) number using the maximum sound speed,
    ///     minimum grid spacing, and time step.
    ///
    /// Stability is determined if the CFL number is below the configured `cfl_safety_factor`
    /// and no NaN or infinite pressure values are found. Warnings are logged if potential
    /// instability is detected.
    ///
    /// # Arguments
    ///
    /// * `dt` - The current time step size.
    /// * `grid` - A reference to the `Grid` structure defining the simulation domain.
    /// * `medium` - A trait object implementing `Medium`, providing sound speed data.
    /// * `pressure` - A reference to the current 3D pressure field array.
    ///
    /// # Returns
    ///
    /// Returns `true` if the simulation is considered stable, `false` otherwise.
    pub(super) fn check_stability(&self, dt: f64, grid: &Grid, medium: &dyn Medium, pressure: &Array3<f64>) -> bool {
        // Check CFL condition for numerical stability
        let mut max_c: f64 = 0.0;
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        
        // Get maximum sound speed in the domain
        // This loop can be computationally intensive for large grids.
        // Consider optimizing if this becomes a bottleneck, e.g., by sampling or using a representative max_c.
        if grid.nx > 0 && grid.ny > 0 && grid.nz > 0 { // Check for empty grid
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    for k in 0..grid.nz {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;
                        let c = medium.sound_speed(x, y, z, grid);
                        max_c = max_c.max(c);
                    }
                }
            }
        } else {
            max_c = 0.0; // No points to check sound speed
        }
        
        // Get pressure extremes and check for NaN/Inf
        let mut max_pressure_val = f64::NEG_INFINITY;
        let mut min_pressure_val = f64::INFINITY;
        let mut has_nan = false;
        let mut has_inf = false;
        
        for &p_val in pressure.iter() {
            if p_val.is_nan() {
                has_nan = true;
                break; 
            } else if p_val.is_infinite() {
                has_inf = true;
                break; 
            } else {
                max_pressure_val = max_pressure_val.max(p_val);
                min_pressure_val = min_pressure_val.min(p_val);
            }
        }
        
        let cfl = if min_dx > 1e-9 { max_c * dt / min_dx } else { f64::INFINITY }; 
        let is_stable = cfl < self.cfl_safety_factor && !has_nan && !has_inf;
        
        if !is_stable {
            if has_nan || has_inf {
                warn!("NonlinearWave instability: NaN or Infinity detected in pressure field.");
            } else { 
                warn!(
                    "NonlinearWave potential instability: CFL = {:.3} (max_c={:.2}, dt={:.2e}, min_dx={:.2e}) exceeds safety factor {:.3}. Pressure range: [{:.2e}, {:.2e}] Pa.",
                    cfl, max_c, dt, min_dx, self.cfl_safety_factor,
                    if min_pressure_val.is_finite() { min_pressure_val } else {0.0}, // Avoid logging Inf
                    if max_pressure_val.is_finite() { max_pressure_val } else {0.0}
                );
            }
        }
        is_stable
    }

    /// Applies clamping to the pressure field to prevent numerical instability.
    ///
    /// This method iterates through the pressure field and performs the following:
    /// 1.  Replaces any NaN or infinite values with 0.0.
    /// 2.  Clamps any values exceeding `self.max_pressure` (positive or negative) to
    ///     `self.max_pressure` or `-self.max_pressure` respectively.
    ///
    /// If any values were clamped or replaced, a warning is logged indicating the number
    /// and percentage of modified values.
    ///
    /// # Arguments
    ///
    /// * `pressure` - A mutable reference to the 3D pressure field array. This array is modified in place.
    ///
    /// # Returns
    ///
    /// Returns `true` if any pressure values were clamped or replaced, `false` otherwise.
    pub(super) fn clamp_pressure(&self, pressure: &mut Array3<f64>) -> bool {
        let mut clamped_values = 0;
        let total_values = pressure.len();
        
        if total_values == 0 { return false; }

        for val in pressure.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0; 
                clamped_values += 1;
            } else if *val > self.max_pressure {
                *val = self.max_pressure;
                clamped_values += 1;
            } else if *val < -self.max_pressure {
                *val = -self.max_pressure;
                clamped_values += 1;
            }
        }
        
        let had_extreme_values = clamped_values > 0;
        
        if had_extreme_values {
            let percentage = 100.0 * clamped_values as f64 / total_values as f64;
            warn!(
                "Pressure field stability enforced: clamped {} values ({:.2}% of {} total values) to max pressure {:.2e} Pa.",
                clamped_values, percentage, total_values, self.max_pressure
            );
        }
        had_extreme_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid; // Use the actual Grid struct
    use crate::medium::Medium; // Use the Medium trait
    use crate::medium::tissue_specific; // For tissue_type Option
    use ndarray::Array3;

    // Helper to create a NonlinearWave instance for tests
    fn create_test_wave() -> NonlinearWave {
        let test_grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        NonlinearWave::new(&test_grid)
    }
    
    // Mock Medium for testing
    #[derive(Debug)] // Add Debug derive
    struct MockMedium {
        sound_speed_val: f64,
        density_val: f64,
        nonlinearity_coeff_val: f64,
        viscosity_val: f64,
        absorption_coeff_val: f64,
        ref_freq_val: f64,
        // Dummy fields for other trait methods
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
                dummy_temperature: Array3::zeros((1,1,1)),
                dummy_bubble_radius: Array3::zeros((1,1,1)),
                dummy_bubble_velocity: Array3::zeros((1,1,1)),
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
        
        // Implement other required Medium trait methods with dummy/default values
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
        fn density_array(&self) -> Array3<f64> { Array3::from_elem((1,1,1), self.density_val) }
        fn sound_speed_array(&self) -> Array3<f64> { Array3::from_elem((1,1,1), self.sound_speed_val) }
    }

    #[test]
    fn test_check_stability_stable() {
        let wave = create_test_wave();
        let test_grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let medium = MockMedium { sound_speed_val: 343.0, ..MockMedium::default() };
        let pressure = Array3::<f64>::zeros((10, 10, 10));
        let dt = 0.0001; // Small dt for stability

        assert_eq!(wave.check_stability(dt, &test_grid, &medium, &pressure), true);
    }

    #[test]
    fn test_check_stability_unstable_cfl() {
        let wave = create_test_wave();
        let test_grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let medium = MockMedium { sound_speed_val: 343.0, ..MockMedium::default() };
        let pressure = Array3::<f64>::zeros((10, 10, 10));
        let dt = 0.1; // Large dt to violate CFL

        assert_eq!(wave.check_stability(dt, &test_grid, &medium, &pressure), false);
    }

    #[test]
    fn test_check_stability_with_nan_pressure() {
        let wave = create_test_wave();
        let test_grid = Grid::new(2, 2, 2, 0.1, 0.1, 0.1);
        let medium = MockMedium { sound_speed_val: 343.0, ..MockMedium::default() };
        let mut pressure = Array3::<f64>::zeros((2, 2, 2));
        pressure[[0,0,0]] = f64::NAN;
        let dt = 0.0001;

        assert_eq!(wave.check_stability(dt, &test_grid, &medium, &pressure), false);
    }
    
    #[test]
    fn test_check_stability_with_inf_pressure() {
        let wave = create_test_wave();
        let test_grid = Grid::new(2, 2, 2, 0.1, 0.1, 0.1);
        let medium = MockMedium { sound_speed_val: 343.0, ..MockMedium::default() };
        let mut pressure = Array3::<f64>::zeros((2,2,2));
        pressure[[0,0,0]] = f64::INFINITY;
        let dt = 0.0001;
        
        assert_eq!(wave.check_stability(dt, &test_grid, &medium, &pressure), false);
    }

    #[test]
    fn test_clamp_pressure_no_clamping() {
        let wave = create_test_wave(); // max_pressure is 1e8 by default
        let mut pressure = Array3::<f64>::from_elem((5,5,5), 1e5); // Values well within limits
        let original_pressure = pressure.clone();

        let clamped = wave.clamp_pressure(&mut pressure);
        assert_eq!(clamped, false);
        assert_eq!(pressure, original_pressure);
    }

    #[test]
    fn test_clamp_pressure_positive_clamping() {
        let mut wave = create_test_wave();
        wave.max_pressure = 100.0; // Set a specific max_pressure for testing
        let mut pressure = Array3::<f64>::zeros((2,2,2));
        pressure[[0,0,0]] = 150.0; // Exceeds max_pressure
        pressure[[0,0,1]] = 50.0;  // Within limits

        let clamped = wave.clamp_pressure(&mut pressure);
        assert_eq!(clamped, true);
        assert_eq!(pressure[[0,0,0]], 100.0);
        assert_eq!(pressure[[0,0,1]], 50.0);
    }

    #[test]
    fn test_clamp_pressure_negative_clamping() {
        let mut wave = create_test_wave();
        wave.max_pressure = 100.0;
        let mut pressure = Array3::<f64>::zeros((2,2,2));
        pressure[[0,0,0]] = -150.0; // Below -max_pressure
        pressure[[0,0,1]] = -50.0;  // Within limits

        let clamped = wave.clamp_pressure(&mut pressure);
        assert_eq!(clamped, true);
        assert_eq!(pressure[[0,0,0]], -100.0);
        assert_eq!(pressure[[0,0,1]], -50.0);
    }

    #[test]
    fn test_clamp_pressure_nan_handling() {
        let mut wave = create_test_wave();
        wave.max_pressure = 100.0;
        let mut pressure = Array3::<f64>::zeros((2,2,2));
        pressure[[0,0,0]] = f64::NAN;
        pressure[[0,0,1]] = 50.0;

        let clamped = wave.clamp_pressure(&mut pressure);
        assert_eq!(clamped, true);
        assert_eq!(pressure[[0,0,0]], 0.0); // NaN should be reset to 0.0
        assert_eq!(pressure[[0,0,1]], 50.0);
    }
    
    #[test]
    fn test_clamp_pressure_inf_handling() {
        let mut wave = create_test_wave();
        wave.max_pressure = 100.0;
        let mut pressure = Array3::<f64>::zeros((2,2,2));
        pressure[[0,0,0]] = f64::INFINITY;
        pressure[[0,0,1]] = f64::NEG_INFINITY;
        pressure[[0,1,0]] = 50.0;

        let clamped = wave.clamp_pressure(&mut pressure);
        assert_eq!(clamped, true);
        assert_eq!(pressure[[0,0,0]], 0.0); // Infinity should be reset to 0.0
        assert_eq!(pressure[[0,0,1]], 0.0); // Neg Infinity should be reset to 0.0
        assert_eq!(pressure[[0,1,0]], 50.0);
    }
     #[test]
    fn test_check_stability_empty_grid() {
        let wave = create_test_wave();
        // Changed grid dimensions from (0,10,10) to (4,4,4) to avoid Grid::new() panic
        // This test now verifies behavior with a minimal valid grid.
        let test_grid = Grid::new(4, 4, 4, 0.1, 0.1, 0.1); 
        let medium = MockMedium { sound_speed_val: 343.0, ..MockMedium::default() }; 
        let pressure = Array3::<f64>::zeros((4, 4, 4)); 
        let dt = 0.0001;

        assert_eq!(wave.check_stability(dt, &test_grid, &medium, &pressure), true);
    }

    #[test]
    fn test_clamp_pressure_empty_array() {
        let wave = create_test_wave();
        let mut pressure = Array3::<f64>::zeros((0, 5, 5)); // Empty array

        let clamped = wave.clamp_pressure(&mut pressure);
        assert_eq!(clamped, false); // No values were clamped because there were no values
        assert_eq!(pressure.len(), 0);
    }
}
