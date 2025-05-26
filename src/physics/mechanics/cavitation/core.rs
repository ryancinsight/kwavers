// src/physics/mechanics/cavitation/core.rs
use super::model::{
    CavitationModel, // MAX_RADIUS_MODEL_DEFAULT, MIN_RADIUS_MODEL_DEFAULT, etc. are not directly used here
};
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array3, Zip}; 
use log::debug; 

impl CavitationModel {
    /// Advances the cavitation simulation by a single time step.
    ///
    /// This is the primary public method for updating the state of the `CavitationModel`.
    /// It orchestrates the various calculations involved in simulating bubble dynamics
    /// and their physical effects over one `dt`.
    ///
    /// Key steps performed:
    /// 1.  **Adaptive Time-Stepping**: Checks for extreme pressure values in the input acoustic
    ///     field `p`. If such pressures are detected, `actual_dt` is reduced from the base `dt`
    ///     to maintain numerical stability.
    /// 2.  **Bubble Acceleration**: Calls `self.calculate_second_derivative()` to compute the
    ///     acceleration of the bubble walls based on the current state and driving pressures.
    ///     The calculated accelerations are clamped to `LOCAL_MAX_ACCELERATION`.
    /// 3.  **Bubble Dynamics Update**: Calls `self.update_bubble_dynamics()` to integrate the
    ///     accelerations over `actual_dt`, updating bubble radii and velocities.
    /// 4.  **Local Clamping**: Applies further clamping to bubble radii and velocities based on
    ///     `LOCAL_MIN_RADIUS`, `LOCAL_MAX_RADIUS`, and `LOCAL_MAX_VELOCITY` defined within this method.
    ///     It also handles cases where bubbles hit these limits (e.g., by zeroing velocity).
    /// 5.  **Acoustic and Optical Effects**: Calls `self.calculate_acoustic_effects()` to compute
    ///     the feedback of bubble activity on the acoustic pressure field (`p_update`) and to
    ///     calculate light emission (sonoluminescence).
    ///
    /// # Arguments
    ///
    /// * `self` - A mutable reference to the `CavitationModel` instance, allowing its state
    ///   (radius, velocity, temperature, etc.) to be updated.
    /// * `p_update` - A mutable reference to a 3D array representing the acoustic pressure field.
    ///   This field will be modified to include the pressure changes caused by bubble oscillations.
    /// * `p` - A reference to the current 3D acoustic pressure field that drives the bubble dynamics.
    /// * `grid` - A reference to the `Grid` structure defining the simulation domain.
    /// * `dt` - The base time step for this update (seconds). The actual time step used internally
    ///   may be smaller if adaptive time-stepping is triggered.
    /// * `medium` - A trait object implementing `Medium`, providing material properties.
    /// * `frequency` - The driving acoustic frequency (Hz), used in some sub-calculations (e.g., `calculate_second_derivative`).
    ///
    /// # Returns
    ///
    /// An `Array3<f64>` representing the light emission power density (W/m^3) from sonoluminescence
    /// at each grid point.
    ///
    /// # Modifies
    ///
    /// * `self`: Updates internal state fields like `radius`, `velocity`, `prev_velocity`, `temperature`, `d2r_dt2`, and scattering arrays.
    /// * `p_update`: Modifies the input pressure field to account for bubble-induced pressure changes.
    pub fn update_cavitation(
        &mut self,
        p_update: &mut Array3<f64>,
        p: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
        frequency: f64,
    ) -> Array3<f64> {
        // Safety parameters for stability, kept local as in original update_cavitation
        // These might be more specific than the general model defaults.
        const LOCAL_MAX_RADIUS: f64 = 2.0e-4; 
        const LOCAL_MIN_RADIUS: f64 = 1.0e-10;
        const LOCAL_MAX_VELOCITY: f64 = 1.0e3; 
        const LOCAL_MAX_ACCELERATION: f64 = 1.0e12;
        const LOCAL_MAX_PRESSURE: f64 = 5.0e7; 
        
        let mut has_extreme_pressure = false;
        let max_p_abs = p.iter().fold(0.0, |max_abs: f64, &val| max_abs.max(val.abs()));
        if max_p_abs > LOCAL_MAX_PRESSURE {
            has_extreme_pressure = true;
            debug!("Extreme pressure detected: {:.2e} Pa, using adaptive time step", max_p_abs);
        }
        
        let actual_dt = if has_extreme_pressure {
            dt * (LOCAL_MAX_PRESSURE / max_p_abs.max(1.0)).min(0.5).max(0.01) 
        } else {
            dt
        };
        
        self.calculate_second_derivative(p, grid, medium, frequency, actual_dt);
        
        for val in self.d2r_dt2.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            } else {
                *val = val.clamp(-LOCAL_MAX_ACCELERATION, LOCAL_MAX_ACCELERATION);
            }
        }
        
        self.update_bubble_dynamics(actual_dt); 

        Zip::from(&mut self.radius)
            .and(&mut self.velocity)
            .for_each(|r, v| {
                *r = r.clamp(LOCAL_MIN_RADIUS, LOCAL_MAX_RADIUS);
                *v = v.clamp(-LOCAL_MAX_VELOCITY, LOCAL_MAX_VELOCITY);
                
                if (*r == LOCAL_MIN_RADIUS && *v < 0.0) || (*r == LOCAL_MAX_RADIUS && *v > 0.0) {
                    *v = 0.0;
                }
            });
        
        let light_emission = self.calculate_acoustic_effects(p_update, p, grid, medium, has_extreme_pressure);
        
        light_emission
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
            let default_dim = (4,4,4); // Match common test grid size from core.rs tests for NonlinearWave
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
    fn test_update_cavitation_runs_without_panic() {
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

        let _light_emission = model.update_cavitation(
            &mut p_update,
            &p_acoustic,
            &test_grid,
            dt,
            &mock_medium,
            frequency,
        );

        // Basic check: ensure no NaNs were introduced if inputs were not extreme
        assert!(!p_update.iter().any(|&x| x.is_nan()), "p_update contains NaN values");
        assert!(!model.radius.iter().any(|&x| x.is_nan()), "model.radius contains NaN values");
        assert!(!model.velocity.iter().any(|&x| x.is_nan()), "model.velocity contains NaN values");
        assert!(!model.temperature.iter().any(|&x| x.is_nan()), "model.temperature contains NaN values");
    }
}
