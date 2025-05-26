// src/physics/mechanics/cavitation/dynamics.rs
use super::model::{
    CavitationModel, MIN_RADIUS_MODEL_DEFAULT, MAX_RADIUS_MODEL_DEFAULT,
    MAX_VELOCITY_MODEL_DEFAULT, MAX_ACCELERATION_MODEL_DEFAULT,
};
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;
use log::debug;

impl CavitationModel {
    /// Calculates the second time derivative of the bubble radius (acceleration) for each bubble in the grid.
    ///
    /// This method implements the core physics of bubble dynamics, often based on a form of the
    /// Rayleigh-Plesset equation or similar models. It considers factors like ambient pressure,
    /// surface tension, viscosity, and the driving acoustic pressure. The calculated accelerations
    /// are stored in `self.d2r_dt2`.
    ///
    /// Various safety checks and clamping mechanisms are applied to ensure numerical stability,
    /// such as clamping pressure values, limiting ratios, and capping acceleration.
    ///
    /// # Arguments
    ///
    /// * `p` - A reference to the 3D array of acoustic pressures at each grid point.
    /// * `grid` - A reference to the `Grid` structure defining the simulation domain.
    /// * `medium` - A trait object implementing `Medium`, providing material properties.
    /// * `_frequency` - The driving acoustic frequency (currently unused in this implementation but kept for signature consistency).
    /// * `_dt` - The time step (currently unused in this specific calculation of acceleration but kept for signature consistency).
    ///
    /// # Modifies
    ///
    /// * `self.d2r_dt2`: This field is updated with the newly calculated bubble accelerations.
    pub(crate) fn calculate_second_derivative(
        &mut self,
        p: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        _frequency: f64, 
        _dt: f64,       
    ) {
        self.d2r_dt2.fill(0.0);
        
        for i in 0..grid.nx {
            let x = i as f64 * grid.dx;
            for j in 0..grid.ny {
                let y = j as f64 * grid.dy;
                for k in 0..grid.nz {
                    let z = k as f64 * grid.dz;
                    
                    let rho = medium.density(x, y, z, grid);
                    let mu = medium.viscosity(x, y, z, grid);
                    let sigma = medium.surface_tension(x, y, z, grid);
                    let p0 = medium.ambient_pressure(x, y, z, grid);
                    let pv = medium.vapor_pressure(x, y, z, grid);
                    let gamma = medium.polytropic_index(x, y, z, grid);
                    let kappa = medium.thermal_conductivity(x, y, z, grid);
                    let dg = medium.gas_diffusion_coefficient(x, y, z, grid);
                    let medium_temp = medium.temperature()[[i, j, k]]; 
                    
                    let r = self.radius[[i, j, k]];
                    let v = self.velocity[[i, j, k]];
                    let t_bubble = self.temperature[[i, j, k]];
                    let mut p_val = p[[i, j, k]];
                    
                    if !p_val.is_finite() || p_val.abs() > 1.0e9 {
                        p_val = p_val.clamp(-1.0e9, 1.0e9);
                        if !p_val.is_finite() { p_val = 0.0; }
                    }
                    
                    let r_clamped = r.max(MIN_RADIUS_MODEL_DEFAULT); 
                    let r0 = 10e-6; 
                    
                    let r_ratio = (r0 / r_clamped).min(1.0e3); 
                    let p_gas = (p0 + 2.0 * sigma / r0 - pv) * r_ratio.powf(3.0 * gamma);
                    
                    let r_inv = 1.0 / r_clamped;
                    let viscous_term = 4.0 * mu * v * r_inv;
                    let surface_term = 2.0 * sigma * r_inv;
                    
                    let temp_diff = (t_bubble - medium_temp).clamp(-100.0, 100.0); 
                    let thermal_damping = 3.0 * gamma * kappa * temp_diff * r_inv * r_inv;
                    
                    let p_diff = (p0 - p_gas).clamp(-1.0e9, 1.0e9); 
                    let diffusion_term = dg * p_diff * r_inv / rho;
                    
                    let pressure_diff = (p_gas + pv - p_val).clamp(-1.0e9, 1.0e9);
                    let rhs = (pressure_diff - viscous_term - surface_term - thermal_damping - diffusion_term) / rho;
                    let velocity_term = 1.5 * v.powi(2).min(1.0e6); 
                    
                    let d2r = (rhs - velocity_term) * r_inv;
                    
                    self.d2r_dt2[[i, j, k]] = if d2r.is_finite() { 
                        d2r.clamp(-MAX_ACCELERATION_MODEL_DEFAULT, MAX_ACCELERATION_MODEL_DEFAULT) 
                    } else { 0.0 };
                }
            }
        }
        
        for val in self.d2r_dt2.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
                debug!("NaN/Inf found in bubble acceleration, reset to 0");
            }
        }
    }
    
    /// Updates the bubble radius and velocity for each bubble based on the calculated accelerations.
    ///
    /// This method uses a simple time integration scheme (e.g., forward Euler or similar)
    /// to update `self.radius` and `self.velocity` using the accelerations stored in `self.d2r_dt2`.
    /// It may employ an adaptive time step (`actual_dt`) if high accelerations are detected,
    /// reducing `dt` to maintain stability.
    ///
    /// After updating, it applies several safety checks:
    /// - Clamps radii and velocities to predefined model defaults (e.g., `MIN_RADIUS_MODEL_DEFAULT`).
    /// - Resets NaN or infinite values for radius and velocity to sensible defaults.
    /// - Applies additional velocity damping if acceleration remains extremely high.
    ///
    /// # Arguments
    ///
    /// * `dt` - The base time step for the simulation. The actual time step used for integration
    ///   might be smaller if adaptive time-stepping is triggered.
    ///
    /// # Modifies
    ///
    /// * `self.radius`: Updated with new bubble radii.
    /// * `self.velocity`: Updated with new bubble wall velocities.
    /// * `self.prev_velocity`: Stores the velocities from before this update.
    pub(crate) fn update_bubble_dynamics(&mut self, dt: f64) {
        let mut new_radius = self.radius.clone();
        let mut new_velocity = Array3::zeros(self.velocity.dim());
        
        let actual_dt = if self.d2r_dt2.iter().any(|&v| v.abs() > 1.0e10) {
            debug!("Using reduced time step for bubble dynamics due to high accelerations");
            dt * 0.1
        } else {
            dt
        };
        
        for ((i, j, k), &d2r) in self.d2r_dt2.indexed_iter() {
            self.prev_velocity[[i, j, k]] = self.velocity[[i, j, k]];
            
            let new_vel = self.velocity[[i, j, k]] + d2r * actual_dt;
            new_velocity[[i, j, k]] = new_vel.clamp(-MAX_VELOCITY_MODEL_DEFAULT, MAX_VELOCITY_MODEL_DEFAULT);
            
            // Velocity used here should be the new_velocity for a more accurate integration step (e.g. semi-implicit Euler)
            // Original code uses self.velocity which is old velocity. Let's stick to original for now.
            let new_rad = self.radius[[i, j, k]] + self.velocity[[i, j, k]] * actual_dt; 
            new_radius[[i, j, k]] = new_rad.clamp(MIN_RADIUS_MODEL_DEFAULT, MAX_RADIUS_MODEL_DEFAULT);
        }
        
        self.velocity = new_velocity;
        self.radius = new_radius;
        
        let mut nan_count = 0;
        let mut clamped_count = 0;
        
        for ((r, v), a) in self.radius.iter_mut()
            .zip(self.velocity.iter_mut())
            .zip(self.d2r_dt2.iter())
        {
            if r.is_nan() || r.is_infinite() {
                *r = 5.0e-6; 
                nan_count += 1;
            }
            
            if v.is_nan() || v.is_infinite() {
                *v = 0.0; 
                nan_count += 1;
            }
            
            if *r > MAX_RADIUS_MODEL_DEFAULT { 
                *r = MAX_RADIUS_MODEL_DEFAULT;
                *v = v.min(0.0); 
                clamped_count += 1;
            } else if *r < MIN_RADIUS_MODEL_DEFAULT { 
                *r = MIN_RADIUS_MODEL_DEFAULT;
                *v = v.max(0.0); 
                clamped_count += 1;
            }
            
            if v.abs() > MAX_VELOCITY_MODEL_DEFAULT { 
                *v = v.signum() * MAX_VELOCITY_MODEL_DEFAULT;
                clamped_count += 1;
            }
            
            if a.abs() > 1.0e10 { 
                *v *= 0.9; 
            }
        }
        
        if nan_count > 0 || clamped_count > 0 {
            debug!(
                "Bubble dynamics stabilized: {} NaN/Inf reset, {} values clamped to physical range",
                nan_count,
                clamped_count
            );
        }
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
        dummy_bubble_radius: Array3<f64>, // For trait completeness
        dummy_bubble_velocity: Array3<f64>, // For trait completeness
    }

    impl Default for MockMedium {
        fn default() -> Self {
            let default_dim = (1,1,1); // Minimal dimension for dummy arrays
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
    fn test_calculate_second_derivative_basic() {
        let grid_dims = (1, 1, 1); // Simplest grid for single point test
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut model = CavitationModel::new(&test_grid, 5e-6);
        
        model.radius[[0,0,0]] = 5e-6;
        model.velocity[[0,0,0]] = 1.0; 
        model.temperature[[0,0,0]] = 300.0;

        let mut pressure = Array3::zeros(grid_dims.f());
        pressure[[0,0,0]] = 1.5e5; 

        let mock_medium = MockMedium { 
            medium_temperature_val: Array3::from_elem(grid_dims.f(), 293.15),
            ..MockMedium::default() 
        };
        
        let initial_d2r_dt2_val = model.d2r_dt2[[0,0,0]]; // Should be 0.0

        model.calculate_second_derivative(&pressure, &test_grid, &mock_medium, 1e6, 1e-7);

        assert_ne!(model.d2r_dt2[[0,0,0]], initial_d2r_dt2_val, "d2r_dt2 should be updated from zero");
        assert!(model.d2r_dt2[[0,0,0]].is_finite(), "d2r_dt2 should be finite");
        // A more specific assertion would require manual calculation of Rayleigh-Plesset,
        // which is complex. For now, checking it changed and is finite is a basic sanity check.
    }

    #[test]
    fn test_update_bubble_dynamics_basic() {
        let grid_dims = (1,1,1);
        let test_grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut model = CavitationModel::new(&test_grid, 5e-6);
        
        let initial_radius = 5e-6;
        let initial_velocity = 0.5;
        let test_acceleration = 1e10; // A large, positive acceleration
        let dt = 1e-7;

        model.radius[[0,0,0]] = initial_radius;
        model.velocity[[0,0,0]] = initial_velocity;
        model.d2r_dt2[[0,0,0]] = test_acceleration; // Set a known acceleration

        model.update_bubble_dynamics(dt);

        let expected_new_velocity_no_clamp = initial_velocity + test_acceleration * dt;
        let expected_new_velocity = expected_new_velocity_no_clamp.clamp(-MAX_VELOCITY_MODEL_DEFAULT, MAX_VELOCITY_MODEL_DEFAULT);
        
        // Original code uses old velocity for radius update: R_new = R_old + V_old * dt
        // Corrected semi-implicit Euler would be: R_new = R_old + V_new * dt
        // Sticking to original for now:
        let expected_new_radius_no_clamp = initial_radius + initial_velocity * dt; 
        let expected_new_radius = expected_new_radius_no_clamp.clamp(MIN_RADIUS_MODEL_DEFAULT, MAX_RADIUS_MODEL_DEFAULT);


        assert_eq!(model.prev_velocity[[0,0,0]], initial_velocity, "prev_velocity not stored correctly");
        assert!((model.velocity[[0,0,0]] - expected_new_velocity).abs() < 1e-9, "Velocity not updated as expected. Got {}, expected {}", model.velocity[[0,0,0]], expected_new_velocity);
        assert!((model.radius[[0,0,0]] - expected_new_radius).abs() < 1e-9, "Radius not updated as expected. Got {}, expected {}", model.radius[[0,0,0]], expected_new_radius);
    }
}
