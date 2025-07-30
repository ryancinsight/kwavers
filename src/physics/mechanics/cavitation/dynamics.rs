// src/physics/mechanics/cavitation/dynamics.rs
use super::model::{
    LegacyCavitationModel as CavitationModel, MIN_RADIUS_MODEL_DEFAULT, MAX_RADIUS_MODEL_DEFAULT,
    MAX_VELOCITY_MODEL_DEFAULT, MAX_ACCELERATION_MODEL_DEFAULT,
};
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array3;
use log::debug;
use std::f64::consts::PI;

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
        frequency: f64, 
        dt: f64,       
    ) {
        self.d2r_dt2.fill(0.0);
        
        // Calculate acoustic wavelength for compressibility effects
        let c0 = medium.sound_speed(0.0, 0.0, 0.0, grid);
        let wavelength = c0 / frequency.max(1.0);
        
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
                    let c = medium.sound_speed(x, y, z, grid);
                    
                    let r = self.radius[[i, j, k]];
                    let v = self.velocity[[i, j, k]];
                    let t_bubble = self.temperature[[i, j, k]];
                    let mut p_val = p[[i, j, k]];
                    
                    if !p_val.is_finite() || p_val.abs() > 1.0e9 {
                        p_val = p_val.clamp(-1.0e9, 1.0e9);
                        if !p_val.is_finite() { p_val = 0.0; }
                    }
                    
                    let r_clamped = r.max(MIN_RADIUS_MODEL_DEFAULT); 
                    let r0 = self.r0[[i, j, k]].max(1e-9); // Use actual equilibrium radius
                    
                    // Van der Waals hard-core correction for extreme compression
                    let b_factor = 8.65e-6; // m³/mol for air
                    let n_gas = self.n_gas[[i, j, k]];
                    let vh = 4.0 * PI * r0.powi(3) / 3.0;
                    let v_current = 4.0 * PI * r_clamped.powi(3) / 3.0;
                    
                    // Modified polytropic relation with hard core
                    let volume_ratio = if n_gas > 0.0 {
                        ((vh - n_gas * b_factor) / (v_current - n_gas * b_factor)).max(0.001)
                    } else {
                        (r0 / r_clamped).powi(3)
                    };
                    
                    // Effective polytropic index based on Peclet number
                    let thermal_diffusivity = kappa / (rho * 4200.0); // Approximate specific heat
                    let peclet = (r_clamped * v.abs()) / thermal_diffusivity.max(1e-10);
                    let gamma_eff = 1.0 + (gamma - 1.0) / (1.0 + 10.0 / peclet.max(0.1));
                    
                    let p_gas = (p0 + 2.0 * sigma / r0 - pv) * volume_ratio.powf(gamma_eff);
                    
                    let r_inv = 1.0 / r_clamped;
                    let viscous_term = 4.0 * mu * v * r_inv;
                    let surface_term = 2.0 * sigma * r_inv;
                    
                    // Enhanced thermal damping with radiation
                    let temp_diff = (t_bubble - medium_temp).clamp(-1000.0, 10000.0); 
                    let thermal_damping = 3.0 * gamma_eff * kappa * temp_diff * r_inv * r_inv;
                    
                    // Radiation damping for small bubbles
                    let radiation_damping = if r_clamped < wavelength / (2.0 * PI) {
                        2.0 * PI * frequency * r_clamped / c
                    } else {
                        0.0
                    };
                    
                    let p_diff = (p0 - p_gas).clamp(-1.0e9, 1.0e9); 
                    let diffusion_term = dg * p_diff * r_inv / rho;
                    
                    // Liquid pressure at bubble wall (including compressibility)
                    let p_liquid = p0 + p_val - pv;
                    
                    // Keller-Miksis formulation for compressibility
                    let mach = v / c;
                    let compressibility_factor = 1.0 - mach;
                    
                    // Pressure gradient term for Keller-Miksis
                    let dp_dt = if frequency > 0.0 {
                        // Estimate pressure time derivative from acoustic field
                        -p_val * (2.0 * PI * frequency) * (2.0 * PI * frequency * dt).sin()
                    } else {
                        0.0
                    };
                    
                    let pressure_diff = (p_gas - p_liquid).clamp(-1.0e9, 1.0e9);
                    
                    // Keller-Miksis equation
                    let numerator = (pressure_diff - viscous_term - surface_term - thermal_damping 
                                    - radiation_damping * rho * c * v - diffusion_term) / rho
                                    + r_clamped * dp_dt / (rho * c);
                    
                    let denominator = r_clamped * compressibility_factor + r_clamped * r_clamped * mach / c;
                    
                    let velocity_correction = 1.5 * v.powi(2) * (1.0 - mach / 3.0);
                    
                    let d2r = if denominator.abs() > 1e-10 {
                        (numerator - velocity_correction) / denominator
                    } else {
                        0.0
                    };
                    
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
            
            // Update radius using the NEW velocity for Semi-Implicit Euler integration: r_new = r_old + v_new * dt
            let new_rad = self.radius[[i, j, k]] + new_velocity[[i, j, k]] * actual_dt; 
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
    
    /// Updates the temperature of cavitation bubbles based on compression and shock heating.
    ///
    /// This method implements:
    /// - Adiabatic compression heating
    /// - Shock wave heating during violent collapse
    /// - Heat transfer to surrounding liquid
    /// - Tracking of maximum temperatures for sonoluminescence
    ///
    /// # Arguments
    ///
    /// * `grid` - Reference to the simulation grid
    /// * `medium` - Reference to the medium properties
    /// * `dt` - Time step
    pub(crate) fn update_temperature(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        let shape = self.radius.shape();
        
        for i in 0..shape[0] {
            let x = i as f64 * grid.dx;
            for j in 0..shape[1] {
                let y = j as f64 * grid.dy;
                for k in 0..shape[2] {
                    let z = k as f64 * grid.dz;
                    
                    let r = self.radius[[i, j, k]];
                    let v = self.velocity[[i, j, k]];
                    let r0 = self.r0[[i, j, k]];
                    let t_current = self.temperature[[i, j, k]];
                    let t_ambient = medium.temperature()[[i, j, k]];
                    
                    // Skip if bubble is too small
                    if r < MIN_RADIUS_MODEL_DEFAULT {
                        continue;
                    }
                    
                    // Calculate compression ratio
                    let compression_ratio = r0 / r;
                    self.max_compression[[i, j, k]] = self.max_compression[[i, j, k]].max(compression_ratio);
                    
                    // Get medium properties
                    let rho = medium.density(x, y, z, grid);
                    let c = medium.sound_speed(x, y, z, grid);
                    let kappa = medium.thermal_conductivity(x, y, z, grid);
                    let gamma = medium.polytropic_index(x, y, z, grid);
                    
                    // Calculate effective polytropic index based on Peclet number
                    let thermal_diffusivity = kappa / (rho * 4200.0); // Approximate cp
                    let peclet = (r * v.abs()) / thermal_diffusivity.max(1e-10);
                    let gamma_eff = 1.0 + (gamma - 1.0) / (1.0 + 10.0 / peclet.max(0.1));
                    
                    // Adiabatic compression temperature
                    let t_adiabatic = t_ambient * compression_ratio.powf(gamma_eff - 1.0);
                    
                    // Shock heating during violent collapse
                    let mach = v.abs() / c;
                    let t_shock = if v < 0.0 && mach > 0.3 {
                        // Rankine-Hugoniot shock temperature jump
                        let shock_factor = 2.0 * gamma_eff * mach.powi(2) - (gamma_eff - 1.0);
                        t_current * shock_factor / (gamma_eff + 1.0)
                    } else {
                        0.0
                    };
                    
                    // Heat transfer to liquid
                    let h = kappa / r; // Heat transfer coefficient
                    let area = 4.0 * PI * r * r;
                    let volume = 4.0 * PI * r.powi(3) / 3.0;
                    let mass = self.n_gas[[i, j, k]] * 0.029 / 6.022e23; // Air molecular mass
                    let cv = 717.0; // J/(kg·K) for air
                    
                    // PdV work
                    let p_internal = self.pressure_internal[[i, j, k]];
                    let pdv_work = -p_internal * 4.0 * PI * r * r * v;
                    
                    // Heat transfer rate
                    let q_transfer = h * area * (t_ambient - t_current);
                    
                    // Temperature change
                    let dt_adiabatic = (t_adiabatic - t_current) / dt;
                    let dt_shock = t_shock / dt;
                    let dt_work = pdv_work / (mass * cv).max(1e-10);
                    let dt_transfer = q_transfer / (mass * cv).max(1e-10);
                    
                    // Update temperature
                    let new_temp = t_current + (dt_adiabatic + dt_shock + dt_work + dt_transfer) * dt;
                    self.temperature[[i, j, k]] = new_temp.max(t_ambient).min(50000.0); // Cap at 50,000K
                    
                    // Track maximum temperature
                    self.max_temperature[[i, j, k]] = self.max_temperature[[i, j, k]].max(new_temp);
                }
            }
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

        // Default implementations for new elastic methods
        fn lame_lambda(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.0 }
        fn lame_mu(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { 0.0 }
        fn lame_lambda_array(&self) -> Array3<f64> { Array3::zeros(self.medium_temperature_val.dim()) }
        fn lame_mu_array(&self) -> Array3<f64> { Array3::zeros(self.medium_temperature_val.dim()) }
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
        
        // Expected radius update using Semi-Implicit Euler: R_new = R_old + V_new * dt
        let expected_new_radius_no_clamp = initial_radius + expected_new_velocity * dt; 
        let expected_new_radius = expected_new_radius_no_clamp.clamp(MIN_RADIUS_MODEL_DEFAULT, MAX_RADIUS_MODEL_DEFAULT);


        assert_eq!(model.prev_velocity[[0,0,0]], initial_velocity, "prev_velocity not stored correctly");
        assert!((model.velocity[[0,0,0]] - expected_new_velocity).abs() < 1e-9, "Velocity not updated as expected. Got {}, expected {}", model.velocity[[0,0,0]], expected_new_velocity);
        assert!((model.radius[[0,0,0]] - expected_new_radius).abs() < 1e-9, "Radius not updated as expected. Got {}, expected {}", model.radius[[0,0,0]], expected_new_radius);
    }
}
