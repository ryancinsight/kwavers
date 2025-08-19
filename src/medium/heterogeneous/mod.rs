// src/medium/heterogeneous/mod.rs
use crate::grid::Grid;
use crate::medium::{absorption, absorption::power_law_absorption, Medium};
use log::debug;
use ndarray::{Array3, Zip};

pub mod tissue;

#[derive(Debug, Clone)]
pub struct HeterogeneousMedium {
    pub density: Array3<f64>,
    pub sound_speed: Array3<f64>,
    pub viscosity: Array3<f64>,
    pub surface_tension: Array3<f64>,
    pub ambient_pressure: f64,
    pub vapor_pressure: Array3<f64>,
    pub polytropic_index: Array3<f64>,
    pub specific_heat: Array3<f64>,
    pub thermal_conductivity: Array3<f64>,
    pub thermal_expansion: Array3<f64>,
    pub gas_diffusion_coeff: Array3<f64>,
    pub thermal_diffusivity: Array3<f64>,
    pub mu_a: Array3<f64>,
    pub mu_s_prime: Array3<f64>,
    pub temperature: Array3<f64>,
    pub bubble_radius: Array3<f64>,
    pub bubble_velocity: Array3<f64>,
    pub alpha0: Array3<f64>,
    pub delta: Array3<f64>,
    pub b_a: Array3<f64>,
    pub reference_frequency: f64, // Added
    // New fields for viscoelastic properties
    pub shear_sound_speed: Array3<f64>,
    pub shear_viscosity_coeff: Array3<f64>,
    pub bulk_viscosity_coeff: Array3<f64>,
    // New fields for elastic properties
    pub lame_lambda: Array3<f64>,
    pub lame_mu: Array3<f64>,
}

impl HeterogeneousMedium {
    /// Helper to get grid indices with fallback to boundary values
    #[inline]
    fn get_indices(&self, x: f64, y: f64, z: f64, grid: &Grid) -> (usize, usize, usize) {
        grid.position_to_indices(x, y, z).unwrap_or_else(|| {
            // Clamp to grid boundaries if out of bounds
            let ix = ((x / grid.dx).floor() as usize).min(grid.nx - 1);
            let iy = ((y / grid.dy).floor() as usize).min(grid.ny - 1);
            let iz = ((z / grid.dz).floor() as usize).min(grid.nz - 1);
            (ix, iy, iz)
        })
    }

    pub fn tissue(grid: &Grid) -> Self {
        let density = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1050.0);
        let sound_speed = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1540.0);
        let viscosity = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2.5e-3);
        let surface_tension = Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.06);
        let ambient_pressure = 1.013e5;
        let vapor_pressure = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2.338e3);
        let polytropic_index = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.4);
        let specific_heat = Array3::from_elem((grid.nx, grid.ny, grid.nz), 3630.0);
        let thermal_conductivity = Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.52);
        let thermal_expansion = Array3::from_elem((grid.nx, grid.ny, grid.nz), 3.0e-4);
        let gas_diffusion_coeff = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.8e-9);
        let thermal_diffusivity = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.35e-7);
        let mu_a = Array3::from_elem((grid.nx, grid.ny, grid.nz), 10.0);
        let mu_s_prime = Array3::from_elem((grid.nx, grid.ny, grid.nz), 100.0);
        let b_a = Array3::from_elem((grid.nx, grid.ny, grid.nz), 7.0);
        let reference_frequency = 180000.0; // Default for tissue simulation

        let temperature = Array3::from_elem((grid.nx, grid.ny, grid.nz), 310.15); // 37°C
        let bubble_radius = Array3::from_elem((grid.nx, grid.ny, grid.nz), 10e-6);
        let bubble_velocity = Array3::zeros((grid.nx, grid.ny, grid.nz));

        let alpha0 = Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.5);
        let delta = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.1);

        // Initialize viscoelastic fields with tissue-appropriate values
        // Shear wave speed in soft tissue: typically 1-10 m/s
        let shear_sound_speed = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(i, j, _k)| {
            // Vary shear speed based on position to simulate tissue heterogeneity
            let base_speed = 3.0; // m/s (typical for muscle tissue)
            let variation = 0.5 * ((i as f64 / grid.nx as f64).sin() + 
                                  (j as f64 / grid.ny as f64).cos());
            (base_speed + variation).max(1.0).min(8.0)
        });
        
        // Shear viscosity coefficient for soft tissue: 0.1-10 Pa·s
        let shear_viscosity_coeff = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(i, j, k)| {
            // Higher viscosity near boundaries, lower in center
            let center_x = grid.nx as f64 / 2.0;
            let center_y = grid.ny as f64 / 2.0;
            let center_z = grid.nz as f64 / 2.0;
            let dist_from_center = ((i as f64 - center_x).powi(2) + 
                                   (j as f64 - center_y).powi(2) + 
                                   (k as f64 - center_z).powi(2)).sqrt();
            let max_dist = (center_x.powi(2) + center_y.powi(2) + center_z.powi(2)).sqrt();
            let normalized_dist = (dist_from_center / max_dist).min(1.0);
            
            // Base viscosity + position-dependent variation
            1.0 + 2.0 * normalized_dist // Range: 1.0-3.0 Pa·s
        });
        
        // Bulk viscosity coefficient: typically 2-5x shear viscosity
        let bulk_viscosity_coeff = shear_viscosity_coeff.mapv(|shear_visc| shear_visc * 3.0);

        // Initialize new elastic fields (default to fluid-like: mu=0, lambda=K)
        // K = rho * c^2. Using default density and sound_speed from above.
        let default_density: f64 = 1050.0;
        let default_sound_speed: f64 = 1540.0;
        let default_bulk_modulus = default_density * default_sound_speed.powi(2); // lambda for a fluid
        let lame_lambda = Array3::from_elem((grid.nx, grid.ny, grid.nz), default_bulk_modulus);
        let lame_mu = Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.0); // mu is 0 for ideal fluid

        debug!(
            "Initialized HeterogeneousMedium: grid {}x{}x{}, freq = {:.2e}",
            grid.nx, grid.ny, grid.nz, reference_frequency
        );
        Self {
            density,
            sound_speed,
            viscosity,
            surface_tension,
            ambient_pressure,
            vapor_pressure,
            polytropic_index,
            specific_heat,
            thermal_conductivity,
            thermal_expansion,
            gas_diffusion_coeff,
            thermal_diffusivity,
            mu_a,
            mu_s_prime,
            temperature,
            bubble_radius,
            bubble_velocity,
            alpha0,
            delta,
            b_a,
            reference_frequency,
            shear_sound_speed,
            shear_viscosity_coeff,
            bulk_viscosity_coeff,
            lame_lambda,
            lame_mu,
        }
    }
}

impl Medium for HeterogeneousMedium {
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.lame_lambda[[ix, iy, iz]]
    }

    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.lame_mu[[ix, iy, iz]]
    }

    fn lame_lambda_array(&self) -> Array3<f64> {
        self.lame_lambda.clone()
    }

    fn lame_mu_array(&self) -> Array3<f64> {
        self.lame_mu.clone()
    }

    fn density(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.density[[ix, iy, iz]].max(1.0)
    }
    fn sound_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.sound_speed[[ix, iy, iz]].max(100.0)
    }
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.viscosity[[ix, iy, iz]].max(1e-6)
    }
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.surface_tension[[ix, iy, iz]].max(0.01)
    }
    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.ambient_pressure
    }
    fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.vapor_pressure[[ix, iy, iz]].max(1.0)
    }
    fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.polytropic_index[[ix, iy, iz]]
    }
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.specific_heat[[ix, iy, iz]]
    }
    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.thermal_conductivity[[ix, iy, iz]].max(0.1)
    }
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        let t = self.temperature[[ix, iy, iz]];
        let alpha0 = self.alpha0[[ix, iy, iz]];
        let delta = self.delta[[ix, iy, iz]];
        power_law_absorption::power_law_absorption_coefficient(frequency, alpha0, delta)
            + absorption::absorption_coefficient(frequency, t, Some(self.bubble_radius[[ix, iy, iz]]))
    }
    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.thermal_expansion[[ix, iy, iz]].max(1e-6)
    }
    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.gas_diffusion_coeff[[ix, iy, iz]].max(1e-10)
    }
    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Calculate thermal diffusivity using thermal conductivity, density, and specific heat
        let rho = self.density(x, y, z, grid);
        let cp = self.specific_heat(x, y, z, grid);
        let k = self.thermal_conductivity(x, y, z, grid);
        (k / (rho * cp)).max(1e-8)
    }
    fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.b_a[[ix, iy, iz]].max(0.0)
    }
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.mu_a[[ix, iy, iz]].max(0.1)
    }
    fn optical_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (ix, iy, iz) = self.get_indices(x, y, z, grid);
        self.mu_s_prime[[ix, iy, iz]].max(1.0)
    }
    fn reference_frequency(&self) -> f64 { self.reference_frequency } // Added

    fn update_temperature(&mut self, temperature: &Array3<f64>) {
        Zip::from(&mut self.temperature)
            .and(temperature)
            .for_each(|t_self, &t_updated| *t_self = t_updated.max(273.15));
    }
    fn temperature(&self) -> &Array3<f64> { &self.temperature }
    fn bubble_radius(&self) -> &Array3<f64> { &self.bubble_radius }
    fn bubble_velocity(&self) -> &Array3<f64> { &self.bubble_velocity }
    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        Zip::from(&mut self.bubble_radius)
            .and(radius)
            .for_each(|r_self, &r_updated| *r_self = r_updated.max(1e-10));
        Zip::from(&mut self.bubble_velocity)
            .and(velocity)
            .for_each(|v_self, &v_updated| *v_self = v_updated);
    }
    fn density_array(&self) -> &Array3<f64> { &self.density }
    fn sound_speed_array(&self) -> &Array3<f64> { &self.sound_speed }

    // Implement new trait methods
    fn shear_sound_speed_array(&self) -> Array3<f64> {
        self.shear_sound_speed.clone()
    }

    fn shear_viscosity_coeff_array(&self) -> Array3<f64> {
        self.shear_viscosity_coeff.clone()
    }

    fn bulk_viscosity_coeff_array(&self) -> Array3<f64> {
        self.bulk_viscosity_coeff.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use ndarray::Array3;

    fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
        Grid::new(nx, ny, nz, 0.01, 0.01, 0.01)
    }

    #[test]
    fn test_tissue_initialization_includes_shear_props() {
        let grid_dims = (2, 3, 4);
        let grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let medium = HeterogeneousMedium::tissue(&grid);

        // Check dimensions of new fields
        assert_eq!(medium.shear_sound_speed.dim(), grid_dims);
        assert_eq!(medium.shear_viscosity_coeff.dim(), grid_dims);
        assert_eq!(medium.bulk_viscosity_coeff.dim(), grid_dims);

        // Check realistic tissue values are set (these are set in new_tissue)
        // Shear sound speed should be in range 1-8 m/s (tissue-appropriate values)
        assert!(medium.shear_sound_speed.iter().all(|&x| x >= 1.0 && x <= 8.0));
        // Shear viscosity should be in range 1-3 Pa·s
        assert!(medium.shear_viscosity_coeff.iter().all(|&x| x >= 1.0 && x <= 3.0));
        // Bulk viscosity should be about 3x shear viscosity
        assert!(medium.bulk_viscosity_coeff.iter().all(|&x| x >= 3.0 && x <= 9.0));
    }

    #[test]
    fn test_shear_property_array_methods_heterogeneous() {
        let grid_dims = (2, 2, 2);
        let grid = create_test_grid(grid_dims.0, grid_dims.1, grid_dims.2);
        let mut medium = HeterogeneousMedium::new_tissue(&grid);

        // Modify the arrays directly for testing the getter methods
        let new_sss = Array3::from_elem(grid_dims, 25.0);
        let new_svc = Array3::from_elem(grid_dims, 0.25);
        let new_bvc = Array3::from_elem(grid_dims, 0.35);

        medium.shear_sound_speed = new_sss.clone();
        medium.shear_viscosity_coeff = new_svc.clone();
        medium.bulk_viscosity_coeff = new_bvc.clone();

        let sss_arr = medium.shear_sound_speed_array();
        assert_eq!(sss_arr, new_sss);
        // Ensure it's a clone, not the same instance
        assert_ne!(sss_arr.as_ptr(), medium.shear_sound_speed.as_ptr());


        let svc_arr = medium.shear_viscosity_coeff_array();
        assert_eq!(svc_arr, new_svc);
        assert_ne!(svc_arr.as_ptr(), medium.shear_viscosity_coeff.as_ptr());

        let bvc_arr = medium.bulk_viscosity_coeff_array();
        assert_eq!(bvc_arr, new_bvc);
        assert_ne!(bvc_arr.as_ptr(), medium.bulk_viscosity_coeff.as_ptr());
    }
}