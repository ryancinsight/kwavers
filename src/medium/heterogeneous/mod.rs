// src/medium/heterogeneous/mod.rs
use crate::grid::Grid;
use crate::medium::{absorption, power_law_absorption, Medium};
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
}

impl HeterogeneousMedium {
    pub fn new_tissue(grid: &Grid) -> Self {
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
        }
    }
}

impl Medium for HeterogeneousMedium {
    fn density(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.density[[ix, iy, iz]].max(1.0)
    }
    fn sound_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.sound_speed[[ix, iy, iz]].max(100.0)
    }
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.viscosity[[ix, iy, iz]].max(1e-6)
    }
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.surface_tension[[ix, iy, iz]].max(0.01)
    }
    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.ambient_pressure
    }
    fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.vapor_pressure[[ix, iy, iz]].max(1.0)
    }
    fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.polytropic_index[[ix, iy, iz]]
    }
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.specific_heat[[ix, iy, iz]]
    }
    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.thermal_conductivity[[ix, iy, iz]].max(0.1)
    }
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        let t = self.temperature[[ix, iy, iz]];
        let alpha0 = self.alpha0[[ix, iy, iz]];
        let delta = self.delta[[ix, iy, iz]];
        power_law_absorption::power_law_absorption_coefficient(frequency, alpha0, delta)
            + absorption::absorption_coefficient(frequency, t, Some(self.bubble_radius[[ix, iy, iz]]))
    }
    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.thermal_expansion[[ix, iy, iz]].max(1e-6)
    }
    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.gas_diffusion_coeff[[ix, iy, iz]].max(1e-10)
    }
    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        let rho = self.density(x, y, z, grid);
        let cp = self.specific_heat(x, y, z, grid);
        let k = self.thermal_conductivity(x, y, z, grid);
        (k / (rho * cp)).max(1e-8)
    }
    fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.b_a[[ix, iy, iz]].max(0.0)
    }
    fn absorption_coefficient_light(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.mu_a[[ix, iy, iz]].max(0.1)
    }
    fn reduced_scattering_coefficient_light(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        self.mu_s_prime[[ix, iy, iz]].max(1.0)
    }
    fn reference_frequency(&self) -> f64 { self.reference_frequency } // Added

    fn update_temperature(&mut self, temperature: &Array3<f64>) {
        Zip::from(&mut self.temperature)
            .and(temperature)
            .for_each(|t_self, &t_new| *t_self = t_new.max(273.15));
    }
    fn temperature(&self) -> &Array3<f64> { &self.temperature }
    fn bubble_radius(&self) -> &Array3<f64> { &self.bubble_radius }
    fn bubble_velocity(&self) -> &Array3<f64> { &self.bubble_velocity }
    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        Zip::from(&mut self.bubble_radius)
            .and(radius)
            .for_each(|r_self, &r_new| *r_self = r_new.max(1e-10));
        Zip::from(&mut self.bubble_velocity)
            .and(velocity)
            .for_each(|v_self, &v_new| *v_self = v_new);
    }
    fn density_array(&self) -> Array3<f64> { self.density.clone() }
    fn sound_speed_array(&self) -> Array3<f64> { self.sound_speed.clone() }
}