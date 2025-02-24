// src/medium/homogeneous/mod.rs
use crate::grid::Grid;
use crate::medium::{absorption, power_law_absorption, Medium};
use log::debug;
use ndarray::{Array3, Zip};

#[derive(Debug, Clone)]
pub struct HomogeneousMedium {
    pub density: f64,
    pub sound_speed: f64,
    pub viscosity: f64,
    pub surface_tension: f64,
    pub ambient_pressure: f64,
    pub vapor_pressure: f64,
    pub polytropic_index: f64,
    pub specific_heat: f64,
    pub thermal_conductivity: f64,
    pub thermal_expansion: f64,
    pub gas_diffusion_coeff: f64,
    pub thermal_diffusivity: f64,
    pub mu_a: f64,
    pub mu_s_prime: f64,
    pub temperature: Array3<f64>,
    pub bubble_radius: Array3<f64>,
    pub bubble_velocity: Array3<f64>,
    pub alpha0: f64,
    pub delta: f64,
    pub b_a: f64,
    pub reference_frequency: f64, // Added
}

impl HomogeneousMedium {
    pub fn new(density: f64, sound_speed: f64, grid: &Grid, mu_a: f64, mu_s_prime: f64) -> Self {
        assert!(density > 0.0 && sound_speed > 0.0 && mu_a >= 0.0 && mu_s_prime >= 0.0);

        let viscosity = 1.002e-3; // Water at 20°C
        let surface_tension = 0.0728; // Water
        let ambient_pressure = 1.013e5; // Standard atmospheric pressure
        let vapor_pressure = 2.338e3; // Water at 20°C
        let polytropic_index = 1.4; // Typical for gases
        let specific_heat = 4182.0; // Water
        let thermal_conductivity = 0.598; // Water
        let thermal_expansion = 2.1e-4; // Water
        let gas_diffusion_coeff = 2e-9; // Typical value
        let thermal_diffusivity = 1.43e-7; // Water
        let b_a = 5.2; // Water
        let reference_frequency = 180000.0; // Default ultrasound frequency (180 kHz)

        let temperature = Array3::from_elem((grid.nx, grid.ny, grid.nz), 293.15); // 20°C
        let bubble_radius = Array3::from_elem((grid.nx, grid.ny, grid.nz), 10e-6);
        let bubble_velocity = Array3::zeros((grid.nx, grid.ny, grid.nz));

        let alpha0 = 0.025; // Water attenuation coefficient
        let delta = 1.0; // Power law exponent

        debug!(
            "Initialized HomogeneousMedium: density = {:.2}, sound_speed = {:.2}, b_a = {:.2}, freq = {:.2e}",
            density, sound_speed, b_a, reference_frequency
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

    pub fn water(grid: &Grid) -> Self {
        Self::new(998.0, 1500.0, grid, 1.0, 50.0)
    }
}

impl Medium for HomogeneousMedium {
    fn density(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.density }
    fn sound_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.sound_speed }
    fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.viscosity }
    fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.surface_tension }
    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.ambient_pressure }
    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.vapor_pressure }
    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.polytropic_index }
    fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.specific_heat }
    fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_conductivity }
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let ix = grid.x_idx(x);
        let iy = grid.y_idx(y);
        let iz = grid.z_idx(z);
        let t = self.temperature[[ix, iy, iz]];
        power_law_absorption::power_law_absorption_coefficient(frequency, self.alpha0, self.delta)
            + absorption::absorption_coefficient(frequency, t, Some(self.bubble_radius[[ix, iy, iz]]))
    }
    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_expansion }
    fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.gas_diffusion_coeff }
    fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.thermal_diffusivity }
    fn nonlinearity_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.b_a }
    fn absorption_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.mu_a }
    fn reduced_scattering_coefficient_light(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 { self.mu_s_prime }
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
    fn density_array(&self) -> Array3<f64> {
        Array3::from_elem(
            (self.temperature.dim().0, self.temperature.dim().1, self.temperature.dim().2),
            self.density,
        )
    }
    fn sound_speed_array(&self) -> Array3<f64> {
        Array3::from_elem(
            (self.temperature.dim().0, self.temperature.dim().1, self.temperature.dim().2),
            self.sound_speed,
        )
    }
}