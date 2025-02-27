// config/simulation.rs

use crate::grid::Grid;
use crate::medium::homogeneous::HomogeneousMedium;
use crate::time::Time;
use log::{debug, info};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct SimulationConfig {
    pub domain_size_x: f64,
    pub domain_size_yz: f64,
    pub points_per_wavelength: usize,
    pub frequency: f64,
    pub amplitude: f64,
    pub num_cycles: f64,
    #[serde(default)]
    pub pml_thickness: usize,
    // PML specific parameters
    #[serde(default = "default_pml_sigma_acoustic")]
    pub pml_sigma_acoustic: f64,
    #[serde(default = "default_pml_sigma_light")]
    pub pml_sigma_light: f64,
    #[serde(default = "default_pml_polynomial_order")]
    pub pml_polynomial_order: usize,
    #[serde(default = "default_pml_reflection")]
    pub pml_reflection: f64,
    #[serde(default = "default_light_wavelength")]
    pub light_wavelength: f64,
    // k-space specific parameters
    #[serde(default = "default_kspace_padding")]
    pub kspace_padding: usize, // Padding for FFT to avoid aliasing
    #[serde(default = "default_kspace_alpha")]
    pub kspace_alpha: f64, // k-space correction coefficient
}

fn default_pml_sigma_acoustic() -> f64 {
    100.0
}

fn default_pml_sigma_light() -> f64 {
    10.0
}

fn default_pml_polynomial_order() -> usize {
    3
}

fn default_pml_reflection() -> f64 {
    1e-6
}

fn default_light_wavelength() -> f64 {
    500.0
}
fn default_kspace_padding() -> usize {
    0
} // No padding by default
fn default_kspace_alpha() -> f64 {
    1.0
} // Default correction factor

impl SimulationConfig {
    pub fn initialize_grid(&self) -> Result<Grid, String> {
        let wavelength = 1500.0 / self.frequency;
        info!("Calculated acoustic wavelength: {} m", wavelength);
        let dx = wavelength / self.points_per_wavelength as f64;
        let nx = (self.domain_size_x / dx).ceil() as usize + self.kspace_padding;
        let ny = (self.domain_size_yz / dx).ceil() as usize + self.kspace_padding;
        let nz = (self.domain_size_yz / dx).ceil() as usize + self.kspace_padding;

        let grid = Grid::new(nx, ny, nz, dx, dx, dx);
        debug!(
            "Grid initialized with k-space padding: {}x{}x{}, dx = {}",
            grid.nx, grid.ny, grid.nz, grid.dx
        );
        Ok(grid)
    }

    pub fn initialize_time(&self, grid: &Grid) -> Result<Time, String> {
        let duration = self.num_cycles / self.frequency;
        let c = 1500.0;
        let max_dt = grid.dx / (3.0f64.sqrt() * c);
        let dt = max_dt * 0.95 * self.kspace_alpha; // Adjusted for k-space stability
        let n_steps = (duration / dt).ceil() as usize;

        let time = Time::new(dt, n_steps);
        debug!(
            "Time initialized: n_steps = {}, dt = {}",
            time.n_steps, time.dt
        );
        if !time.is_stable(grid.dx, grid.dy, grid.dz, c) {
            return Err("Unstable time step detected".to_string());
        }
        Ok(time)
    }

    pub fn initialize_medium(&self, grid: &Grid) -> Box<HomogeneousMedium> {
        Box::new(HomogeneousMedium::water(grid))
    }
}
