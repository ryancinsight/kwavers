// config/simulation.rs

use crate::grid::Grid;
use crate::medium::Medium; // Added
use crate::medium::homogeneous::HomogeneousMedium;
use crate::medium::heterogeneous::tissue::HeterogeneousTissueMedium; // Added
use crate::time::Time;
use log::{debug, info, warn}; // Added warn
use serde::Deserialize;
use std::sync::Arc; // Added

#[derive(Debug, Deserialize)]
pub struct SimulationConfig {
    pub domain_size_x: f64,
    pub domain_size_yz: f64,
    pub points_per_wavelength: usize,
    pub frequency: f64,
    // pub amplitude: f64, // Amplitude is now part of SourceConfig
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
    #[serde(default)]
    pub medium_type: Option<String>, // Added for medium selection
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

pub fn default_light_wavelength() -> f64 { // Made public
    500.0
}
pub fn default_kspace_padding() -> usize { // Made public
    0
} // No padding by default
fn default_kspace_alpha() -> f64 {
    1.0
} // Default correction factor

impl SimulationConfig {
    pub fn initialize_grid(&self) -> Result<Grid, String> {
        self.initialize_grid_with_sound_speed(1500.0)
    }

    pub fn initialize_grid_with_sound_speed(&self, sound_speed: f64) -> Result<Grid, String> {
        let wavelength = sound_speed / self.frequency;
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
        self.initialize_time_with_sound_speed(grid, 1500.0)
    }

    pub fn initialize_time_with_sound_speed(&self, grid: &Grid, sound_speed: f64) -> Result<Time, String> {
        let duration = self.num_cycles / self.frequency;
        let max_dt = grid.cfl_timestep_default(sound_speed);
        let dt = max_dt * 0.95 * self.kspace_alpha; // Adjusted for k-space stability
        let n_steps = (duration / dt).ceil() as usize;

        let time = Time::new(dt, n_steps);
        debug!(
            "Time initialized: n_steps = {}, dt = {}",
            time.n_steps, time.dt
        );
        if !time.is_stable(grid.dx, grid.dy, grid.dz, sound_speed) {
            return Err("Unstable time step detected".to_string());
        }
        Ok(time)
    }

    pub fn initialize_medium(&self, grid: &Grid) -> Result<Arc<dyn Medium>, String> {
        match self.medium_type.as_deref() {
            Some("layered_tissue") => {
                info!("Initializing layered tissue medium.");
                Ok(Arc::new(HeterogeneousTissueMedium::new_layered(grid)))
            }
            Some("homogeneous_water") | None => {
                if self.medium_type.is_none() {
                    warn!("medium_type not specified in config, defaulting to homogeneous_water.");
                }
                info!("Initializing homogeneous water medium.");
                Ok(Arc::new(HomogeneousMedium::water(grid)))
            }
            Some(unknown_type) => {
                Err(format!("Unknown medium_type in config: {}", unknown_type))
            }
        }
    }

    /// Initialize time from a medium, using the medium's maximum sound speed for CFL condition
    pub fn initialize_time_from_medium(&self, grid: &Grid, medium: &dyn crate::medium::Medium) -> Result<Time, String> {
        let max_sound_speed = crate::medium::max_sound_speed(medium);
        self.initialize_time_with_sound_speed(grid, max_sound_speed)
    }
}
