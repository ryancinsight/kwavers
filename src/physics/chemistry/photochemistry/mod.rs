// physics/chemistry/photochemistry/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};


#[derive(Debug)]
pub struct PhotochemicalEffects {
    pub reactive_oxygen_species: Array3<f64>, // ROS like singlet oxygen (¹O₂), superoxide (O₂•⁻)
}

impl PhotochemicalEffects {
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing PhotochemicalEffects");
        Self {
            reactive_oxygen_species: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_photochemical(
        &mut self,
        light: &Array3<f64>,
        emission_spectrum: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        temperature: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
    ) {
        debug!("Updating photochemical effects");

        Zip::indexed(&mut self.reactive_oxygen_species)
            .and(light)
            .and(emission_spectrum)
            .and(bubble_radius)
            .and(temperature)
            .for_each(|(i, j, k), ros, &light_val, &spec_val, &r_val, &t| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                // Photosensitizer-like activation (e.g., ROS production in sonodynamic therapy)
                let mu_a = medium.absorption_coefficient_light(x, y, z, grid);
                let k_photo = 1e-6 * mu_a * (t / 298.15); // Temperature-dependent rate
                let light_intensity = light_val.max(0.0);

                // ROS production enhanced by bubble presence and emission spectrum
                let ros_rate = k_photo * light_intensity * spec_val / 1e-9; // Normalize spectrum (nm)
                *ros += ros_rate * dt * (1.0 + r_val / 1e-6); // Bubble amplification
                *ros = ros.max(0.0);
            });
    }

    pub fn reactive_oxygen_species(&self) -> &Array3<f64> {
        &self.reactive_oxygen_species
    }
}