// physics/chemistry/radical_initiation/mod.rs
use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};

#[derive(Debug, Clone)]
pub struct RadicalInitiation {
    pub radical_concentration: Array3<f64>, // General radical concentration (e.g., H•, OH• precursors)
}

impl RadicalInitiation {
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing RadicalInitiation");
        Self {
            radical_concentration: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_radicals(
        &mut self,
        p: &Array3<f64>,
        light: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
        frequency: f64,
    ) {
        debug!("Updating radical initiation from cavitation and light");

        Zip::indexed(&mut self.radical_concentration)
            .and(p)
            .and(light)
            .and(bubble_radius)
            .for_each(|(i, j, k), conc, &p_val, &light_val, &r_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let alpha = kwavers_medium::AcousticProperties::absorption_coefficient(
                    medium, x, y, z, grid, frequency,
                );

                // Cavitation-induced radical formation (e.g., water sonolysis)
                let cav_rate = if p_val < -MPA_TO_PA {
                    // Threshold for cavitation collapse at -1 MPa
                    1e-6 * alpha * (-p_val - MPA_TO_PA) * r_val.powi(2) // Proportional to bubble surface area
                } else {
                    0.0
                };

                // Basic light-induced radical formation (pre-photochemistry)
                let light_rate = kwavers_core::constants::chemistry::BASE_PHOTOCHEMICAL_RATE
                    * light_val.max(0.0);

                let total_rate = (cav_rate + light_rate) * dt;
                if rand::random::<f64>() < total_rate {
                    *conc += 1e-6; // Incremental radical increase
                }
                *conc = conc.max(0.0);
            });
    }
}
