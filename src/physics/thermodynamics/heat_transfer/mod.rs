// physics/thermodynamics/heat_transfer/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::utils::laplacian;
use log::debug;
use ndarray::{Array3, Array4, Axis, Zip};

pub const TEMPERATURE_IDX: usize = 2;

#[derive(Debug)]
pub struct ThermalModel {
    pub temperature: Array3<f64>,
    pub tau_q: f64, // Thermal relaxation time for heat flux
    pub tau_t: f64, // Thermal relaxation time for temperature
}

impl ThermalModel {
    pub fn new(grid: &Grid, initial_temp: f64, tau_q: f64, tau_t: f64) -> Self {
        debug!(
            "Initializing ThermalModel with tau_q = {}, tau_t = {}",
            tau_q, tau_t
        );
        Self {
            temperature: Array3::from_elem((grid.nx, grid.ny, grid.nz), initial_temp),
            tau_q,
            tau_t,
        }
    }

    pub fn update_thermal(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &mut dyn Medium,
        dt: f64,
        frequency: f64,
    ) {
        debug!("Updating thermal effects with conductivity gradients");
        let mut heat_source = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let lap_t = laplacian(fields, TEMPERATURE_IDX, grid).expect("Laplacian failed");

        let pressure = fields.index_axis(Axis(0), 0); // PRESSURE_IDX
        let light = fields.index_axis(Axis(0), 1); // LIGHT_IDX
        Zip::indexed(&mut heat_source)
            .and(&pressure)
            .and(&light)
            .par_for_each(|(i, j, k), q, &p_val, &l_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let alpha = medium.absorption_coefficient(x, y, z, grid, frequency);
                let mu_a = medium.absorption_coefficient_light(x, y, z, grid);
                let rho = medium.density(x, y, z, grid);
                let c = medium.sound_speed(x, y, z, grid);
                *q = 2.0 * alpha * (p_val.powi(2) / (2.0 * rho * c)) + mu_a * l_val.max(0.0);
            });

        let mut temp_new = Array3::zeros(self.temperature.dim());
        Zip::indexed(&mut temp_new)
            .and(&self.temperature)
            .and(&lap_t)
            .and(&heat_source)
            .par_for_each(|(i, j, k), t_new, &t_old, &lap, &q| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let k = medium.thermal_conductivity(x, y, z, grid);
                let rho = medium.density(x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);
                let d_alpha = medium.thermal_diffusivity(x, y, z, grid);
                *t_new = t_old + dt * (d_alpha * lap + q - self.tau_q * q / dt + self.tau_t * k * lap) / (rho * cp);
                if t_new.is_nan() || t_new.is_infinite() {
                    *t_new = t_old;
                }
            });

        self.temperature.assign(&temp_new);
        fields.index_axis_mut(Axis(0), TEMPERATURE_IDX).assign(&self.temperature);
        medium.update_temperature(&self.temperature);
    }

    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }
}