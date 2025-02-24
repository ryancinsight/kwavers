// physics/scattering/acoustic/rayleigh.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

pub fn compute_rayleigh_scattering(
    scatter: &mut Array3<f64>,
    radius: &Array3<f64>,
    p: &Array3<f64>,
    grid: &Grid,
    medium: &dyn Medium,
    frequency: f64,
) {
    debug!(
        "Computing Rayleigh scattering at frequency = {:.2e} Hz",
        frequency
    );
    let wavenumber = 2.0 * PI * frequency / medium.sound_speed(0.0, 0.0, 0.0, grid);

    Zip::indexed(scatter)
        .and(radius)
        .and(p)
        .for_each(|(i, j, k_idx), s, &r, &p_val| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k_idx as f64 * grid.dz;
            let rho = medium.density(x, y, z, grid);
            let r_clamped = r.max(1e-10);
            let kr = wavenumber * r_clamped;

            let scattering_amplitude = if kr < 1.0 {
                let factor = kr.powi(4) / (grid.dx * grid.dy * grid.dz).powf(1.0 / 3.0);
                p_val * factor
            } else {
                0.0
            };

            *s = rho * scattering_amplitude;
            if s.is_nan() || s.is_infinite() {
                *s = 0.0;
            }
        });
}