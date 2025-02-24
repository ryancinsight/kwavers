// physics/scattering/acoustic/mie.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

pub fn compute_mie_scattering(
    scatter: &mut Array3<f64>,
    radius: &Array3<f64>,
    p: &Array3<f64>,
    grid: &Grid,
    medium: &dyn Medium,
    frequency: f64,
) {
    debug!(
        "Computing Mie scattering at frequency = {:.2e} Hz",
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

            if kr >= 1.0 {
                let sin_kr = kr.sin();
                let cos_kr = kr.cos();
                let kr_sq = kr * kr;

                let a0 = sin_kr / kr;
                let a1 = (2.0 * sin_kr - kr * cos_kr) / kr_sq;
                let b1 = sin_kr / kr_sq;
                let a2 = (9.0 * sin_kr - 3.0 * kr * cos_kr - kr_sq * sin_kr) / (kr * kr_sq);
                let b2 = (3.0 * sin_kr - kr * cos_kr) / (kr * kr_sq);

                let amplitude =
                    (a0.abs() + (2.0 * a1.abs() + b1.abs()) + (5.0 * a2.abs() + b2.abs())) * kr_sq;
                *s = rho * p_val * amplitude / (grid.dx * grid.dy * grid.dz).powf(1.0 / 3.0);
            } else {
                *s = 0.0;
            }

            if s.is_nan() || s.is_infinite() {
                *s = 0.0;
            }
        });
}