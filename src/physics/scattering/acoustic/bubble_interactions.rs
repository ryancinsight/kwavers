// physics/scattering/acoustic/bubble_interactions.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

pub fn compute_bubble_interactions(
    scatter: &mut Array3<f64>,
    radius: &Array3<f64>,
    velocity: &Array3<f64>,
    p: &Array3<f64>,
    grid: &Grid,
    medium: &dyn Medium,
    frequency: f64,
) {
    debug!(
        "Computing bubble interactions at frequency = {:.2e} Hz",
        frequency
    );
    let wavenumber = 2.0 * PI * frequency / medium.sound_speed(0.0, 0.0, 0.0, grid);

    Zip::indexed(scatter)
        .and(radius)
        .and(velocity)
        .and(p)
        .for_each(|(i, j, k_idx), s, &r, &v, &_p_val| {
            let x0 = i as f64 * grid.dx;
            let y0 = j as f64 * grid.dy;
            let z0 = k_idx as f64 * grid.dz;
            let rho = medium.density(x0, y0, z0, grid);
            let r_clamped = r.max(1e-10);

            let mut force_sum = 0.0;
            let interaction_range = 5.0 * grid.dx;

            for di in -2..=2 {
                for dj in -2..=2 {
                    for dk in -2..=2 {
                        let ni = (i as isize + di).clamp(0, grid.nx as isize - 1) as usize;
                        let nj = (j as isize + dj).clamp(0, grid.ny as isize - 1) as usize;
                        let nk = (k_idx as isize + dk).clamp(0, grid.nz as isize - 1) as usize;

                        if (ni, nj, nk) != (i, j, k_idx) {
                            let x1 = ni as f64 * grid.dx;
                            let y1 = nj as f64 * grid.dy;
                            let z1 = nk as f64 * grid.dz;
                            let dist =
                                ((x1 - x0).powi(2) + (y1 - y0).powi(2) + (z1 - z0).powi(2)).sqrt();

                            if dist < interaction_range && dist > 0.0 {
                                let r_other = radius[[ni, nj, nk]].max(1e-10);
                                let v_other = velocity[[ni, nj, nk]];
                                let volume = 4.0 / 3.0 * PI * r_clamped.powi(3);
                                let volume_other = 4.0 / 3.0 * PI * r_other.powi(3);
                                let phase_diff = wavenumber * dist;
                                let force =
                                    (volume * v_other * volume_other * v * phase_diff.cos())
                                        / (dist * dist);
                                force_sum += rho * force;
                            }
                        }
                    }
                }
            }

            *s = force_sum / (grid.dx * grid.dy * grid.dz);
            if s.is_nan() || s.is_infinite() {
                *s = 0.0;
            }
        });
}