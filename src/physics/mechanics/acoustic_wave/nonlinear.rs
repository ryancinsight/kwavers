// physics/mechanics/acoustic_wave/nonlinear.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::source::Source;
use crate::utils::{fft_3d, ifft_3d};
use log::{debug, warn};
use ndarray::{Array3, Array4, Axis, Zip};
use num_complex::Complex;

pub const PRESSURE_IDX: usize = 0;

#[derive(Debug)]
pub struct NonlinearWave {}

impl NonlinearWave {
    pub fn new(_grid: &Grid) -> Self {
        debug!("Initializing NonlinearWave for k-space with split-step method");
        Self {}
    }

    pub fn update_wave(
        &self,
        fields: &mut Array4<f64>,
        prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) {
        debug!("Updating nonlinear wave with split-step k-space at t = {:.6e}", t);

        let pressure = fields.index_axis(Axis(0), PRESSURE_IDX);

        // Nonlinear term in spatial domain
        let mut nonlinear_term = Array3::zeros(pressure.dim());
        Zip::indexed(&mut nonlinear_term)
            .and(pressure)
            .and(prev_pressure)
            .par_for_each(|(i, j, k), nl, &p, &p_old| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let rho = medium.density(x, y, z, grid);
                let c = medium.sound_speed(x, y, z, grid);
                let b_a = medium.nonlinearity_coefficient(x, y, z, grid);
                let dp_dt = (p - p_old) / dt;
                *nl = b_a * dp_dt * dp_dt / (2.0 * rho * c * c);
                if nl.is_nan() || nl.is_infinite() {
                    *nl = 0.0;
                    warn!("Nonlinear term corrected: NaN/infinite at ({}, {}, {}) t = {:.6e}", i, j, k, t);
                }
            });

        // Linear propagation in k-space with viscosity
        let p_fft = fft_3d(fields, PRESSURE_IDX, grid);
        let k2 = grid.k_squared();
        let kspace_corr = grid.kspace_correction(medium.sound_speed(0.0, 0.0, 0.0, grid), dt);
        let mut p_linear_fft = Array3::zeros(p_fft.dim());
        let freq = medium.reference_frequency();

        Zip::indexed(&mut p_linear_fft)
            .and(&p_fft)
            .and(&kspace_corr)
            .par_for_each(|(i, j, k), p_new, &p, &corr| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let c = medium.sound_speed(x, y, z, grid);
                let mu = medium.viscosity(x, y, z, grid);
                let rho = medium.density(x, y, z, grid);
                let k_val = k2[[i, j, k]].sqrt();
                let phase = -c * k_val * dt;
                let viscous_damping = (mu * k_val * k_val * dt / rho).exp(); // Simplified viscous attenuation
                let decay = (-medium.absorption_coefficient(x, y, z, grid, freq) * dt).exp() * viscous_damping;
                *p_new = p * Complex::new(phase.cos(), phase.sin()) * corr * decay;
            });

        let mut p_linear = ifft_3d(&p_linear_fft, grid);

        // Apply source term
        let mut src = Array3::zeros(pressure.dim());
        Zip::indexed(&mut src).for_each(|(i, j, k), s| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            *s = source.get_source_term(t, x, y, z, grid);
        });

        // Combine linear and nonlinear terms
        Zip::from(&mut p_linear)
            .and(&nonlinear_term)
            .and(&src)
            .par_for_each(|p, &nl, &s| {
                *p += dt * (nl + s);
                if p.is_nan() || p.is_infinite() {
                    *p = 0.0;
                    warn!("Pressure corrected: NaN/infinite at t = {:.6e}", t);
                }
            });

        fields.index_axis_mut(Axis(0), PRESSURE_IDX).assign(&p_linear);
    }
}