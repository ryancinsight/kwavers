// physics/mechanics/cavitation/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::scattering::acoustic::{compute_bubble_interactions, compute_mie_scattering, compute_rayleigh_scattering};
use log::{debug, trace};
use ndarray::{Array3, Axis, ArrayViewMut3, Zip};
use rayon::prelude::*;
use std::f64::consts::PI;
const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;
#[derive(Debug)]
pub struct CavitationModel {
    pub radius: Array3<f64>,
    pub velocity: Array3<f64>,
    pub prev_velocity: Array3<f64>,
    pub temperature: Array3<f64>,
}

impl CavitationModel {
    pub fn new(grid: &Grid, initial_radius: f64) -> Self {
        debug!(
            "Initializing CavitationModel with initial radius = {:.6e} m",
            initial_radius
        );
        Self {
            radius: Array3::from_elem((grid.nx, grid.ny, grid.nz), initial_radius),
            velocity: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            prev_velocity: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            temperature: Array3::from_elem((grid.nx, grid.ny, grid.nz), 293.15),
        }
    }

    pub fn update_cavitation(
        &mut self,
        p_update: &mut Array3<f64>,
        p: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &mut dyn Medium,
        frequency: f64,
    ) -> Array3<f64> {
        debug!(
            "Updating cavitation dynamics at frequency = {:.2e} Hz",
            frequency
        );
        

        let mut d2r_dt2 = Array3::zeros(p.dim());
        let mut light_source = Array3::zeros(p.dim());

        // Split d2r_dt2 into chunks and process in parallel
        let chunk_size = grid.nx / rayon::current_num_threads().max(1);
        let chunk_size = chunk_size.max(1); // Ensure at least one element per chunk
        d2r_dt2.axis_chunks_iter_mut(Axis(0), chunk_size).enumerate().for_each(|(chunk_idx, mut d2r_chunk)| {
            light_source.axis_chunks_iter_mut(Axis(0), chunk_size).enumerate().for_each(|(ls_chunk_idx, mut ls_chunk)| {
                if chunk_idx == ls_chunk_idx {
                    let start_i = chunk_idx * chunk_size;
                    let end_i = (start_i + chunk_size).min(grid.nx);
                    Self::process_chunk(
                        &mut d2r_chunk,
                        &mut ls_chunk,
                        self,
                        p,
                        grid,
                        dt,
                        medium,
                        frequency,
                        start_i,
                        end_i,
                    );
                }
            });
        });

        let mut rayleigh_scatter = Array3::zeros(p.dim());
        let mut mie_scatter = Array3::zeros(p.dim());
        let mut interaction_scatter = Array3::zeros(p.dim());

        compute_rayleigh_scattering(&mut rayleigh_scatter, &self.radius, p, grid, medium, frequency);
        compute_mie_scattering(&mut mie_scatter, &self.radius, p, grid, medium, frequency);
        compute_bubble_interactions(&mut interaction_scatter, &self.radius, &self.velocity, p, grid, medium, frequency);

        let mut total_scatter = Array3::zeros(p.dim());
        Zip::from(&mut total_scatter)
            .and(&rayleigh_scatter)
            .and(&mie_scatter)
            .and(&interaction_scatter)
            .par_for_each(|total, &ray, &mie, &inter| {
                *total = ray + mie + inter;
            });

        Zip::from(p_update)
            .and(&self.radius)
            .and(&self.velocity)
            .and(&total_scatter)
            .par_for_each(|p, &r, &v, &s| {
                let d_volume_dt = 4.0 * PI * r.powi(2) * v;
                *p -= d_volume_dt / dt + s;
                if p.is_nan() || p.is_infinite() {
                    *p = 0.0;
                }
            });

        medium.update_bubble_state(&self.radius, &self.velocity);
        trace!(
            "Cavitation light at center: {:.6e} W/m³",
            light_source[[grid.nx / 2, grid.ny / 2, grid.nz / 2]]
        );
        light_source
    }

    fn process_chunk(
        d2r_chunk: &mut ArrayViewMut3<f64>,
        ls_chunk: &mut ArrayViewMut3<f64>,
        cavitation: &mut CavitationModel,
        p: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &mut dyn Medium,
        frequency: f64,
        start_i: usize,
        end_i: usize,
    ) {
        for i in start_i..end_i {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    let rho = medium.density(x, y, z, grid);
                    let _c = medium.sound_speed(x, y, z, grid);
                    let mu = medium.viscosity(x, y, z, grid);
                    let sigma = medium.surface_tension(x, y, z, grid);
                    let p0 = medium.ambient_pressure(x, y, z, grid);
                    let pv = medium.vapor_pressure(x, y, z, grid);
                    let gamma = medium.polytropic_index(x, y, z, grid);
                    let kappa = medium.thermal_conductivity(x, y, z, grid);
                    let dg = medium.gas_diffusion_coefficient(x, y, z, grid);

                    let chunk_idx = i - start_i;
                    let r = &mut cavitation.radius[[i, j, k]];
                    let v = &mut cavitation.velocity[[i, j, k]];
                    let v_prev = &mut cavitation.prev_velocity[[i, j, k]];
                    let t_bubble = &mut cavitation.temperature[[i, j, k]];
                    let p_val = p[[i, j, k]];
                    let d2r = &mut d2r_chunk[[chunk_idx, j, k]];
                    let l = &mut ls_chunk[[chunk_idx, j, k]];

                    let r_clamped = r.max(1e-10);
                    let p_infinity = p_val;
                    let r0 = 10e-6;

                    let p_gas = (p0 + 2.0 * sigma / r0 - pv) * (r0 / r_clamped).powf(3.0 * gamma);
                    let viscous_term = 4.0 * mu * *v / r_clamped;
                    let surface_term = 2.0 * sigma / r_clamped;
                    let thermal_damping = 3.0 * gamma * kappa * (*t_bubble - medium.temperature()[[i, j, k]]) / (r_clamped * r_clamped);
                    let diffusion_term = dg * (p0 - p_gas) / (r_clamped * rho);
                    let pressure_diff = p_gas + pv - p_infinity;

                    let rhs = (pressure_diff - viscous_term - surface_term - thermal_damping - diffusion_term) / rho
                        - 1.5 * *v * *v;

                    *d2r = rhs / r_clamped;
                    if d2r.is_nan() || d2r.is_infinite() {
                        *d2r = 0.0;
                    }

                    *v_prev = *v;
                    *v += *d2r * dt;
                    *r += *v * dt;
                    *r = r.max(1e-10);

                    if *v_prev < 0.0 && *v >= 0.0 {
                        let t_max = *t_bubble * (r0 / r_clamped).powf(3.0 * (gamma - 1.0));
                        *t_bubble = t_max;
                        let power = 4.0 * PI * r_clamped.powi(2) * STEFAN_BOLTZMANN * t_max.powi(4);
                        *l = power / (grid.dx * grid.dy * grid.dz);
                    } else {
                        *l = 0.0;
                        *t_bubble -= kappa * (*t_bubble - medium.temperature()[[i, j, k]]) * dt / (r_clamped * rho);
                        *t_bubble = t_bubble.max(medium.temperature()[[i, j, k]]);
                    }
                }
            }
        }
    }

    pub fn radius(&self) -> &Array3<f64> {
        &self.radius
    }
    pub fn velocity(&self) -> &Array3<f64> {
        &self.velocity
    }
    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }
}