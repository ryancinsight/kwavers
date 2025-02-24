use crate::boundary::Boundary;
use crate::grid::Grid;
use crate::medium::Medium;
use log::{debug, trace};
use ndarray::{Array3, Zip};
use rustfft::num_complex::Complex;

#[derive(Debug, Clone)]
pub struct PMLBoundary {
    thickness: usize,
    sigma_max_acoustic: f64,
    sigma_max_light: f64,
    acoustic_damping: Vec<f64>,
    light_damping: Vec<f64>,
}

impl PMLBoundary {
    pub fn new(
        thickness: usize,
        sigma_max_acoustic: f64,
        sigma_max_light: f64,
        medium: &dyn Medium,
        grid: &Grid,
        acoustic_freq: f64,
    ) -> Self {
        debug!(
            "Initializing PMLBoundary for k-space: sigma_acoustic = {}, sigma_light = {}",
            sigma_max_acoustic, sigma_max_light
        );

        let c = medium.sound_speed(0.0, 0.0, 0.0, grid);
        let wavelength = c / acoustic_freq;
        let acoustic_thickness = (wavelength * 10.0 / grid.dx).ceil() as usize;

        let mu_a = medium.absorption_coefficient_light(0.0, 0.0, 0.0, grid);
        let mu_s_prime = medium.reduced_scattering_coefficient_light(0.0, 0.0, 0.0, grid);
        let diffusion_length = (3.0 * (mu_a + mu_s_prime)).sqrt();
        let light_thickness = (diffusion_length * 10.0 / grid.dx).ceil() as usize;

        let final_thickness = if thickness == 0 {
            acoustic_thickness.max(light_thickness)
        } else {
            thickness
        };

        let acoustic_damping =
            Self::damping_profile(final_thickness, grid.nx, grid.dx, sigma_max_acoustic);
        let light_damping =
            Self::damping_profile(final_thickness, grid.nx, grid.dx, sigma_max_light);

        debug!(
            "PML thickness: acoustic = {}, light = {}, final = {}",
            acoustic_thickness, light_thickness, final_thickness
        );
        assert!(
            sigma_max_acoustic > 0.0 && sigma_max_light > 0.0,
            "Damping coefficients must be positive"
        );

        Self {
            thickness: final_thickness,
            sigma_max_acoustic,
            sigma_max_light,
            acoustic_damping,
            light_damping,
        }
    }

    fn damping_profile(thickness: usize, length: usize, dx: f64, sigma_max: f64) -> Vec<f64> {
        (0..length)
            .map(|i| {
                let pml_start = length.saturating_sub(thickness);
                if i >= pml_start {
                    let dist = (i - pml_start) as f64 * dx;
                    let max_dist = thickness as f64 * dx;
                    sigma_max * (dist / max_dist).powi(2)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

impl Boundary for PMLBoundary {
    fn apply_acoustic(&self, field: &mut Array3<f64>, grid: &Grid, time_step: usize) {
        debug!("Applying spatial acoustic PML at step {}", time_step);
        let (_nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);

        let sigma_y = Self::damping_profile(self.thickness, ny, dy, self.sigma_max_acoustic);
        let sigma_z = Self::damping_profile(self.thickness, nz, dz, self.sigma_max_acoustic);

        Zip::indexed(&mut *field).for_each(|(i, j, k), val| {
            let damping = self.acoustic_damping[i] + sigma_y[j] + sigma_z[k];
            if damping > 0.0 {
                *val *= (-damping * dx).exp();
            }
        });

        if time_step % 10 == 0 {
            trace!(
                "Acoustic PML center sample: {}",
                field[[grid.nx / 2, ny / 2, nz / 2]]
            );
        }
    }

    fn apply_acoustic_freq(&self, field: &mut Array3<Complex<f64>>, grid: &Grid, time_step: usize) {
        debug!(
            "Applying frequency domain acoustic PML at step {}",
            time_step
        );
        let (_nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);

        let sigma_y = Self::damping_profile(self.thickness, ny, dy, self.sigma_max_acoustic);
        let sigma_z = Self::damping_profile(self.thickness, nz, dz, self.sigma_max_acoustic);

        Zip::indexed(field).for_each(|(i, j, k), val| {
            let damping = self.acoustic_damping[i] + sigma_y[j] + sigma_z[k];
            if damping > 0.0 {
                let decay = (-damping * dx).exp();
                *val = Complex {
                    re: val.re * decay,
                    im: val.im * decay,
                };
            }
        });
    }

    fn apply_light(&self, field: &mut Array3<f64>, grid: &Grid, time_step: usize) {
        debug!("Applying light PML at step {}", time_step);
        let (_nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);

        let sigma_y = Self::damping_profile(self.thickness, ny, dy, self.sigma_max_light);
        let sigma_z = Self::damping_profile(self.thickness, nz, dz, self.sigma_max_light);

        Zip::indexed(&mut *field).for_each(|(i, j, k), val| {
            let damping = self.light_damping[i] + sigma_y[j] + sigma_z[k];
            if damping > 0.0 {
                *val *= (-damping * dx).exp();
            }
        });

        if time_step % 10 == 0 {
            trace!(
                "Light PML center sample: {}",
                field[[grid.nx / 2, ny / 2, nz / 2]]
            );
        }
    }
}
