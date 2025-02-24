// physics/optics/diffusion/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::optics::{PolarizationModel, OpticalThermalModel, polarization::SimplePolarizationModel};
use crate::physics::scattering::optic::{OpticalScatteringModel, rayleigh::RayleighOpticalScatteringModel};
use crate::utils::{fft_3d, ifft_3d};
use log::debug;
use ndarray::{Array3, Array4, Axis, Zip};
use rustfft::num_complex::Complex;

pub const LIGHT_IDX: usize = 1;

#[derive(Debug)]
pub struct LightDiffusion {
    pub fluence_rate: Array4<f64>,
    pub emission_spectrum: Array3<f64>,
    polarization: Option<Box<dyn PolarizationModel>>,
    scattering: Option<Box<dyn OpticalScatteringModel>>,
    thermal: Option<OpticalThermalModel>,
    enable_polarization: bool,
    enable_scattering: bool,
    enable_thermal: bool,
}

impl LightDiffusion {
    pub fn new(
        grid: &Grid,
        enable_polarization: bool,
        enable_scattering: bool,
        enable_thermal: bool,
    ) -> Self {
        debug!(
            "Initializing LightDiffusion, polarization: {}, scattering: {}, thermal: {}",
            enable_polarization, enable_scattering, enable_thermal
        );
        Self {
            fluence_rate: Array4::zeros((1, grid.nx, grid.ny, grid.nz)),
            emission_spectrum: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            polarization: if enable_polarization {
                Some(Box::new(SimplePolarizationModel::new()))
            } else {
                None
            },
            scattering: if enable_scattering {
                Some(Box::new(RayleighOpticalScatteringModel::new()))
            } else {
                None
            },
            thermal: if enable_thermal {
                Some(OpticalThermalModel::new(grid))
            } else {
                None
            },
            enable_polarization,
            enable_scattering,
            enable_thermal,
        }
    }

    pub fn update_light(
        &mut self,
        fields: &mut Array4<f64>,
        light_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        debug!("Updating light diffusion with integrated effects");

        let mut fluence_fft = fft_3d(fields, LIGHT_IDX, grid);
        let k2 = grid.k_squared();

        Zip::indexed(&mut fluence_fft)
            .and(&k2)
            .and(light_source)
            .par_for_each(|(i, j, k), f, &k_val, &s_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let mu_a = medium.absorption_coefficient_light(x, y, z, grid);
                let mu_s_prime = medium.reduced_scattering_coefficient_light(x, y, z, grid);
                let d = 1.0 / (3.0 * (mu_a + mu_s_prime));
                let denom = 1.0 + dt * (d * k_val + mu_a);
                *f = (*f + Complex::new(dt * s_val, 0.0)) / denom;
            });

        let mut fluence = ifft_3d(&fluence_fft, grid);

        Zip::from(&mut self.emission_spectrum)
            .and(light_source)
            .par_for_each(|spec, &source| {
                *spec = source.max(0.0) * 1e-9;
            });

        if self.enable_polarization {
            if let Some(pol) = &mut self.polarization {
                pol.apply_polarization(&mut fluence, &self.emission_spectrum, grid, medium);
            }
        }

        if self.enable_scattering {
            if let Some(scat) = &mut self.scattering {
                scat.apply_scattering(&mut fluence, grid, medium);
            }
        }

        if self.enable_thermal {
            if let Some(therm) = &mut self.thermal {
                therm.update_thermal(fields, &fluence, grid, medium, dt);
            }
        }

        fields.index_axis_mut(Axis(0), LIGHT_IDX).assign(&fluence);
    }

    pub fn fluence_rate(&self) -> &Array4<f64> {
        &self.fluence_rate
    }

    pub fn emission_spectrum(&self) -> &Array3<f64> {
        &self.emission_spectrum
    }
}