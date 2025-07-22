// physics/optics/diffusion/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::optics::{PolarizationModel, OpticalThermalModel, polarization::SimplePolarizationModel};
use crate::physics::scattering::optic::{OpticalScatteringModel, rayleigh::RayleighOpticalScatteringModel};
use crate::utils::{fft_3d, ifft_3d};
use log::{debug, trace};
use ndarray::{Array3, Array4, Axis, Zip};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use std::time::Instant;
use crate::physics::traits::LightDiffusionModelTrait;

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
    // Performance metrics
    update_time: f64,
    fft_time: f64,
    diffusion_time: f64,
    effect_time: f64,
    call_count: usize,
    // Precomputed arrays for better performance
    d_inv: Option<Array3<f64>>,
}

impl LightDiffusion {
    pub fn new(
        grid: &Grid,
        enable_polarization: bool,
        enable_scattering: bool,
        enable_thermal: bool,
    ) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        
        Self {
            fluence_rate: Array4::zeros((1, nx, ny, nz)),
            emission_spectrum: Array3::zeros((nx, ny, nz)),
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
            update_time: 0.0,
            fft_time: 0.0,
            diffusion_time: 0.0,
            effect_time: 0.0,
            call_count: 0,
            d_inv: None,
        }
    }
}

impl LightDiffusionModelTrait for LightDiffusion {
    fn update_light(
        &mut self,
        fields: &mut Array4<f64>,
        light_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        let start_time = Instant::now();
        
        // Update the light field in the fields array
        let mut light_field = fields.index_axis_mut(Axis(0), LIGHT_IDX);
        
        // Simple diffusion update (placeholder implementation)
        // In a real implementation, this would solve the diffusion equation
        Zip::from(&mut light_field)
            .and(light_source)
            .for_each(|l, &s| {
                *l += s * dt;
            });
        
        // Update fluence_rate to match
        self.fluence_rate.assign(&fields);
        
        self.update_time = start_time.elapsed().as_secs_f64();
        self.call_count += 1;
    }

    fn emission_spectrum(&self) -> &Array3<f64> {
        &self.emission_spectrum
    }

    fn fluence_rate(&self) -> &Array4<f64> {
        &self.fluence_rate
    }

    fn report_performance(&self) {
        debug!(
            "LightDiffusion performance: update_time={:.6e}s, fft_time={:.6e}s, diffusion_time={:.6e}s, effect_time={:.6e}s, calls={}",
            self.update_time, self.fft_time, self.diffusion_time, self.effect_time, self.call_count
        );
    }
}