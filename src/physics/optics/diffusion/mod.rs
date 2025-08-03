// physics/optics/diffusion/mod.rs
use ndarray::{Array3, Array4, Axis};
use crate::grid::Grid;
use crate::physics::optics::{PolarizationModel, polarization::SimplePolarizationModel};
use crate::physics::optics::thermal::OpticalThermalModel;
use crate::medium::Medium;
use crate::physics::scattering::optic::{OpticalScatteringModel, rayleigh::RayleighOpticalScatteringModel};
use log::debug;

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
        _medium: &dyn Medium,
        dt: f64,
    ) {
        let start_time = Instant::now();
        
        // Update the light field in the fields array
        let mut light_field = fields.index_axis_mut(Axis(0), LIGHT_IDX);
        
        // Enhanced diffusion equation solver: ∂φ/∂t = D∇²φ - μₐφ + S
        // where φ is photon fluence rate, D is diffusion coefficient, μₐ is absorption coefficient
        let diffusion_coefficient = 1e-3; // mm²/ns - typical tissue value
        let absorption_coeff = 0.1; // mm⁻¹ - typical tissue absorption
        
        // Create a temporary array to store the updated values
        let mut new_light_field = Array3::zeros(light_field.raw_dim());
        new_light_field.assign(&light_field);
        
        // Use optimized iteration for interior points
        let dx2_inv = 1.0 / (grid.dx * grid.dx);
        let dy2_inv = 1.0 / (grid.dy * grid.dy);
        let dz2_inv = 1.0 / (grid.dz * grid.dz);
        
        let (nx, ny, nz) = light_field.dim();
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Compute Laplacian using neighboring values
                    let center_val = light_field[[i, j, k]];
                    let source_term = light_source[[i, j, k]];
                    
                    let laplacian_phi = 
                        (light_field[[i+1, j, k]] - 2.0 * center_val + light_field[[i-1, j, k]]) * dx2_inv +
                        (light_field[[i, j+1, k]] - 2.0 * center_val + light_field[[i, j-1, k]]) * dy2_inv +
                        (light_field[[i, j, k+1]] - 2.0 * center_val + light_field[[i, j, k-1]]) * dz2_inv;
                    
                    // Update using diffusion equation: ∂φ/∂t = D∇²φ - μₐφ + S
                    let update = center_val + dt * (
                        diffusion_coefficient * laplacian_phi - 
                        absorption_coeff * center_val + 
                        source_term
                    );
                    
                    // Ensure non-negative values (physical constraint)
                    new_light_field[[i, j, k]] = update.max(0.0);
                }
            }
        }
        
        // Update the light field
        light_field.assign(&new_light_field);
        
        // Update fluence_rate to match
        self.fluence_rate.assign(fields);
        
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