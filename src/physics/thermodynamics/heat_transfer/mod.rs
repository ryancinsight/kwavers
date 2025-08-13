//! Heat Transfer Models for Thermal Effects
//!
//! **DEPRECATED**: This module is being phased out in favor of the plugin architecture.
//! For new code, use the plugin-based thermal diffusion solver in `crate::solver::thermal_diffusion`
//! or create a thermal physics plugin using the `PhysicsPlugin` trait.

use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_indices::TEMPERATURE_IDX;
use crate::physics::traits::ThermalModelTrait;
use log::debug;
use ndarray::{Array3, Array4, Axis, Zip};

use std::time::Instant;


#[deprecated(since = "1.4.0", note = "Use crate::solver::thermal_diffusion::ThermalDiffusionSolver or create a thermal physics plugin")]
#[derive(Debug)]
pub struct ThermalModel {
    temperature: Array3<f64>, // Changed to private (or pub(crate) by default within module)
    pub tau_q: f64, // Thermal relaxation time for heat flux
    pub tau_t: f64, // Thermal relaxation time for temperature
    // Performance metrics
    update_time: f64,
    heat_source_time: f64,
    laplacian_time: f64,
    diffusion_time: f64,
    call_count: usize,
    // Precomputed arrays for performance
    thermal_factor: Option<Array3<f64>>,
}

impl ThermalModel {
    pub fn new(grid: &Grid, initial_temp: f64, tau_q: f64, tau_t: f64) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        
        Self {
            temperature: Array3::from_elem((nx, ny, nz), initial_temp),
            tau_q,
            tau_t,
            update_time: 0.0,
            heat_source_time: 0.0,
            laplacian_time: 0.0,
            diffusion_time: 0.0,
            call_count: 0,
            thermal_factor: None,
        }
    }

    fn precompute_thermal_factors(&mut self, grid: &Grid, medium: &dyn Medium, dt: f64) {
        let start_time = Instant::now();
        let mut thermal_factor = Array3::<f64>::zeros(self.temperature.dim());
        
        // Compute thermal factors in parallel
        Zip::indexed(&mut thermal_factor).for_each(|(i, j, k), val| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            let k_thermal = medium.thermal_conductivity(x, y, z, grid);
            let d_alpha = medium.thermal_diffusivity(x, y, z, grid);
            *val = d_alpha * dt / (1.0 + self.tau_t * k_thermal);
        });
        
        self.thermal_factor = Some(thermal_factor);
        self.laplacian_time += start_time.elapsed().as_secs_f64();
    }

    pub fn report_performance(&self) {
        if self.call_count == 0 {
            debug!("No calls to ThermalModel::update_thermal yet");
            return;
        }

        let total_time = self.update_time;
        let avg_time = if self.call_count > 0 { total_time / self.call_count as f64} else {0.0};

        debug!(
            "ThermalModel performance (avg over {} calls):",
            self.call_count
        );
        debug!("  Total time per call:   {:.3e} s", avg_time);
        if total_time > 0.0 {
            debug!(
                "  Heat source calc:       {:.3e} s ({:.1}%)",
                self.heat_source_time / self.call_count as f64,
                100.0 * self.heat_source_time / total_time
            );
            debug!(
                "  Laplacian:              {:.3e} s ({:.1}%)",
                self.laplacian_time / self.call_count as f64,
                100.0 * self.laplacian_time / total_time
            );
            debug!(
                "  Diffusion update:       {:.3e} s ({:.1}%)",
                self.diffusion_time / self.call_count as f64,
                100.0 * self.diffusion_time / total_time
            );
        } else {
            debug!("  Detailed breakdown not available (total_time or call_count is zero).");
            debug!("    Heat source calc:       {:.3e} s", self.heat_source_time / self.call_count.max(1) as f64);
            debug!("    Laplacian:              {:.3e} s", self.laplacian_time / self.call_count.max(1) as f64);
            debug!("    Diffusion update:       {:.3e} s", self.diffusion_time / self.call_count.max(1) as f64);
        }
    }

    pub fn update_thermal(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        frequency: f64,
    ) {
        let start_total = Instant::now();
        self.call_count += 1;

        // Precompute thermal factors if needed
        if self.thermal_factor.is_none() {
            self.precompute_thermal_factors(grid, medium, dt);
        }

        let start_heat_source = Instant::now();
        let pressure = fields.index_axis(Axis(0), 0); // Pressure field
        let light = fields.index_axis(Axis(0), 1); // Light field
        
        // Compute heat source from acoustic and optical absorption
        let mut heat_source = grid.zeros_array();
        ndarray::Zip::indexed(&mut heat_source)
            .and(&pressure)
            .and(&light)
            .for_each(|(i, j, k), heat, &p_val, &light_val| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let alpha = medium.absorption_coefficient(x, y, z, grid, frequency);
                let mu_a = medium.absorption_coefficient_light(x, y, z, grid);
                let rho = medium.density(x, y, z, grid);
                let c = medium.sound_speed(x, y, z, grid);
                let acoustic_heating = 2.0 * alpha * p_val * p_val / (rho * c);
                let optical_heating = mu_a * light_val;
                *heat = acoustic_heating + optical_heating;
            });
        self.heat_source_time += start_heat_source.elapsed().as_secs_f64();

        let start_diffusion = Instant::now();
        let mut updated_temperature = grid.zeros_array();

        // Compute Laplacian of temperature using efficient ndarray operations
        let mut lap_t = grid.zeros_array();
        
        // Calculate thermal diffusivity array for heterogeneous media
        let mut diffusivity_array = grid.zeros_array();
        Zip::indexed(&mut diffusivity_array).for_each(|(i, j, k), diff| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            *diff = medium.thermal_diffusivity(x, y, z, grid);
        });
        
        // Enhanced Laplacian calculation with proper boundary conditions
        // Interior points (second-order central differences)
        for i in 1..grid.nx-1 {
            for j in 1..grid.ny-1 {
                for k in 1..grid.nz-1 {
                    let t_ijk = self.temperature[[i, j, k]];
                    
                    // Temperature values for finite differences
                    let t_ip1 = self.temperature[[i+1, j, k]];
                    let t_im1 = self.temperature[[i-1, j, k]];
                    let t_jp1 = self.temperature[[i, j+1, k]];
                    let t_jm1 = self.temperature[[i, j-1, k]];
                    let t_kp1 = self.temperature[[i, j, k+1]];
                    let t_km1 = self.temperature[[i, j, k-1]];
                    
                    // Second derivatives with uniform grid spacing
                    let d2t_dx2 = (t_ip1 - 2.0 * t_ijk + t_im1) / (grid.dx * grid.dx);
                    let d2t_dy2 = (t_jp1 - 2.0 * t_ijk + t_jm1) / (grid.dy * grid.dy);
                    let d2t_dz2 = (t_kp1 - 2.0 * t_ijk + t_km1) / (grid.dz * grid.dz);
                    
                    // Handle anisotropic thermal conductivity if needed
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let k_thermal = medium.thermal_conductivity(x, y, z, grid);
                    let rho = medium.density(x, y, z, grid);
                    let cp = medium.specific_heat(x, y, z, grid);
                    
                    // Thermal diffusivity: α = k/(ρ*cp)
                    let alpha = k_thermal / (rho * cp);
                    
                    // Laplacian with thermal diffusivity
                    lap_t[[i, j, k]] = alpha * (d2t_dx2 + d2t_dy2 + d2t_dz2);
                }
            }
        }
        
        // Boundary conditions (Neumann - zero flux at boundaries)
        // This represents insulated boundaries which is appropriate for most tissue simulations
        
        // X boundaries (i=0 and i=nx-1)
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                // Left boundary (i=0): use forward difference
                let i = 0;
                let t_ijk = self.temperature[[i, j, k]];
                let t_ip1 = self.temperature[[i+1, j, k]];
                
                // Handle Y derivatives with boundary checks
                let d2t_dy2 = if j == 0 {
                    // Forward difference at j=0
                    let t_jp1 = self.temperature[[i, j+1, k]];
                    2.0 * (t_jp1 - t_ijk) / (grid.dy * grid.dy)
                } else if j == grid.ny - 1 {
                    // Backward difference at j=ny-1
                    let t_jm1 = self.temperature[[i, j-1, k]];
                    2.0 * (t_jm1 - t_ijk) / (grid.dy * grid.dy)
                } else {
                    // Central difference for interior
                    let t_jp1 = self.temperature[[i, j+1, k]];
                    let t_jm1 = self.temperature[[i, j-1, k]];
                    (t_jp1 - 2.0 * t_ijk + t_jm1) / (grid.dy * grid.dy)
                };
                
                // Handle Z derivatives with boundary checks
                let d2t_dz2 = if k == 0 {
                    // Forward difference at k=0
                    let t_kp1 = self.temperature[[i, j, k+1]];
                    2.0 * (t_kp1 - t_ijk) / (grid.dz * grid.dz)
                } else if k == grid.nz - 1 {
                    // Backward difference at k=nz-1
                    let t_km1 = self.temperature[[i, j, k-1]];
                    2.0 * (t_km1 - t_ijk) / (grid.dz * grid.dz)
                } else {
                    // Central difference for interior
                    let t_kp1 = self.temperature[[i, j, k+1]];
                    let t_km1 = self.temperature[[i, j, k-1]];
                    (t_kp1 - 2.0 * t_ijk + t_km1) / (grid.dz * grid.dz)
                };
                
                let d2t_dx2 = 2.0 * (t_ip1 - t_ijk) / (grid.dx * grid.dx); // Forward difference
                let alpha = diffusivity_array[[i, j, k]];
                lap_t[[i, j, k]] = alpha * (d2t_dx2 + d2t_dy2 + d2t_dz2);
                
                // Right boundary (i=nx-1): use backward difference  
                let i = grid.nx - 1;
                let t_ijk = self.temperature[[i, j, k]];
                let t_im1 = self.temperature[[i-1, j, k]];
                
                // Handle Y derivatives with boundary checks (same logic as above)
                let d2t_dy2 = if j == 0 {
                    let t_jp1 = self.temperature[[i, j+1, k]];
                    2.0 * (t_jp1 - t_ijk) / (grid.dy * grid.dy)
                } else if j == grid.ny - 1 {
                    let t_jm1 = self.temperature[[i, j-1, k]];
                    2.0 * (t_jm1 - t_ijk) / (grid.dy * grid.dy)
                } else {
                    let t_jp1 = self.temperature[[i, j+1, k]];
                    let t_jm1 = self.temperature[[i, j-1, k]];
                    (t_jp1 - 2.0 * t_ijk + t_jm1) / (grid.dy * grid.dy)
                };
                
                // Handle Z derivatives with boundary checks (same logic as above)
                let d2t_dz2 = if k == 0 {
                    let t_kp1 = self.temperature[[i, j, k+1]];
                    2.0 * (t_kp1 - t_ijk) / (grid.dz * grid.dz)
                } else if k == grid.nz - 1 {
                    let t_km1 = self.temperature[[i, j, k-1]];
                    2.0 * (t_km1 - t_ijk) / (grid.dz * grid.dz)
                } else {
                    let t_kp1 = self.temperature[[i, j, k+1]];
                    let t_km1 = self.temperature[[i, j, k-1]];
                    (t_kp1 - 2.0 * t_ijk + t_km1) / (grid.dz * grid.dz)
                };
                
                let d2t_dx2 = 2.0 * (t_im1 - t_ijk) / (grid.dx * grid.dx); // Backward difference
                let alpha = diffusivity_array[[i, j, k]];
                lap_t[[i, j, k]] = alpha * (d2t_dx2 + d2t_dy2 + d2t_dz2);
            }
        }
        
        // Y boundaries (j=0 and j=ny-1) - excluding corners already handled in X boundaries
        for i in 1..grid.nx-1 {
            for k in 0..grid.nz {
                // Bottom boundary (j=0): use forward difference
                let j = 0;
                let t_ijk = self.temperature[[i, j, k]];
                let t_ip1 = self.temperature[[i+1, j, k]];
                let t_im1 = self.temperature[[i-1, j, k]];
                let t_jp1 = self.temperature[[i, j+1, k]];
                
                // Handle Z derivatives with boundary checks
                let d2t_dz2 = if k == 0 {
                    let t_kp1 = self.temperature[[i, j, k+1]];
                    2.0 * (t_kp1 - t_ijk) / (grid.dz * grid.dz)
                } else if k == grid.nz - 1 {
                    let t_km1 = self.temperature[[i, j, k-1]];
                    2.0 * (t_km1 - t_ijk) / (grid.dz * grid.dz)
                } else {
                    let t_kp1 = self.temperature[[i, j, k+1]];
                    let t_km1 = self.temperature[[i, j, k-1]];
                    (t_kp1 - 2.0 * t_ijk + t_km1) / (grid.dz * grid.dz)
                };
                
                let d2t_dx2 = (t_ip1 - 2.0 * t_ijk + t_im1) / (grid.dx * grid.dx);
                let d2t_dy2 = 2.0 * (t_jp1 - t_ijk) / (grid.dy * grid.dy); // Forward difference
                let alpha = diffusivity_array[[i, j, k]];
                lap_t[[i, j, k]] = alpha * (d2t_dx2 + d2t_dy2 + d2t_dz2);
                
                // Top boundary (j=ny-1): use backward difference
                let j = grid.ny - 1;
                let t_ijk = self.temperature[[i, j, k]];
                let t_ip1 = self.temperature[[i+1, j, k]];
                let t_im1 = self.temperature[[i-1, j, k]];
                let t_jm1 = self.temperature[[i, j-1, k]];
                
                // Handle Z derivatives with boundary checks (same logic as above)
                let d2t_dz2 = if k == 0 {
                    let t_kp1 = self.temperature[[i, j, k+1]];
                    2.0 * (t_kp1 - t_ijk) / (grid.dz * grid.dz)
                } else if k == grid.nz - 1 {
                    let t_km1 = self.temperature[[i, j, k-1]];
                    2.0 * (t_km1 - t_ijk) / (grid.dz * grid.dz)
                } else {
                    let t_kp1 = self.temperature[[i, j, k+1]];
                    let t_km1 = self.temperature[[i, j, k-1]];
                    (t_kp1 - 2.0 * t_ijk + t_km1) / (grid.dz * grid.dz)
                };
                
                let d2t_dx2 = (t_ip1 - 2.0 * t_ijk + t_im1) / (grid.dx * grid.dx);
                let d2t_dy2 = 2.0 * (t_jm1 - t_ijk) / (grid.dy * grid.dy); // Backward difference
                let alpha = diffusivity_array[[i, j, k]];
                lap_t[[i, j, k]] = alpha * (d2t_dx2 + d2t_dy2 + d2t_dz2);
            }
        }
        
        // Z boundaries (k=0 and k=nz-1) - excluding edges already handled in X and Y boundaries
        for i in 1..grid.nx-1 {
            for j in 1..grid.ny-1 {
                // Front boundary (k=0): use forward difference
                let k = 0;
                let t_ijk = self.temperature[[i, j, k]];
                let t_ip1 = self.temperature[[i+1, j, k]];
                let t_im1 = self.temperature[[i-1, j, k]];
                let t_jp1 = self.temperature[[i, j+1, k]];
                let t_jm1 = self.temperature[[i, j-1, k]];
                let t_kp1 = self.temperature[[i, j, k+1]];
                
                let d2t_dx2 = (t_ip1 - 2.0 * t_ijk + t_im1) / (grid.dx * grid.dx);
                let d2t_dy2 = (t_jp1 - 2.0 * t_ijk + t_jm1) / (grid.dy * grid.dy);
                let d2t_dz2 = 2.0 * (t_kp1 - t_ijk) / (grid.dz * grid.dz); // Forward difference
                let alpha = diffusivity_array[[i, j, k]];
                lap_t[[i, j, k]] = alpha * (d2t_dx2 + d2t_dy2 + d2t_dz2);
                
                // Back boundary (k=nz-1): use backward difference
                let k = grid.nz - 1;
                let t_ijk = self.temperature[[i, j, k]];
                let t_ip1 = self.temperature[[i+1, j, k]];
                let t_im1 = self.temperature[[i-1, j, k]];
                let t_jp1 = self.temperature[[i, j+1, k]];
                let t_jm1 = self.temperature[[i, j-1, k]];
                let t_km1 = self.temperature[[i, j, k-1]];
                
                let d2t_dx2 = (t_ip1 - 2.0 * t_ijk + t_im1) / (grid.dx * grid.dx);
                let d2t_dy2 = (t_jp1 - 2.0 * t_ijk + t_jm1) / (grid.dy * grid.dy);
                let d2t_dz2 = 2.0 * (t_km1 - t_ijk) / (grid.dz * grid.dz); // Backward difference
                let alpha = diffusivity_array[[i, j, k]];
                lap_t[[i, j, k]] = alpha * (d2t_dx2 + d2t_dy2 + d2t_dz2);
            }
        }
        
        let thermal_factor = self.thermal_factor.as_ref().unwrap();

        Zip::indexed(&mut updated_temperature)
            .and(&self.temperature)
            .and(&lap_t)
            .and(&heat_source)
            .and(thermal_factor)
            .for_each(|(i, j, k), t_updated, &t_old, &lap, &q, &factor| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let k_thermal = medium.thermal_conductivity(x, y, z, grid);
                let d_alpha = medium.thermal_diffusivity(x, y, z, grid);
                let term1 = d_alpha * lap;
                let term2 = q * (1.0 - self.tau_q / dt);
                let term3 = self.tau_t * k_thermal * lap;
                *t_updated = t_old + (term1 + term2 + term3) * factor;
                let is_valid = t_updated.is_finite() && *t_updated > 0.0;
                *t_updated = if is_valid { *t_updated } else { t_old };
            });
        self.diffusion_time += start_diffusion.elapsed().as_secs_f64();

        self.temperature.assign(&updated_temperature);
        fields.index_axis_mut(Axis(0), TEMPERATURE_IDX).assign(&self.temperature);

        self.update_time += start_total.elapsed().as_secs_f64();
    }
}

impl ThermalModelTrait for ThermalModel {
    fn update_thermal(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        frequency: f64,
    ) {
        self.update_thermal(fields, grid, medium, dt, frequency);
    }

    fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }

    fn set_temperature(&mut self, new_temp: &Array3<f64>) {
        if self.temperature.shape() == new_temp.shape() {
            self.temperature.assign(new_temp);
        } else {
            debug!("Shape mismatch in set_temperature for ThermalModel. Current: {:?}, New: {:?}", self.temperature.shape(), new_temp.shape());
            // Ndarray's assign will panic if shapes are not compatible.
            self.temperature.assign(new_temp);
        }
    }

    fn report_performance(&self) {
        self.report_performance();
    }
}