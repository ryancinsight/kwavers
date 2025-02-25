// physics/thermodynamics/heat_transfer/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::utils::laplacian;
use log::{debug, trace};
use ndarray::{Array3, Array4, Axis, Zip, s};
use rayon::prelude::*;
use std::time::{Instant};
use std::cmp::min;

pub const TEMPERATURE_IDX: usize = 2;

#[derive(Debug)]
pub struct ThermalModel {
    pub temperature: Array3<f64>,
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
        debug!(
            "Initializing ThermalModel with tau_q = {}, tau_t = {}",
            tau_q, tau_t
        );
        Self {
            temperature: Array3::from_elem((grid.nx, grid.ny, grid.nz), initial_temp),
            tau_q,
            tau_t,
            // Initialize performance metrics
            update_time: 0.0,
            heat_source_time: 0.0,
            laplacian_time: 0.0,
            diffusion_time: 0.0,
            call_count: 0,
            // Start with no precomputed arrays
            thermal_factor: None,
        }
    }

    /// Precomputes thermal factors for better performance
    fn precompute_thermal_factors(&mut self, grid: &Grid, medium: &dyn Medium, dt: f64) {
        trace!("Precomputing thermal factors");
        let mut thermal_factor = Array3::<f64>::zeros(self.temperature.dim());
        
        // Compute thermal factors in parallel
        Zip::indexed(&mut thermal_factor).par_for_each(|(i, j, k), val| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            let k_thermal = medium.thermal_conductivity(x, y, z, grid);
            let rho = medium.density(x, y, z, grid);
            let cp = medium.specific_heat(x, y, z, grid);
            let d_alpha = medium.thermal_diffusivity(x, y, z, grid);
            
            // Precompute division factor
            *val = dt / (rho * cp);
        });
        
        self.thermal_factor = Some(thermal_factor);
    }

    /// Reports performance metrics for thermal model computation
    pub fn report_performance(&self) {
        if self.call_count == 0 {
            debug!("No calls to ThermalModel::update_thermal yet");
            return;
        }

        let total_time = self.update_time;
        let avg_time = total_time / self.call_count as f64;

        debug!(
            "ThermalModel performance (avg over {} calls):",
            self.call_count
        );
        debug!("  Total time per call:   {:.3e} s", avg_time);
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
        
        debug!("Updating thermal effects with conductivity gradients");
        
        // Lazily initialize thermal factors if not already done
        if self.thermal_factor.is_none() {
            self.precompute_thermal_factors(grid, medium, dt);
        }
        let thermal_factor = self.thermal_factor.as_ref().unwrap();

        // Create temporary arrays for computation
        let mut heat_source = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        // Calculate laplacian of temperature field
        let start_laplacian = Instant::now();
        let lap_t = laplacian(fields, TEMPERATURE_IDX, grid).expect("Laplacian failed");
        self.laplacian_time += start_laplacian.elapsed().as_secs_f64();

        // Get views for pressure and light fields
        let pressure = fields.index_axis(Axis(0), 0); // PRESSURE_IDX
        let light = fields.index_axis(Axis(0), 1); // LIGHT_IDX
        
        // Calculate heat source term
        let start_heat_source = Instant::now();
        
        // Process with regular loops instead of using rayon parallel chunks
        for i in 0..grid.nx {
            let x = i as f64 * grid.dx;
            
            for j in 0..grid.ny {
                let y = j as f64 * grid.dy;
                
                for k in 0..grid.nz {
                    let z = k as f64 * grid.dz;
                    
                    // Cache medium properties to avoid repeated lookups
                    let alpha = medium.absorption_coefficient(x, y, z, grid, frequency);
                    let mu_a = medium.absorption_coefficient_light(x, y, z, grid);
                    let rho = medium.density(x, y, z, grid);
                    let c = medium.sound_speed(x, y, z, grid);
                    
                    // Acoustic heating: Q = 2*alpha*p^2 / (rho*c)
                    let p_val = pressure[[i, j, k]];
                    let acoustic_heating = 2.0 * alpha * p_val * p_val / (rho * c);
                    
                    // Optical heating: Q = mu_a * L
                    let light_val = light[[i, j, k]];
                    let optical_heating = mu_a * light_val;
                    
                    // Total heating
                    heat_source[[i, j, k]] = acoustic_heating + optical_heating;
                    
                    // In a real implementation, we might add more heat source terms:
                    // - Chemical reactions
                    // - Metabolic heat generation (for biological tissues)
                    // - External heating sources (e.g., radiofrequency, microwave)
                }
            }
        }
        
        self.heat_source_time += start_heat_source.elapsed().as_secs_f64();
        
        // Update temperature field with explicit scheme
        let start_diffusion = Instant::now();
        
        // Create temporary array for new temperature values
        let mut temp_new = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        // Update temperature with the heat equation
        Zip::indexed(&mut temp_new)
            .and(&self.temperature)
            .and(&lap_t)
            .and(&heat_source)
            .and(thermal_factor)
            .par_for_each(|(i, j, k), t_new, &t_old, &lap, &q, &factor| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                
                // Get medium properties
                let k_thermal = medium.thermal_conductivity(x, y, z, grid);
                let d_alpha = medium.thermal_diffusivity(x, y, z, grid);
                
                // Use precomputed factor for faster computation
                let term1 = d_alpha * lap;
                let term2 = q * (1.0 - self.tau_q / dt);
                let term3 = self.tau_t * k_thermal * lap;
                
                *t_new = t_old + (term1 + term2 + term3) * factor;
                
                // Use branchless approach for stability
                let is_valid = t_new.is_finite() && *t_new > 0.0;
                *t_new = if is_valid { *t_new } else { t_old };
            });

        self.diffusion_time += start_diffusion.elapsed().as_secs_f64();

        // Update temperature field in our model
        self.temperature.assign(&temp_new);
        fields.index_axis_mut(Axis(0), TEMPERATURE_IDX).assign(&self.temperature);
        
        // Note: Medium temperature update would be handled externally since we don't have mutable access
        // Previously: medium.update_temperature(&self.temperature);
        
        self.update_time += start_total.elapsed().as_secs_f64();
    }

    pub fn temperature(&self) -> &Array3<f64> {
        &self.temperature
    }
}