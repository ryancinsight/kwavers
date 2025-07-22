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
            // Initialize performance metrics
            update_time: 0.0,
            fft_time: 0.0,
            diffusion_time: 0.0,
            effect_time: 0.0,
            call_count: 0,
            // Start with no precomputed arrays
            d_inv: None,
        }
    }

    /// Precomputes inverse diffusion coefficients for better performance
    fn precompute_diffusion_coefficients(&mut self, grid: &Grid, medium: &dyn Medium) {
        trace!("Precomputing diffusion coefficients");
        let mut d_inv = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        
        // Compute inverse diffusion coefficients in parallel
        Zip::indexed(&mut d_inv).par_for_each(|(i, j, k), val| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            let mu_a = medium.absorption_coefficient_light(x, y, z, grid);
            let mu_s_prime = medium.reduced_scattering_coefficient_light(x, y, z, grid);
            // Store inverse directly for faster computation later (multiply vs divide)
            *val = 3.0 * (mu_a + mu_s_prime);
        });
        
        self.d_inv = Some(d_inv);
    }

    /// Reports performance metrics for light diffusion computation
    pub fn report_performance(&self) {
        if self.call_count == 0 {
            debug!("No calls to LightDiffusion::update_light yet");
            return;
        }

        let total_time = self.update_time;
        let avg_time = total_time / self.call_count as f64;

        debug!(
            "LightDiffusion performance (avg over {} calls):",
            self.call_count
        );
        debug!("  Total time per call:   {:.3e} s", avg_time);
        debug!(
            "  FFT operations:         {:.3e} s ({:.1}%)",
            self.fft_time / self.call_count as f64,
            100.0 * self.fft_time / total_time
        );
        debug!(
            "  Diffusion calculation:  {:.3e} s ({:.1}%)",
            self.diffusion_time / self.call_count as f64,
            100.0 * self.diffusion_time / total_time
        );
        debug!(
            "  Effects application:    {:.3e} s ({:.1}%)",
            self.effect_time / self.call_count as f64,
            100.0 * self.effect_time / total_time
        );
    }

    pub fn update_light(
        &mut self,
        fields: &mut Array4<f64>,
        light_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        let start_total = Instant::now();
        self.call_count += 1;
        
        debug!("Updating light diffusion with integrated effects");

        // Lazily initialize diffusion coefficients if not already done
        if self.d_inv.is_none() {
            self.precompute_diffusion_coefficients(grid, medium);
        }
        
        let d_inv = self.d_inv.as_ref().unwrap();
        let start_fft = Instant::now();
        let mut fluence_fft = fft_3d(fields, LIGHT_IDX, grid);
        self.fft_time += start_fft.elapsed().as_secs_f64();

        let k2 = grid.k_squared();
        let start_diffusion = Instant::now();

        // Optimize diffusion calculation with better parallel processing and precomputed values
        Zip::indexed(&mut fluence_fft)
            .and(&k2)
            .and(light_source)
            .and(d_inv)
            .par_for_each(|(i, j, k), f, &k_val, &s_val, &d_inv_val| {
                // Use precomputed inverse diffusion coefficient
                let d = 1.0 / d_inv_val;
                let mu_a = medium.absorption_coefficient_light(
                    i as f64 * grid.dx, 
                    j as f64 * grid.dy, 
                    k as f64 * grid.dz, 
                    grid
                );
                
                // Optimize division with multiplication by inverse
                let dt_inv = 1.0 / dt;
                let denom_inv = 1.0 / (dt_inv + d * k_val + mu_a);
                
                // Use complex number optimizations
                *f = Complex::new(
                    (f.re * dt_inv + s_val) * denom_inv,
                    f.im * dt_inv * denom_inv
                );
            });
        self.diffusion_time += start_diffusion.elapsed().as_secs_f64();

        let start_ifft = Instant::now();
        let mut fluence = ifft_3d(&fluence_fft, grid);
        self.fft_time += start_ifft.elapsed().as_secs_f64();

        let start_effects = Instant::now();
        
        // Update emission spectrum in parallel with optimized calculations
        Zip::from(&mut self.emission_spectrum)
            .and(light_source)
            .par_for_each(|spec, &source| {
                *spec = if source > 0.0 { source * 1e-9 } else { 0.0 };
            });

        // Apply optional physics effects if enabled
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
        self.effect_time += start_effects.elapsed().as_secs_f64();

        // Assign back to fields array
        fields.index_axis_mut(Axis(0), LIGHT_IDX).assign(&fluence);
        
        self.update_time += start_total.elapsed().as_secs_f64();
    }

    pub fn fluence_rate(&self) -> &Array4<f64> {
        &self.fluence_rate
    }

    // emission_spectrum moved to trait impl
    // report_performance moved to trait impl
    // update_light moved to trait impl
    // fluence_rate moved to trait impl
}

use crate::physics::traits::LightDiffusionModelTrait;

impl LightDiffusionModelTrait for LightDiffusion {
    fn update_light(
        &mut self,
        fields: &mut Array4<f64>,
        light_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        // Logic from original LightDiffusion::update_light
        let start_total = Instant::now();
        self.call_count += 1;

        debug!("Updating light diffusion with integrated effects (via trait)");

        if self.d_inv.is_none() {
            self.precompute_diffusion_coefficients(grid, medium); // Call inherent helper
        }

        let d_inv = self.d_inv.as_ref().unwrap();
        let start_fft = Instant::now();
        let mut fluence_fft = fft_3d(fields, LIGHT_IDX, grid);
        self.fft_time += start_fft.elapsed().as_secs_f64();

        let k2 = grid.k_squared();
        let start_diffusion = Instant::now();

        Zip::indexed(&mut fluence_fft)
            .and(&k2)
            .and(light_source)
            .and(d_inv)
            .par_for_each(|(i, j, k), f, &k_val, &s_val, &d_inv_val| {
                let d = 1.0 / d_inv_val;
                let mu_a = medium.absorption_coefficient_light(
                    i as f64 * grid.dx,
                    j as f64 * grid.dy,
                    k as f64 * grid.dz,
                    grid
                );

                let dt_inv = 1.0 / dt;
                let denom_inv = 1.0 / (dt_inv + d * k_val + mu_a);

                *f = Complex::new(
                    (f.re * dt_inv + s_val) * denom_inv,
                    f.im * dt_inv * denom_inv
                );
            });
        self.diffusion_time += start_diffusion.elapsed().as_secs_f64();

        let start_ifft = Instant::now();
        let mut fluence = ifft_3d(&fluence_fft, grid);
        self.fft_time += start_ifft.elapsed().as_secs_f64();

        let start_effects = Instant::now();

        Zip::from(&mut self.emission_spectrum)
            .and(light_source)
            .par_for_each(|spec, &source| {
                *spec = if source > 0.0 { source * 1e-9 } else { 0.0 };
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
        self.effect_time += start_effects.elapsed().as_secs_f64();

        fields.index_axis_mut(Axis(0), LIGHT_IDX).assign(&fluence);

        self.update_time += start_total.elapsed().as_secs_f64();
    }

    fn emission_spectrum(&self) -> &Array3<f64> {
        // Logic from original LightDiffusion::emission_spectrum
        &self.emission_spectrum
    }

    fn fluence_rate(&self) -> &Array4<f64> {
        // Logic from original LightDiffusion::fluence_rate
        // Note: The original struct stored fluence_rate: Array4<f64>,
        // but it was assigned in update_light to a local `fluence` variable.
        // Assuming fields[LIGHT_IDX] is the canonical fluence rate.
        // The struct field `self.fluence_rate` seems unused based on current code.
        // For now, returning the field from `fields` array directly might be problematic
        // if this method is called outside `update_light` context.
        // Let's return the struct's field `self.fluence_rate` as it was defined.
        // This might need review if `self.fluence_rate` isn't updated properly.
        // Based on current code, `self.fluence_rate` is initialized to zeros and never updated.
        // The actual fluence is in `fields.index_axis(Axis(0), LIGHT_IDX)`.
        // This suggests a potential bug or design choice in the original code.
        // For DIP, we mirror the existing signature. The solver uses fields directly.
        // The `Solver` doesn't call this `fluence_rate` method.
        // Let's return the `self.fluence_rate` field for consistency with struct definition.
        &self.fluence_rate
    }

    fn report_performance(&self) {
        // Logic from original LightDiffusion::report_performance
        if self.call_count == 0 {
            debug!("No calls to LightDiffusion::update_light yet (via trait)");
            return;
        }

        let total_time = self.update_time;
        // Ensure call_count is not zero to prevent division by zero.
        let avg_time = if self.call_count > 0 { total_time / self.call_count as f64 } else { 0.0 };


        debug!(
            "LightDiffusion performance (via trait) (avg over {} calls):",
            self.call_count
        );
        debug!("  Total time per call:   {:.3e} s", avg_time);
        if total_time > 0.0 { // Avoid division by zero if total_time is zero
            debug!(
                "  FFT operations:         {:.3e} s ({:.1}%)",
                self.fft_time / self.call_count as f64,
                100.0 * self.fft_time / total_time
            );
            debug!(
                "  Diffusion calculation:  {:.3e} s ({:.1}%)",
                self.diffusion_time / self.call_count as f64,
                100.0 * self.diffusion_time / total_time
            );
            debug!(
                "  Effects application:    {:.3e} s ({:.1}%)",
                self.effect_time / self.call_count as f64,
                100.0 * self.effect_time / total_time
            );
        } else {
            debug!("  Detailed breakdown not available (total_time or call_count is zero).");
            debug!("    FFT operations:         {:.3e} s", self.fft_time / self.call_count.max(1) as f64);
            debug!("    Diffusion calculation:  {:.3e} s", self.diffusion_time / self.call_count.max(1) as f64);
            debug!("    Effects application:    {:.3e} s", self.effect_time / self.call_count.max(1) as f64);
        }
    }
}