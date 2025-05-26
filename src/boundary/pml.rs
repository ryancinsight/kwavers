use crate::boundary::Boundary;
use crate::grid::Grid;
use crate::medium::Medium;
use log::{debug, trace};
use ndarray::{Array3, Zip};
use rayon::prelude::*;
use rustfft::num_complex::Complex;

/// Perfectly Matched Layer (PML) boundary condition for absorbing outgoing waves.
///
/// This implementation uses a polynomial grading of the absorption profile
/// with optional backing by a theoretical model for automatic parameter selection.
#[derive(Debug, Clone)]
pub struct PMLBoundary {
    // thickness: usize, // Removed
    // sigma_max_acoustic: f64, // Removed
    // sigma_max_light: f64, // Removed
    /// Pre-computed damping profiles for each dimension
    acoustic_damping_x: Vec<f64>,
    acoustic_damping_y: Vec<f64>,
    acoustic_damping_z: Vec<f64>,
    light_damping_x: Vec<f64>,
    light_damping_y: Vec<f64>,
    light_damping_z: Vec<f64>,
    // polynomial_order: usize, // Removed
    // target_reflection: f64, // Removed
    /// Pre-computed combined damping factors for optimization
    acoustic_damping_3d: Option<Array3<f64>>,
    light_damping_3d: Option<Array3<f64>>,
}

impl PMLBoundary {
    /// Creates a new PML boundary with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `thickness` - PML thickness in grid points (0 for auto-selection)
    /// * `sigma_max_acoustic` - Maximum acoustic absorption coefficient (0 for auto-selection)
    /// * `sigma_max_light` - Maximum light absorption coefficient (0 for auto-selection)
    /// * `medium` - The simulation medium
    /// * `grid` - The simulation grid
    /// * `acoustic_freq` - Characteristic acoustic frequency
    /// * `polynomial_order` - Order of polynomial grading (default: 3)
    /// * `target_reflection` - Target theoretical reflection coefficient (default: 1e-6)
    pub fn new(
        thickness: usize,
        sigma_max_acoustic: f64,
        sigma_max_light: f64,
        medium: &dyn Medium,
        grid: &Grid,
        acoustic_freq: f64,
        polynomial_order: Option<usize>,
        target_reflection: Option<f64>,
    ) -> Self {
        let poly_order = polynomial_order.unwrap_or(3);
        let reflection = target_reflection.unwrap_or(1e-6);
        
        debug!(
            "Initializing PMLBoundary: sigma_acoustic = {}, sigma_light = {}, polynomial_order = {}",
            sigma_max_acoustic, sigma_max_light, poly_order
        );

        // Calculate optimal PML thickness based on acoustic wavelength
        let c = medium.sound_speed(0.0, 0.0, 0.0, grid);
        let wavelength = c / acoustic_freq;
        let acoustic_thickness = (wavelength * 2.0 / grid.dx).ceil() as usize;

        // Calculate optimal PML thickness based on light diffusion length
        let mu_a = medium.absorption_coefficient_light(0.0, 0.0, 0.0, grid);
        let mu_s_prime = medium.reduced_scattering_coefficient_light(0.0, 0.0, 0.0, grid);
        let diffusion_length = 1.0 / (3.0 * (mu_a + mu_s_prime)).sqrt();
        let light_thickness = (diffusion_length * 5.0 / grid.dx).ceil() as usize;

        // Use provided thickness or auto-select based on physics
        let final_thickness = if thickness == 0 {
            acoustic_thickness.max(light_thickness).max(10) // Minimum 10 points for stability
        } else {
            thickness
        };

        // Auto-calculate optimal sigma if not provided
        let sigma_acoustic = if sigma_max_acoustic <= 0.0 {
            // Based on Komatitsch & Martin (2007)
            let poly_order_f64 = (poly_order + 1) as f64;
            let optimal_sigma = -poly_order_f64 * c * (reflection.ln()) / 
                                (2.0 * final_thickness as f64 * grid.dx);
            debug!("Auto-calculated acoustic sigma_max = {}", optimal_sigma);
            optimal_sigma
        } else {
            sigma_max_acoustic
        };

        let sigma_light = if sigma_max_light <= 0.0 {
            // For light, we use a similar approach but scaled by diffusion coefficient
            let d = 1.0 / (3.0 * (mu_a + mu_s_prime));
            let poly_order_f64 = (poly_order + 1) as f64;
            let optimal_sigma = -poly_order_f64 * d * (reflection.ln()) / 
                                (2.0 * final_thickness as f64 * grid.dx);
            debug!("Auto-calculated light sigma_max = {}", optimal_sigma);
            optimal_sigma
        } else {
            sigma_max_light
        };

        debug!(
            "PML thickness: acoustic = {}, light = {}, final = {}",
            acoustic_thickness, light_thickness, final_thickness
        );

        // Create damping profiles for each dimension
        let acoustic_damping_x = Self::damping_profile(final_thickness, grid.nx, grid.dx, sigma_acoustic, poly_order);
        let acoustic_damping_y = Self::damping_profile(final_thickness, grid.ny, grid.dy, sigma_acoustic, poly_order);
        let acoustic_damping_z = Self::damping_profile(final_thickness, grid.nz, grid.dz, sigma_acoustic, poly_order);
        
        let light_damping_x = Self::damping_profile(final_thickness, grid.nx, grid.dx, sigma_light, poly_order);
        let light_damping_y = Self::damping_profile(final_thickness, grid.ny, grid.dy, sigma_light, poly_order);
        let light_damping_z = Self::damping_profile(final_thickness, grid.nz, grid.dz, sigma_light, poly_order);

        Self {
            // thickness: final_thickness, // Removed
            // sigma_max_acoustic: sigma_acoustic, // Removed
            // sigma_max_light: sigma_light, // Removed
            acoustic_damping_x,
            acoustic_damping_y,
            acoustic_damping_z,
            light_damping_x,
            light_damping_y,
            light_damping_z,
            // polynomial_order: poly_order, // Removed
            // target_reflection: reflection, // Removed
            acoustic_damping_3d: None,
            light_damping_3d: None,
        }
    }

    /// Creates a damping profile for a PML layer.
    ///
    /// # Arguments
    ///
    /// * `thickness` - PML thickness in grid points
    /// * `length` - Total domain length in grid points
    /// * `dx` - Grid spacing
    /// * `sigma_max` - Maximum absorption coefficient
    /// * `order` - Polynomial order for profile grading
    fn damping_profile(thickness: usize, length: usize, _dx: f64, sigma_max: f64, order: usize) -> Vec<f64> {
        let mut profile = vec![0.0; length];
        
        // Apply PML at both domain boundaries (left/right or top/bottom)
        // Left/bottom boundary
        for i in 0..thickness {
            let normalized_distance = (thickness - i) as f64 / thickness as f64;
            profile[i] = sigma_max * normalized_distance.powi(order as i32);
        }
        
        // Right/top boundary
        for i in 0..thickness {
            let idx = length - i - 1;
            let normalized_distance = i as f64 / thickness as f64;
            profile[idx] = sigma_max * normalized_distance.powi(order as i32);
        }
        
        profile
    }
    
    /// Applies a pre-computed damping factor to a field value
    #[inline]
    fn apply_damping(val: &mut f64, damping: f64, dx: f64) {
        if damping > 0.0 {
            *val *= (-damping * dx).exp();
        }
    }
    
    /// Applies a pre-computed damping factor to a complex field value
    #[inline]
    fn apply_complex_damping(val: &mut Complex<f64>, damping: f64, dx: f64) {
        if damping > 0.0 {
            let decay = (-damping * dx).exp();
            val.re *= decay;
            val.im *= decay;
        }
    }

    /// Precomputes the 3D damping factors for acoustic fields to avoid repeated calculations
    fn precompute_acoustic_damping_3d(&mut self, grid: &Grid) {
        if self.acoustic_damping_3d.is_none() {
            trace!("Precomputing 3D acoustic damping factors");
            let mut damping_3d = Array3::zeros((grid.nx, grid.ny, grid.nz));
            
            Zip::indexed(&mut damping_3d).par_for_each(|(i, j, k), val| {
                *val = self.acoustic_damping_x[i] + self.acoustic_damping_y[j] + self.acoustic_damping_z[k];
            });
            
            self.acoustic_damping_3d = Some(damping_3d);
        }
    }
    
    /// Precomputes the 3D damping factors for light fields to avoid repeated calculations
    fn precompute_light_damping_3d(&mut self, grid: &Grid) {
        if self.light_damping_3d.is_none() {
            trace!("Precomputing 3D light damping factors");
            let mut damping_3d = Array3::zeros((grid.nx, grid.ny, grid.nz));
            
            Zip::indexed(&mut damping_3d).par_for_each(|(i, j, k), val| {
                *val = self.light_damping_x[i] + self.light_damping_y[j] + self.light_damping_z[k];
            });
            
            self.light_damping_3d = Some(damping_3d);
        }
    }
}

impl Boundary for PMLBoundary {
    fn apply_acoustic(&mut self, field: &mut Array3<f64>, grid: &Grid, time_step: usize) {
        trace!("Applying spatial acoustic PML at step {}", time_step);
        let dx = grid.dx;

        // Lazily initialize 3D damping factors if not computed yet
        self.precompute_acoustic_damping_3d(grid);
        let damping_3d = self.acoustic_damping_3d.as_ref().unwrap();
        
        // Apply damping in parallel using precomputed factors
        Zip::from(field)
            .and(damping_3d)
            .par_for_each(|val, &damping| {
                Self::apply_damping(val, damping, dx);
            });
    }

    fn apply_acoustic_freq(&mut self, field: &mut Array3<Complex<f64>>, grid: &Grid, time_step: usize) {
        trace!("Applying frequency domain acoustic PML at step {}", time_step);
        let dx = grid.dx;

        // Lazily initialize 3D damping factors if not computed yet
        self.precompute_acoustic_damping_3d(grid);
        let damping_3d = self.acoustic_damping_3d.as_ref().unwrap();
        
        // Apply damping in parallel using precomputed factors
        Zip::from(field)
            .and(damping_3d)
            .par_for_each(|val, &damping| {
                Self::apply_complex_damping(val, damping, dx);
            });
    }

    fn apply_light(&mut self, field: &mut Array3<f64>, grid: &Grid, time_step: usize) {
        trace!("Applying light PML at step {}", time_step);
        let dx = grid.dx;

        // Lazily initialize 3D damping factors if not computed yet
        self.precompute_light_damping_3d(grid);
        let damping_3d = self.light_damping_3d.as_ref().unwrap();
        
        // Apply damping in parallel using precomputed factors
        Zip::from(field)
            .and(damping_3d)
            .par_for_each(|val, &damping| {
                Self::apply_damping(val, damping, dx);
            });
    }
}
