// physics/mechanics/cavitation/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::scattering::acoustic::{compute_bubble_interactions, compute_mie_scattering, compute_rayleigh_scattering};
use log::{debug, trace, info, warn};
use ndarray::{Array3, Axis, ArrayView3, ArrayViewMut3, Zip, s};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::ops::Range;
use std::cmp::min;

// Constants
const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;
const MIN_RADIUS: f64 = 1.0e-9;  // 1 nm minimum radius
const MAX_RADIUS: f64 = 1.0e-3;  // 1 mm maximum radius
const MAX_VELOCITY: f64 = 1.0e3; // 1000 m/s maximum velocity
const MAX_ACCELERATION: f64 = 1.0e12; // Maximum acceleration

#[derive(Debug)]
pub struct CavitationModel {
    pub radius: Array3<f64>,
    pub velocity: Array3<f64>,
    pub prev_velocity: Array3<f64>,
    pub temperature: Array3<f64>,
    // Cached values for better performance
    d2r_dt2: Array3<f64>,
    rayleigh_scatter: Array3<f64>,
    mie_scatter: Array3<f64>,
    interaction_scatter: Array3<f64>,
}

impl CavitationModel {
    pub fn new(grid: &Grid, initial_radius: f64) -> Self {
        debug!(
            "Initializing CavitationModel with initial radius = {:.6e} m",
            initial_radius
        );
        let dim = (grid.nx, grid.ny, grid.nz);
        Self {
            radius: Array3::from_elem(dim, initial_radius),
            velocity: Array3::zeros(dim),
            prev_velocity: Array3::zeros(dim),
            temperature: Array3::from_elem(dim, 293.15),
            // Pre-allocate arrays to avoid allocation in update
            d2r_dt2: Array3::zeros(dim),
            rayleigh_scatter: Array3::zeros(dim),
            mie_scatter: Array3::zeros(dim),
            interaction_scatter: Array3::zeros(dim),
        }
    }

    pub fn update_cavitation(
        &mut self,
        p_update: &mut Array3<f64>,
        p: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
        frequency: f64,
    ) -> Array3<f64> {
        // Safety parameters for stability
        const MAX_RADIUS: f64 = 2.0e-4; // 200 µm maximum bubble radius
        const MIN_RADIUS: f64 = 1.0e-10; // 0.1 nm minimum bubble radius
        const MAX_VELOCITY: f64 = 1.0e3; // 1000 m/s maximum bubble wall velocity
        const MAX_ACCELERATION: f64 = 1.0e12; // 1e12 m/s² maximum acceleration
        const MAX_PRESSURE: f64 = 5.0e7; // 50 MPa maximum allowed pressure for stability
        
        // Check for extremely high pressure values that might cause instability
        let mut has_extreme_pressure = false;
        let max_p_abs = p.iter().fold(0.0, |max_abs: f64, &val| max_abs.max(val.abs()));
        if max_p_abs > MAX_PRESSURE {
            has_extreme_pressure = true;
            debug!("Extreme pressure detected: {:.2e} Pa, using adaptive time step", max_p_abs);
        }
        
        // Adaptive time stepping for numerical stability
        let actual_dt = if has_extreme_pressure {
            // Use a smaller time step for stability when pressure is extremely high
            dt * (MAX_PRESSURE / max_p_abs).min(0.5).max(0.01)
        } else {
            dt
        };
        
        // Calculate second derivative with safety checks
        self.calculate_second_derivative(p, grid, medium, frequency, actual_dt);
        
        // Apply safety limits to acceleration values
        for val in self.d2r_dt2.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            } else {
                *val = val.clamp(-MAX_ACCELERATION, MAX_ACCELERATION);
            }
        }
        
        // Update bubble dynamics with safety checks
        self.update_bubble_dynamics(actual_dt);
        
        // Limit bubble radius and velocity to physical ranges
        Zip::from(&mut self.radius)
            .and(&mut self.velocity)
            .for_each(|r, v| {
                // Clamp radius to physical range
                *r = r.clamp(MIN_RADIUS, MAX_RADIUS);
                
                // Clamp velocity to physical range
                *v = v.clamp(-MAX_VELOCITY, MAX_VELOCITY);
                
                // Reset velocity if radius hits limits (prevents further expansion/contraction)
                if *r == MIN_RADIUS && *v < 0.0 || *r == MAX_RADIUS && *v > 0.0 {
                    *v = 0.0;
                }
            });
        
        // Calculate acoustic effects from bubble oscillation
        let light_emission = self.calculate_acoustic_effects(p_update, p, grid, medium, has_extreme_pressure);
        
        // Return light emission for use in light propagation model
        light_emission
    }
    
    fn calculate_second_derivative(
        &mut self,
        p: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        frequency: f64,
        dt: f64,
    ) {
        // Reset acceleration array
        self.d2r_dt2.fill(0.0);
        
        // Process each point separately
        for i in 0..grid.nx {
            let x = i as f64 * grid.dx;
            
            for j in 0..grid.ny {
                let y = j as f64 * grid.dy;
                
                for k in 0..grid.nz {
                    let z = k as f64 * grid.dz;
                    
                    // Cache medium properties for this point
                    let rho = medium.density(x, y, z, grid);
                    let mu = medium.viscosity(x, y, z, grid);
                    let sigma = medium.surface_tension(x, y, z, grid);
                    let p0 = medium.ambient_pressure(x, y, z, grid);
                    let pv = medium.vapor_pressure(x, y, z, grid);
                    let gamma = medium.polytropic_index(x, y, z, grid);
                    let kappa = medium.thermal_conductivity(x, y, z, grid);
                    let dg = medium.gas_diffusion_coefficient(x, y, z, grid);
                    let medium_temp = medium.temperature()[[i, j, k]];
                    
                    // Get current bubble state
                    let r = self.radius[[i, j, k]];
                    let v = self.velocity[[i, j, k]];
                    let t_bubble = self.temperature[[i, j, k]];
                    let mut p_val = p[[i, j, k]];
                    
                    // Safety check - clamp pressure values to prevent instability
                    if !p_val.is_finite() || p_val.abs() > 1.0e9 {
                        p_val = p_val.clamp(-1.0e9, 1.0e9);
                        if !p_val.is_finite() {
                            p_val = 0.0;
                        }
                    }
                    
                    // Ensure radius is not too small (avoid division by zero)
                    let r_clamped = r.max(MIN_RADIUS);
                    let r0 = 10e-6; // Reference radius
                    
                    // Calculate pressure components with safety checks
                    let r_ratio = (r0 / r_clamped).min(1.0e3); // Limit ratio to prevent overflow
                    let p_gas = (p0 + 2.0 * sigma / r0 - pv) * r_ratio.powf(3.0 * gamma);
                    
                    // Calculate damping terms
                    let r_inv = 1.0 / r_clamped;
                    let viscous_term = 4.0 * mu * v * r_inv;
                    let surface_term = 2.0 * sigma * r_inv;
                    
                    // Calculate thermal effects
                    let temp_diff = (t_bubble - medium_temp).clamp(-100.0, 100.0); // Limit temperature difference
                    let thermal_damping = 3.0 * gamma * kappa * temp_diff * r_inv * r_inv;
                    
                    // Calculate diffusion effects with safety check
                    let p_diff = (p0 - p_gas).clamp(-1.0e9, 1.0e9); // Limit pressure difference
                    let diffusion_term = dg * p_diff * r_inv / rho;
                    
                    // Rayleigh-Plesset equation with safety checks
                    let pressure_diff = (p_gas + pv - p_val).clamp(-1.0e9, 1.0e9);
                    let rhs = (pressure_diff - viscous_term - surface_term - thermal_damping - diffusion_term) / rho;
                    let velocity_term = 1.5 * v.powi(2).min(1.0e6); // Limit squared velocity term
                    
                    // Calculate acceleration with validity check and limiting
                    let d2r = (rhs - velocity_term) * r_inv;
                    
                    // Limit acceleration to prevent instability
                    self.d2r_dt2[[i, j, k]] = if d2r.is_finite() { 
                        d2r.clamp(-MAX_ACCELERATION, MAX_ACCELERATION) 
                    } else { 
                        0.0 
                    };
                }
            }
        }
        
        // Add this at the end of the method to capture any NaN or infinite values
        for val in self.d2r_dt2.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
                debug!("NaN/Inf found in bubble acceleration, reset to 0");
            }
        }
    }
    
    fn update_bubble_dynamics(&mut self, dt: f64) {
        // Create temporary arrays for updated values
        let mut new_radius = self.radius.clone();
        let mut new_velocity = Array3::zeros(self.velocity.dim());
        
        // Safety check - use adaptive time stepping if needed
        let actual_dt = if self.d2r_dt2.iter().any(|&v| v.abs() > 1.0e10) {
            // If acceleration is too high, use smaller time step
            debug!("Using reduced time step for bubble dynamics due to high accelerations");
            dt * 0.1
        } else {
            dt
        };
        
        // Update velocity and radius based on acceleration
        for ((i, j, k), &d2r) in self.d2r_dt2.indexed_iter() {
            // Store previous velocity
            self.prev_velocity[[i, j, k]] = self.velocity[[i, j, k]];
            
            // Update velocity using acceleration with limiting
            let new_vel = self.velocity[[i, j, k]] + d2r * actual_dt;
            new_velocity[[i, j, k]] = new_vel.clamp(-MAX_VELOCITY, MAX_VELOCITY);
            
            // Update radius using new velocity with limiting
            let new_rad = self.radius[[i, j, k]] + new_velocity[[i, j, k]] * actual_dt;
            new_radius[[i, j, k]] = new_rad.clamp(MIN_RADIUS, MAX_RADIUS);
        }
        
        // Update stored state
        self.velocity = new_velocity;
        self.radius = new_radius;
        
        // Add additional stability check at the end of the method
        let mut nan_count = 0;
        let mut clamped_count = 0;
        
        for ((r, v), a) in self.radius.iter_mut()
            .zip(self.velocity.iter_mut())
            .zip(self.d2r_dt2.iter())
        {
            // Handle NaN/Inf values
            if r.is_nan() || r.is_infinite() {
                *r = 5.0e-6; // Reset to reasonable default (~5 µm)
                nan_count += 1;
            }
            
            if v.is_nan() || v.is_infinite() {
                *v = 0.0; // Reset velocity
                nan_count += 1;
            }
            
            // Physical constraints
            const MAX_RADIUS: f64 = 2.0e-4; // 200 µm
            const MIN_RADIUS: f64 = 1.0e-10; // 0.1 nm
            const MAX_VELOCITY: f64 = 1.0e3; // 1000 m/s
            
            // Apply constraints to prevent instability
            if *r > MAX_RADIUS {
                *r = MAX_RADIUS;
                *v = v.min(0.0); // Only allow contraction
                clamped_count += 1;
            } else if *r < MIN_RADIUS {
                *r = MIN_RADIUS;
                *v = v.max(0.0); // Only allow expansion
                clamped_count += 1;
            }
            
            // Velocity clamping
            if v.abs() > MAX_VELOCITY {
                *v = v.signum() * MAX_VELOCITY;
                clamped_count += 1;
            }
            
            // Additional dampening for stability when acceleration is extreme
            if a.abs() > 1.0e10 {
                *v *= 0.9; // Add damping
            }
        }
        
        if nan_count > 0 || clamped_count > 0 {
            debug!(
                "Bubble dynamics stabilized: {} NaN/Inf reset, {} values clamped to physical range",
                nan_count,
                clamped_count
            );
        }
    }
    
    fn calculate_acoustic_effects(
        &mut self,
        p_update: &mut Array3<f64>,
        p: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        has_extreme_pressure: bool,
    ) -> Array3<f64> {
        // Reset scatter arrays
        self.rayleigh_scatter.fill(0.0);
        self.mie_scatter.fill(0.0);
        self.interaction_scatter.fill(0.0);
        
        // Compute different scattering contributions
        compute_rayleigh_scattering(&mut self.rayleigh_scatter, &self.radius, p, grid, medium, 1.0e6);
        compute_mie_scattering(&mut self.mie_scatter, &self.radius, p, grid, medium, 1.0e6);
        compute_bubble_interactions(&mut self.interaction_scatter, &self.radius, &self.velocity, p, grid, medium, 1.0e6);
        
        // Update pressure field with volume change and scattering
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let r = self.radius[[i, j, k]];
                    let v = self.velocity[[i, j, k]];
                    let total_scatter = self.rayleigh_scatter[[i, j, k]] + 
                                        self.mie_scatter[[i, j, k]] + 
                                        self.interaction_scatter[[i, j, k]];
                    
                    // We need dt from the update_cavitation method, so pass it as a parameter
                    let dx = grid.dx;
                    let dy = grid.dy;
                    let dz = grid.dz;
                    let cell_volume = dx * dy * dz;
                    let d_volume_dt = 4.0 * PI * r.powi(2) * v;
                    
                    // Adjust pressure based on volume change rate and acoustic scattering
                    p_update[[i, j, k]] -= d_volume_dt / cell_volume + total_scatter;
                    
                    // Check for NaN/Inf
                    if !p_update[[i, j, k]].is_finite() {
                        p_update[[i, j, k]] = 0.0;
                    }
                }
            }
        }
        
        let mut light_source = Array3::zeros(p.dim());
        self.calculate_light_emission(&mut light_source, grid, medium, 1.0e-3);
        
        light_source
    }
    
    fn calculate_light_emission(
        &mut self,
        light_source: &mut Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let r = self.radius[[i, j, k]];
                    let v = self.velocity[[i, j, k]];
                    let v_prev = self.prev_velocity[[i, j, k]];
                    let t_bubble = &mut self.temperature[[i, j, k]];
                    let medium_temp = medium.temperature()[[i, j, k]];
                    
                    // Handle bubble collapse/rebound and light emission
                    if v_prev < 0.0 && v >= 0.0 {
                        // Bubble at minimum radius (collapse) - calculate temperature and light emission
                        let r0 = 10e-6; // Reference radius
                        let r_clamped = r.max(MIN_RADIUS);
                        let gamma = medium.polytropic_index(i as f64 * grid.dx, j as f64 * grid.dy, k as f64 * grid.dz, grid);
                        let t_max = *t_bubble * (r0 / r_clamped).powf(3.0 * (gamma - 1.0));
                        *t_bubble = t_max;
                        
                        // Calculate and store light emission
                        let cell_volume = grid.dx * grid.dy * grid.dz;
                        let power = 4.0 * PI * r_clamped.powi(2) * STEFAN_BOLTZMANN * t_max.powi(4);
                        light_source[[i, j, k]] = power / cell_volume;
                    } else {
                        // No light emission during expansion
                        light_source[[i, j, k]] = 0.0;
                        
                        // Cool bubble through thermal conduction
                        let kappa = medium.thermal_conductivity(i as f64 * grid.dx, j as f64 * grid.dy, k as f64 * grid.dz, grid);
                        let rho = medium.density(i as f64 * grid.dx, j as f64 * grid.dy, k as f64 * grid.dz, grid);
                        let r_clamped = r.max(MIN_RADIUS);
                        let temp_diff = *t_bubble - medium_temp;
                        
                        *t_bubble -= kappa * temp_diff * dt / (r_clamped * rho);
                        *t_bubble = t_bubble.max(medium_temp);
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