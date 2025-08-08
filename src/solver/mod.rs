// src/solver/mod.rs
pub mod pstd;
pub mod fdtd;
pub mod hybrid;
pub mod time_integration;
pub mod spectral_dg;
pub mod imex;
pub mod amr;
pub mod cpml_integration;
pub mod validation;
pub mod workspace;
pub mod time_reversal;
pub mod thermal_diffusion;

use crate::grid::Grid;
use crate::KwaversResult;
use crate::boundary::Boundary;
use crate::medium::Medium;
use crate::physics::{
    traits::{AcousticWaveModel, CavitationModelBehavior, LightDiffusionModelTrait, ThermalModelTrait, ChemicalModelTrait, StreamingModelTrait, AcousticScatteringModelTrait, HeterogeneityModelTrait},
};
use crate::recorder::Recorder;
use crate::source::Source;
use crate::time::Time;
use crate::utils::{fft_3d, ifft_3d, array_utils}; // Removed warm_fft_cache, report_fft_statistics
use log::{info, trace, debug}; // Removed debug, warn (used as log::debug, log::warn)
use ndarray::{Array3, Array4, Axis};
// Removed num_complex::Complex
use std::time::{Duration, Instant};
use std::sync::Arc;
use log::warn; // Added warn import

// Import AMR types
use self::amr::{AMRManager, AMRConfig};
// Removed std::sync::atomic::{AtomicBool, Ordering}
// Removed rayon::prelude::*;

// Field indices for Array4 in SimulationFields
// Acoustic + Optical + Thermal fields
pub const PRESSURE_IDX: usize = 0;      // Acoustic pressure
pub const LIGHT_IDX: usize = 1;         // Light intensity (e.g., for sonoluminescence)
pub const TEMPERATURE_IDX: usize = 2;   // Temperature
pub const BUBBLE_RADIUS_IDX: usize = 3; // Bubble radius (for cavitation)
// Elastic wave fields (particle velocities)
pub const VX_IDX: usize = 4;            // Particle velocity in x
pub const VY_IDX: usize = 5;            // Particle velocity in y
pub const VZ_IDX: usize = 6;            // Particle velocity in z
// Elastic wave fields (stress components)
pub const SXX_IDX: usize = 7;           // Normal stress in x
pub const SYY_IDX: usize = 8;           // Normal stress in y
pub const SZZ_IDX: usize = 9;           // Normal stress in z
pub const SXY_IDX: usize = 10;          // Shear stress in xy plane
pub const SXZ_IDX: usize = 11;          // Shear stress in xz plane
pub const SYZ_IDX: usize = 12;          // Shear stress in yz plane

// Total number of fields by default (acoustic + optical + thermal + elastic)
// This might need to be dynamic if not all models are active.
pub const TOTAL_FIELDS: usize = 13;


#[derive(Debug)]
pub struct SimulationFields {
    pub fields: Array4<f64>,
}

impl SimulationFields {
    /// Creates new simulation fields.
    /// The `num_fields` argument is now used to determine the first dimension of the array.
    /// For a simulation including elastic waves, this should be at least `TOTAL_FIELDS`.
    pub fn new(num_fields: usize, nx: usize, ny: usize, nz: usize) -> Self {
        info!("Initializing SimulationFields with {} fields, dimensions: ({}, {}, {})", num_fields, nx, ny, nz);
        if num_fields == 0 && (nx > 0 || ny > 0 || nz > 0) {
            warn!("SimulationFields created with num_fields = 0 but spatial dimensions > 0. This might lead to errors if fields are accessed.");
        }
         if num_fields < TOTAL_FIELDS && num_fields > BUBBLE_RADIUS_IDX + 1 { // Check if it's trying to be elastic but too small
            warn!(
                "SimulationFields initialized with {} fields, which is less than the {} required for full elastic + acoustic/thermal simulation. Ensure this is intended.",
                num_fields, TOTAL_FIELDS
            );
        }
        Self {
            fields: Array4::zeros((num_fields, nx, ny, nz)),
        }
    }
}

pub struct Solver {
    pub grid: Grid,
    pub time: Time,
    pub medium: Arc<dyn Medium>,
    pub fields: SimulationFields,
    pub boundary: Box<dyn Boundary>,
    pub source: Box<dyn Source>,
    pub prev_pressure: Array3<f64>,
    pub wave: Box<dyn AcousticWaveModel>,
    pub cavitation: Box<dyn CavitationModelBehavior>,
    pub light: Box<dyn LightDiffusionModelTrait>,
    pub thermal: Box<dyn ThermalModelTrait>,
    pub chemical: Box<dyn ChemicalModelTrait>,
    pub streaming: Box<dyn StreamingModelTrait>,
    pub scattering: Box<dyn AcousticScatteringModelTrait>,
    pub heterogeneity: Box<dyn HeterogeneityModelTrait>, // Refactored
    pub step_times: Vec<f64>,
    pub physics_times: [Vec<f64>; 9], // Timing for different physics components (including AMR)
    // Track pending medium updates
    pending_temperature_update: Option<Array3<f64>>,
    pending_bubble_update: Option<(Array3<f64>, Array3<f64>)>, // (radius, velocity)
    medium_update_attempts: usize,
    medium_update_successes: usize,
    // Adaptive Mesh Refinement
    pub amr_manager: Option<AMRManager>,
    amr_adapt_interval: usize,
    amr_last_adapt_step: usize,
}

impl Solver {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid: Grid,
        time: Time,
        medium: Arc<dyn Medium>,
        source: Box<dyn Source>,
        boundary: Box<dyn Boundary>,
        wave: Box<dyn AcousticWaveModel>,
        cavitation: Box<dyn CavitationModelBehavior>,
        light: Box<dyn LightDiffusionModelTrait>,
        thermal: Box<dyn ThermalModelTrait>,
        chemical: Box<dyn ChemicalModelTrait>,
        streaming: Box<dyn StreamingModelTrait>,
        scattering: Box<dyn AcousticScatteringModelTrait>,
        heterogeneity: Box<dyn HeterogeneityModelTrait>,
        num_simulation_fields: usize, // Added to allow dynamic field allocation
    ) -> Self {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        // Use the provided num_simulation_fields for field initialization
        let fields = SimulationFields::new(num_simulation_fields, nx, ny, nz);
        let prev_pressure = array_utils::zeros_from_grid(&grid);
        
        // Create a clone of time for use in capacity calculation
        let time_clone = time.clone();
        
        // Initialize physics timing arrays with proper type annotation
        let capacity = time_clone.n_steps / 10 + 1; // Store every 10th step
        let physics_times: [Vec<f64>; 9] = [
            Vec::with_capacity(capacity), // Acoustic wave
            Vec::with_capacity(capacity), // Boundary conditions
            Vec::with_capacity(capacity), // Cavitation
            Vec::with_capacity(capacity), // Light diffusion
            Vec::with_capacity(capacity), // Thermal
            Vec::with_capacity(capacity), // Chemical
            Vec::with_capacity(capacity), // Streaming
            Vec::with_capacity(capacity), // Scattering
            Vec::with_capacity(capacity), // AMR
        ];
        
        Self {
            grid,
            time,
            medium,
            fields,
            boundary,
            source,
            prev_pressure,
            wave,
            cavitation,
            light,
            thermal,
            chemical,
            streaming,
            scattering,
            heterogeneity,
            step_times: Vec::with_capacity(time_clone.n_steps),
            physics_times,
            pending_temperature_update: None,
            pending_bubble_update: None,
            medium_update_attempts: 0,
            medium_update_successes: 0,
            amr_manager: None,
            amr_adapt_interval: 10, // Default: adapt every 10 steps
            amr_last_adapt_step: 0,
        }
    }

    // Removed duplicated pub fn run and part of new() method body

    /// Enable Adaptive Mesh Refinement with given configuration
    pub fn enable_amr(&mut self, config: AMRConfig, adapt_interval: usize) -> KwaversResult<()> {
        info!("Enabling Adaptive Mesh Refinement");
        info!("  Max level: {}", config.max_level);
        info!("  Refine threshold: {:.2e}", config.refine_threshold);
        info!("  Coarsen threshold: {:.2e}", config.coarsen_threshold);
        info!("  Adaptation interval: {} steps", adapt_interval);
        
        self.amr_manager = Some(AMRManager::new(config, &self.grid));
        self.amr_adapt_interval = adapt_interval;
        self.amr_last_adapt_step = 0;
        
        Ok(())
    }

    /// Disable Adaptive Mesh Refinement
    pub fn disable_amr(&mut self) {
        info!("Disabling Adaptive Mesh Refinement");
        self.amr_manager = None;
    }
    
    /// Clear all sources from the solver
    pub fn clear_sources(&mut self) {
        // For now, we'll need to implement a way to clear sources
        // This is a placeholder that will be expanded when we refactor the source system
        warn!("clear_sources called but source clearing not yet implemented");
    }

    pub fn run(&mut self, recorder: &mut Recorder, frequency: f64) -> KwaversResult<()> {
        let dt = self.time.dt;
        let n_steps = self.time.n_steps;
        info!(
            "Starting simulation: dt = {:.6e}, steps = {}",
            dt, n_steps
        );

        // Pre-warm the FFT cache by performing a dummy FFT
        trace!("Pre-warming FFT cache for better performance");
        // Create and execute a dummy FFT to warm up the cache
        let dummy_fft = fft_3d(&self.fields.fields, PRESSURE_IDX, &self.grid);
        let _ = ifft_3d(&dummy_fft, &self.grid);

        // Progress tracking
        let mut last_progress_time = Instant::now();
        let progress_interval = Duration::from_secs(10); // Report progress every 10 seconds

        // Main simulation loop
        for step in 0..n_steps {
            // Use the enhanced step method that includes stability checks
            self.step(step, dt, frequency)?;
            
            // Current simulation time
            let t = step as f64 * dt;
            
            // Record data at this step
            recorder.record(&self.fields.fields, step, t);
            
            // Periodic progress reporting
            let now = Instant::now();
            if step == 0 || step == n_steps - 1 || 
               step % 100 == 0 || 
               now.duration_since(last_progress_time) >= progress_interval {
                
                // Calculate average step time and ETA
                let avg_step_time = if !self.step_times.is_empty() {
                    self.step_times.iter().sum::<f64>() / self.step_times.len() as f64
                } else {
                    0.0
                };
                
                // Calculate estimated time remaining
                let est_remaining = avg_step_time * (n_steps - step - 1) as f64;
                let remaining_hours = (est_remaining / 3600.0).floor();
                let remaining_minutes = ((est_remaining - remaining_hours * 3600.0) / 60.0).floor();
                let remaining_seconds = est_remaining - remaining_hours * 3600.0 - remaining_minutes * 60.0;
                
                // Get center values for display
                let center = [self.grid.nx / 2, self.grid.ny / 2, self.grid.nz / 2];
                
                // Report progress
                info!(
                    "Step {}/{} ({:.1}%): Pressure = {:.3e} Pa, Light = {:.3e} W/m², Temp = {:.2} K, Time = {:.3}s, Avg = {:.3}s, ETA: {:.0}h {:.0}m {:.0}s",
                    step,
                    n_steps,
                    100.0 * step as f64 / n_steps as f64,
                    self.fields.fields[[PRESSURE_IDX, center[0], center[1], center[2]]],
                    self.fields.fields[[LIGHT_IDX, center[0], center[1], center[2]]],
                    self.fields.fields[[TEMPERATURE_IDX, center[0], center[1], center[2]]],
                    self.step_times.last().unwrap_or(&0.0),
                    avg_step_time,
                    remaining_hours,
                    remaining_minutes,
                    remaining_seconds
                );
                
                last_progress_time = now;
            }
        }
        
        // Report performance statistics
        self.report_performance_statistics(n_steps);
        
        // Report medium update statistics
        info!(
            "Medium state update statistics: {}/{} successful ({:.1}%)",
            self.medium_update_successes,
            self.medium_update_attempts,
            100.0 * self.medium_update_successes as f64 / self.medium_update_attempts.max(1) as f64
        );
        Ok(())
    }
    
    fn report_performance_statistics(&self, n_steps: usize) {
        if self.step_times.is_empty() {
            return;
        }
        
        // Calculate total and component times
        let total_time: f64 = self.step_times.iter().sum();
        let avg_time = total_time / self.step_times.len() as f64;
        let min_time = self.step_times.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(&0.0);
        let max_time = self.step_times.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(&0.0);
        
        // Calculate average component times
        let component_times = [
            self.physics_times[0].iter().sum::<f64>() / n_steps as f64, // Preprocessing
            self.physics_times[1].iter().sum::<f64>() / n_steps as f64, // Acoustic Wave
            self.physics_times[2].iter().sum::<f64>() / n_steps as f64, // Cavitation
            self.physics_times[3].iter().sum::<f64>() / n_steps as f64, // Light diffusion
            self.physics_times[4].iter().sum::<f64>() / n_steps as f64, // Thermal
            self.physics_times[5].iter().sum::<f64>() / n_steps as f64, // Chemical
            self.physics_times[6].iter().sum::<f64>() / n_steps as f64, // Streaming
            self.physics_times[7].iter().sum::<f64>() / n_steps as f64, // Postprocessing
        ];
        
        // Calculate component percentages
        let component_percent: Vec<f64> = component_times.iter()
            .map(|&t| 100.0 * t * n_steps as f64 / total_time)
            .collect();
        
        // Report overall statistics
        info!(
            "Simulation completed in {:.2}s. Steps: {}, Avg: {:.3}s/step, Min: {:.3}s, Max: {:.3}s",
            total_time, n_steps, avg_time, min_time, max_time
        );
        
        // Report component times
        info!("Physics component breakdown:");
        info!("  Preprocessing: {:.3}s/step ({:.1}%)", component_times[0], component_percent[0]);
        info!("  Acoustic Wave: {:.3}s/step ({:.1}%)", component_times[1], component_percent[1]);
        info!("  Cavitation: {:.3}s/step ({:.1}%)", component_times[2], component_percent[2]);
        info!("  Light Diffusion: {:.3}s/step ({:.1}%)", component_times[3], component_percent[3]);
        info!("  Thermal: {:.3}s/step ({:.1}%)", component_times[4], component_percent[4]);
        info!("  Chemical: {:.3}s/step ({:.1}%)", component_times[5], component_percent[5]);
        info!("  Streaming/Scattering: {:.3}s/step ({:.1}%)", component_times[6], component_percent[6]);
        info!("  Postprocessing: {:.3}s/step ({:.1}%)", component_times[7], component_percent[7]);
        
        // Report nonlinear wave performance
        self.wave.report_performance();
    }

    /// Try to update the medium with pending changes, requires exclusive ownership
    fn try_update_medium(&mut self) -> bool {
        self.medium_update_attempts += 1;
        
        if let Some(exclusive_medium) = Arc::get_mut(&mut self.medium) {
            // We have exclusive access, apply all pending updates
            if let Some(temp) = &self.pending_temperature_update {
                exclusive_medium.update_temperature(temp);
            }
            
            if let Some((radius, velocity)) = &self.pending_bubble_update {
                exclusive_medium.update_bubble_state(radius, velocity);
            }
            
            // Clear pending updates
            self.pending_temperature_update = None;
            self.pending_bubble_update = None;
            
            self.medium_update_successes += 1;
            true
        } else {
            // Cannot get exclusive access
            log::warn!("Cannot update medium state (temperature, bubble dynamics) because it is shared");
            false
        }
    }
    
    /// Store updates to be applied later when exclusive access is available
    fn queue_medium_updates(&mut self, temperature: Array3<f64>, bubble_radius: Array3<f64>, bubble_velocity: Array3<f64>) {
        self.pending_temperature_update = Some(temperature);
        self.pending_bubble_update = Some((bubble_radius, bubble_velocity));
    }

    // Helper function to check and stabilize field values
    fn check_field(field: &mut Array3<f64>, field_name: &str, min_val: f64, max_val: f64) -> bool {
        let mut unstable_found = false;
        let total_elements = field.len();
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut min_violations = 0;
        let mut max_violations = 0;
        let mut min_value = f64::MAX;
        let mut max_value = f64::MIN;
        
        for val in field.iter_mut() {
            // Track statistics
            if !val.is_nan() && !val.is_infinite() {
                min_value = min_value.min(*val);
                max_value = max_value.max(*val);
            }
            
            // Handle NaN values
            if val.is_nan() {
                nan_count += 1;
                *val = 0.0;
                unstable_found = true;
                continue;
            }
            
            // Handle infinite values
            if val.is_infinite() {
                inf_count += 1;
                *val = if *val > 0.0 { max_val } else { min_val };
                unstable_found = true;
                continue;
            }
            
            // Clamp extreme values
            if *val < min_val {
                min_violations += 1;
                *val = min_val;
                unstable_found = true;
            } else if *val > max_val {
                max_violations += 1;
                *val = max_val;
                unstable_found = true;
            }
        }
        
        // Log issues if any found
        if unstable_found {
            log::warn!(
                "Field '{}' stabilized: {} NaN, {} Inf, {} < min, {} > max (range: {:.3e} to {:.3e})",
                field_name, 
                nan_count, 
                inf_count, 
                min_violations, 
                max_violations,
                min_value,
                max_value
            );
        } else if total_elements > 0 {
            // Log normal statistics periodically
            log::debug!(
                "Field '{}' stable: range {:.3e} to {:.3e}",
                field_name,
                min_value,
                max_value
            );
        }
        
        unstable_found
    }

    pub fn step(&mut self, step: usize, dt: f64, frequency: f64) -> KwaversResult<()> {
        // Performance tracking
        let step_start = Instant::now();
        let mut preprocessing_time = 0.0;
        let mut wave_time = 0.0;
        // let mut boundary_time = 0.0; // Removed unused variable
        let mut cavitation_time = 0.0;
        let mut light_time = 0.0;
        let mut thermal_time = 0.0;
        let mut chemical_time = 0.0;
        let mut streaming_time = 0.0;
        let mut _scattering_time = 0.0;     // Prefix with underscore to indicate intentionally unused
        let mut _heterogeneity_time = 0.0;  // Prefix with underscore to indicate intentionally unused
        let mut postprocessing_time = 0.0;
        let mut amr_time = 0.0;

        let current_time = step as f64 * dt;
        let t = current_time;
        
        // Adaptive Mesh Refinement
        if let Some(ref mut amr_manager) = self.amr_manager {
            if step > 0 && (step - self.amr_last_adapt_step) >= self.amr_adapt_interval {
                let amr_start = Instant::now();
                
                // Get pressure field for error estimation
                let pressure = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
                
                // Adapt mesh based on pressure field
                match amr_manager.adapt_mesh(&pressure, current_time) {
                    Ok(result) => {
                        if result.cells_refined > 0 || result.cells_coarsened > 0 {
                            info!("AMR adaptation at step {}: refined {} cells, coarsened {} cells, max error: {:.2e}",
                                  step, result.cells_refined, result.cells_coarsened, result.max_error);
                            
                            // Adapt all fields to new mesh structure using local operations
                            let octree = amr_manager.octree();
                            use self::amr::local_operations::adapt_all_fields;
                            
                            match adapt_all_fields(&self.fields.fields, octree, amr_manager.interpolation_scheme()) {
                                    Ok(new_fields) => {
                                        // Check if dimensions changed
                                        let old_shape = self.fields.fields.shape();
                                        let new_shape = new_fields.shape();
                                        
                                        if old_shape != new_shape {
                                            info!("AMR: Grid dimensions changed from {:?} to {:?}", 
                                                  &old_shape[1..], &new_shape[1..]);
                                            
                                            // Update grid dimensions if needed
                                            if new_shape[1] != self.grid.nx || 
                                               new_shape[2] != self.grid.ny || 
                                               new_shape[3] != self.grid.nz {
                                                // Note: This would require updating the grid structure
                                                // For now, we log a warning
                                                warn!("Grid dimensions need to be updated to match AMR fields");
                                            }
                                        }
                                        
                                        // Replace fields with adapted version
                                        self.fields.fields = new_fields;
                                        
                                        // Update previous pressure array to match new dimensions
                                        if self.prev_pressure.shape() != self.fields.fields.index_axis(Axis(0), PRESSURE_IDX).shape() {
                                            self.prev_pressure = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
                                        }
                                        
                                        debug!("Completed AMR field adaptation");
                                    }
                                    Err(e) => {
                                        warn!("Failed to adapt fields for AMR: {}", e);
                                    }
                                }
                        }
                        
                        // Update last adapt step
                        self.amr_last_adapt_step = step;
                        
                        // Report memory savings
                        let stats = amr_manager.memory_stats();
                        if step % 100 == 0 {
                            info!("AMR memory stats: {:.1}% saved, {:.2}x compression",
                                  stats.memory_saved_percent, stats.compression_ratio);
                        }
                    }
                    Err(e) => {
                        warn!("AMR adaptation failed at step {}: {}", step, e);
                    }
                }
                
                amr_time = amr_start.elapsed().as_secs_f64();
            }
        }
        
        // Safe max and min values for physical properties to prevent instability
        const MAX_PRESSURE: f64 = 1e9;   // 1 GPa max pressure
        const MIN_PRESSURE: f64 = -1e9;  // -1 GPa min pressure
        const MAX_TEMP: f64 = 1000.0;    // 1000K max temperature
        const MIN_TEMP: f64 = 273.0;     // 0°C min temperature
        const MAX_LIGHT: f64 = 1e10;     // Max light intensity
        
        // Preprocess step - create owned copies of fields we'll need
        let preprocess_start = Instant::now();
        
        // Get pressure field as owned copy
        let mut pressure = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        
        // Create owned copy for boundary condition application
        let mut pressure_owned = pressure.to_owned();
        
        // Apply acoustic boundary conditions
        self.boundary.apply_acoustic(&mut pressure_owned, &self.grid, step)?;
        
        // Update original pressure with boundary-applied version
        pressure.assign(&pressure_owned);
        self.fields.fields.index_axis_mut(Axis(0), PRESSURE_IDX).assign(&pressure);

        // Implement proper elastic wave PML boundary conditions
        // Follows Single Responsibility: Each field type gets appropriate boundary treatment
        let elastic_velocity_indices = [VX_IDX, VY_IDX, VZ_IDX];
        for &field_idx in elastic_velocity_indices.iter() {
            if field_idx < self.fields.fields.shape()[0] { // Check if the field exists
                let mut component = self.fields.fields.index_axis(Axis(0), field_idx).to_owned();
                
                // Apply elastic-specific boundary conditions
                // For velocity components, use modified PML parameters
                self.apply_elastic_pml(&mut component, field_idx, step)?;
                
                self.fields.fields.index_axis_mut(Axis(0), field_idx).assign(&component);
                trace!("Applied elastic PML to velocity component {}", field_idx);
            }
        }
        
        // Handle stress tensor components if they exist
        let stress_indices = [SXX_IDX, SYY_IDX, SZZ_IDX, SXY_IDX, SXZ_IDX, SYZ_IDX];
        for &field_idx in stress_indices.iter() {
            if field_idx < self.fields.fields.shape()[0] {
                let mut component = self.fields.fields.index_axis(Axis(0), field_idx).to_owned();
                
                // Apply stress-specific boundary conditions
                self.apply_stress_pml(&mut component, field_idx, step)?;
                
                self.fields.fields.index_axis_mut(Axis(0), field_idx).assign(&component);
                trace!("Applied stress PML to component {}", field_idx);
            }
        }
        
        preprocessing_time += preprocess_start.elapsed().as_secs_f64();

        // 1. Wave propagation
        let wave_start = Instant::now();
        self.wave.update_wave(
            &mut self.fields.fields,
            &self.prev_pressure,
            self.source.as_ref(),
            &self.grid,
            self.medium.as_ref(),
            dt,
            t,
        );
        wave_time += wave_start.elapsed().as_secs_f64();

        // Check and stabilize after wave update
        {
            let mut pressure = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
            Self::check_field(&mut pressure, "pressure", MIN_PRESSURE, MAX_PRESSURE);
            self.fields.fields.index_axis_mut(Axis(0), PRESSURE_IDX).assign(&pressure);
        }

        // Get updated pressure field 
        pressure = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        let p_update = pressure.clone();
        
        // 2. Update cavitation effects (if bubble radius is significant)
        let cavitation_start = Instant::now();
        // Update cavitation
        let current_time = step as f64 * dt;
        if let Err(e) = self.cavitation.update_cavitation(
            &pressure,
            &self.grid,
            self.medium.as_ref(),
            dt,
            current_time,
        ) {
            log::error!("Cavitation update failed: {}", e);
        }
        cavitation_time += cavitation_start.elapsed().as_secs_f64();
        
        // Check and stabilize after cavitation update
        let mut p_update = p_update.clone();
        Self::check_field(&mut p_update, "pressure_after_cavitation", MIN_PRESSURE, MAX_PRESSURE);
        
        // Update pressure field with cavitation effects
        self.fields.fields.index_axis_mut(Axis(0), PRESSURE_IDX).assign(&p_update);
        
        // Get light emission from cavitation state
        let light_emission = self.cavitation.light_emission();
        
        // 3. Light propagation and fluence
        let light_start = Instant::now();
        self.light.update_light(
            &mut self.fields.fields,
            &light_emission,
            &self.grid,
            self.medium.as_ref(),
            dt,
        );
        light_time += light_start.elapsed().as_secs_f64();
        
        // Check and stabilize light field after update
        {
            let mut light = self.fields.fields.index_axis(Axis(0), LIGHT_IDX).to_owned();
            Self::check_field(&mut light, "light_after_update", 0.0, MAX_LIGHT);
            self.fields.fields.index_axis_mut(Axis(0), LIGHT_IDX).assign(&light);
        }
        
        // 4. Update heterogeneity effects
        let heterogeneity_start = Instant::now();
        // Heterogeneity update code would go here
        _heterogeneity_time += heterogeneity_start.elapsed().as_secs_f64();
        
        // 5. Scattering effects
        let scattering_start = Instant::now();
        self.scattering.compute_scattering(
            &pressure,                       // incident_field
            self.medium.bubble_radius(),    // bubble_radius
            self.medium.bubble_velocity(),  // bubble_velocity (new argument)
            &self.grid,
            self.medium.as_ref(),
            frequency,
        );
        _scattering_time += scattering_start.elapsed().as_secs_f64();
        
        // 6. Thermal model update
        let thermal_start = Instant::now();
        self.thermal.update_thermal(
            &mut self.fields.fields,
            &self.grid,
            self.medium.as_ref(),
            dt,
            frequency,
        );
        thermal_time += thermal_start.elapsed().as_secs_f64();
        
        // Check and stabilize temperature field
        {
            let mut temp = self.fields.fields.index_axis(Axis(0), TEMPERATURE_IDX).to_owned();
            Self::check_field(&mut temp, "temperature", MIN_TEMP, MAX_TEMP);
            self.fields.fields.index_axis_mut(Axis(0), TEMPERATURE_IDX).assign(&temp);
        }
        
        // 7. Chemical reactions
        let chemical_start = Instant::now();
        // Create owned copies of needed fields
        let pressure = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        let light = self.fields.fields.index_axis(Axis(0), LIGHT_IDX).to_owned();
        let emission_spectrum = self.light.emission_spectrum().to_owned();
        let bubble_radius = self.medium.bubble_radius().to_owned();
        let temperature = self.fields.fields.index_axis(Axis(0), TEMPERATURE_IDX).to_owned();
        
        self.chemical.update_chemical(
            &pressure,
            &light,
            &emission_spectrum,
            &bubble_radius,
            &temperature,
            &self.grid,
            dt,
            self.medium.as_ref(),
            frequency,
        );
        chemical_time += chemical_start.elapsed().as_secs_f64();
        
        // 8. Acoustic streaming
        let streaming_start = Instant::now();
        self.streaming.update_velocity(
            &pressure,
            &self.grid,
            self.medium.as_ref(),
            dt,
        );
        streaming_time += streaming_start.elapsed().as_secs_f64();
        
        // Post-processing and state updates
        let postprocess_start = Instant::now();
        
        // Save current pressure for next time step
        self.prev_pressure.assign(&pressure);
        
        // After all physics have been processed, we can now update the medium state
        // Store the current state to apply later
        let temperature = self.fields.fields.index_axis(Axis(0), TEMPERATURE_IDX).to_owned();
                    let bubble_radius = self.cavitation.bubble_radius().unwrap_or_else(|_| Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz)));
                    let bubble_velocity = self.cavitation.bubble_velocity().unwrap_or_else(|_| Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz)));
        
        // Queue the updates
        self.queue_medium_updates(temperature, bubble_radius, bubble_velocity);
        
        // Try to apply the updates (will succeed if we have exclusive ownership)
        self.try_update_medium();
        
        postprocessing_time += postprocess_start.elapsed().as_secs_f64();
        
        // Record timing information
        let step_time = step_start.elapsed().as_secs_f64();
        self.step_times.push(step_time);
        self.physics_times[0].push(preprocessing_time);
        self.physics_times[1].push(wave_time);
        self.physics_times[2].push(cavitation_time);
        self.physics_times[3].push(light_time);
        self.physics_times[4].push(thermal_time);
        self.physics_times[5].push(chemical_time);
        self.physics_times[6].push(streaming_time);
        self.physics_times[7].push(postprocessing_time);
        self.physics_times[8].push(amr_time);
        
        if step % 100 == 0 {
            let timing_msg = if amr_time > 0.0 {
                format!(
                    "Step {} completed in {:.4} s (wave: {:.4}, cav: {:.4}, light: {:.4}, thermal: {:.4}, AMR: {:.4})",
                    step, step_time, wave_time, cavitation_time, light_time, thermal_time, amr_time
                )
            } else {
                format!(
                    "Step {} completed in {:.4} s (wave: {:.4}, cav: {:.4}, light: {:.4}, thermal: {:.4})",
                    step, step_time, wave_time, cavitation_time, light_time, thermal_time
                )
            };
            info!("{}", timing_msg);
        }
        Ok(())
    }

    /// Apply elastic-specific PML boundary conditions for velocity components
    /// Follows Single Responsibility: Handles only elastic velocity boundary conditions
    fn apply_elastic_pml(&mut self, field: &mut Array3<f64>, field_idx: usize, step: usize) -> KwaversResult<()> {
        // Elastic waves require different damping coefficients than acoustic waves
        let velocity_damping_factor = 0.8; // Reduced damping for velocity components
        
        // Apply modified PML with velocity-specific parameters
        self.boundary.apply_acoustic_with_factor(field, &self.grid, step, velocity_damping_factor)?;
        
        // Additional velocity-specific boundary treatment
        self.apply_velocity_boundary_conditions(field, field_idx)?;
        
        Ok(())
    }
    
    /// Apply stress-specific PML boundary conditions for stress tensor components  
    /// Follows Single Responsibility: Handles only stress tensor boundary conditions
    fn apply_stress_pml(&mut self, field: &mut Array3<f64>, field_idx: usize, step: usize) -> KwaversResult<()> {
        // Stress components require different treatment than velocity
        let stress_damping_factor = 1.2; // Enhanced damping for stress components
        
        // Apply modified PML with stress-specific parameters
        self.boundary.apply_acoustic_with_factor(field, &self.grid, step, stress_damping_factor)?;
        
        // Additional stress-specific boundary treatment
        self.apply_stress_boundary_conditions(field, field_idx)?;
        
        Ok(())
    }
    
    /// Apply velocity-specific boundary conditions
    /// Follows Single Responsibility: Handles velocity field boundaries
    fn apply_velocity_boundary_conditions(&self, field: &mut Array3<f64>, field_idx: usize) -> KwaversResult<()> {
        // Apply no-slip boundary conditions at edges for velocity components
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        
        // Set velocity to zero at boundaries (no-slip condition)
        for i in 0..nx {
            for j in 0..ny {
                field[[i, j, 0]] = 0.0;      // z = 0 boundary
                field[[i, j, nz-1]] = 0.0;   // z = max boundary
            }
        }
        
        for i in 0..nx {
            for k in 0..nz {
                field[[i, 0, k]] = 0.0;      // y = 0 boundary  
                field[[i, ny-1, k]] = 0.0;   // y = max boundary
            }
        }
        
        for j in 0..ny {
            for k in 0..nz {
                field[[0, j, k]] = 0.0;      // x = 0 boundary
                field[[nx-1, j, k]] = 0.0;   // x = max boundary
            }
        }
        
        Ok(())
    }
    
    /// Apply stress-specific boundary conditions
    /// Follows Single Responsibility: Handles stress tensor boundaries
    fn apply_stress_boundary_conditions(&self, field: &mut Array3<f64>, field_idx: usize) -> KwaversResult<()> {
        // Stress boundary conditions depend on the specific stress component
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        
        // Apply free surface boundary conditions (zero stress at boundaries)
        for i in 0..nx {
            for j in 0..ny {
                field[[i, j, 0]] = 0.0;      // z = 0 boundary (free surface)
                field[[i, j, nz-1]] = 0.0;   // z = max boundary
            }
        }
        
        // Side boundaries can have different conditions based on field type
        match field_idx {
            // Normal stress components (diagonal of stress tensor)
            SXX_IDX | SYY_IDX | SZZ_IDX => {
                // Apply symmetric boundary conditions for normal stresses
                for i in 0..nx {
                    for k in 0..nz {
                        field[[i, 0, k]] = field[[i, 1, k]];        // y = 0 boundary
                        field[[i, ny-1, k]] = field[[i, ny-2, k]];  // y = max boundary
                    }
                }
                for j in 0..ny {
                    for k in 0..nz {
                        field[[0, j, k]] = field[[1, j, k]];        // x = 0 boundary
                        field[[nx-1, j, k]] = field[[nx-2, j, k]];  // x = max boundary
                    }
                }
            }
            // Shear stress components (off-diagonal of stress tensor)
            SXY_IDX | SXZ_IDX | SYZ_IDX => {
                // Apply antisymmetric boundary conditions for shear stresses
                for i in 0..nx {
                    for k in 0..nz {
                        field[[i, 0, k]] = -field[[i, 1, k]];       // y = 0 boundary
                        field[[i, ny-1, k]] = -field[[i, ny-2, k]]; // y = max boundary
                    }
                }
                for j in 0..ny {
                    for k in 0..nz {
                        field[[0, j, k]] = -field[[1, j, k]];       // x = 0 boundary
                        field[[nx-1, j, k]] = -field[[nx-2, j, k]]; // x = max boundary
                    }
                }
            }
            _ => {
                // Default to zero boundary conditions
                for i in 0..nx {
                    for k in 0..nz {
                        field[[i, 0, k]] = 0.0;
                        field[[i, ny-1, k]] = 0.0;
                    }
                }
                for j in 0..ny {
                    for k in 0..nz {
                        field[[0, j, k]] = 0.0;
                        field[[nx-1, j, k]] = 0.0;
                    }
                }
            }
        }
        
        Ok(())
    }
}

/// Lazy evaluation utilities for solver operations
pub mod lazy {
    use super::*;
    use crate::grid::Grid;
    use crate::error::KwaversResult;
    use ndarray::{Array3, Array4};
    use std::cell::RefCell;
    
    
    /// Lazy field computation that defers calculation until needed
    pub struct LazyField<T> {
        computation: RefCell<Option<Box<dyn FnOnce() -> Array3<T>>>>,
        cached_value: RefCell<Option<Array3<T>>>,
    }
    
    impl<T: Clone> LazyField<T> {
        /// Create a new lazy field
        pub fn new<F>(computation: F) -> Self
        where
            F: FnOnce() -> Array3<T> + 'static,
        {
            Self {
                computation: RefCell::new(Some(Box::new(computation))),
                cached_value: RefCell::new(None),
            }
        }
        
        /// Get the computed value, calculating it if necessary
        pub fn get(&self) -> Array3<T> {
            if self.cached_value.borrow().is_none() {
                if let Some(computation) = self.computation.borrow_mut().take() {
                    *self.cached_value.borrow_mut() = Some(computation());
                }
            }
            
            self.cached_value.borrow()
                .as_ref()
                .expect("LazyField computation failed")
                .clone()
        }
        
        /// Check if the value has been computed
        pub fn is_computed(&self) -> bool {
            self.cached_value.borrow().is_some()
        }
    }
    
    /// Lazy solver state that computes fields on demand
    pub struct LazySolverState {
        _base_fields: Array4<f64>,
        lazy_derivatives: Vec<LazyField<f64>>,
        _grid: Grid,
    }
    
    impl LazySolverState {
        pub fn new(fields: Array4<f64>, grid: Grid) -> Self {
            Self {
                _base_fields: fields,
                lazy_derivatives: Vec::new(),
                _grid: grid,
            }
        }
        
        /// Add a lazy derivative computation
        pub fn add_derivative<F>(&mut self, computation: F)
        where
            F: FnOnce() -> Array3<f64> + 'static,
        {
            self.lazy_derivatives.push(LazyField::new(computation));
        }
        
        /// Get a specific derivative, computing it if needed
        pub fn get_derivative(&self, index: usize) -> Option<Array3<f64>> {
            self.lazy_derivatives.get(index)
                .map(|lazy| lazy.get())
        }
        
        /// Compute only the derivatives that are needed
        pub fn compute_required(&self, indices: &[usize]) -> Vec<Array3<f64>> {
            indices.iter()
                .filter_map(|&idx| self.get_derivative(idx))
                .collect()
        }
    }
    
    /// Lazy chain of solver operations
    pub struct LazySolverChain {
        operations: Vec<Box<dyn FnOnce(&mut Solver) -> KwaversResult<()>>>,
    }
    
    impl LazySolverChain {
        pub fn new() -> Self {
            Self {
                operations: Vec::new(),
            }
        }
        
        /// Add an operation to the chain
        pub fn then<F>(mut self, operation: F) -> Self
        where
            F: FnOnce(&mut Solver) -> KwaversResult<()> + 'static,
        {
            self.operations.push(Box::new(operation));
            self
        }
        
        /// Execute all operations
        pub fn execute(self, solver: &mut Solver) -> KwaversResult<()> {
            for operation in self.operations {
                operation(solver)?;
            }
            Ok(())
        }
        
        /// Execute operations until a condition is met
        pub fn execute_until<F>(
            self,
            solver: &mut Solver,
            condition: F,
        ) -> KwaversResult<usize>
        where
            F: Fn(&Solver) -> bool,
        {
            let mut executed = 0;
            
            for operation in self.operations {
                if condition(solver) {
                    break;
                }
                operation(solver)?;
                executed += 1;
            }
            
            Ok(executed)
        }
    }
    
    /// Lazy iterator over solver time steps
    pub struct LazyTimeStepIterator<'a> {
        solver: &'a mut Solver,
        current_step: usize,
        max_steps: usize,
        dt: f64,
    }
    
    impl<'a> LazyTimeStepIterator<'a> {
        pub fn new(
            solver: &'a mut Solver,
            max_steps: usize,
            dt: f64,
        ) -> Self {
            Self {
                solver,
                current_step: 0,
                max_steps,
                dt,
            }
        }
    }
    
    impl<'a> Iterator for LazyTimeStepIterator<'a> {
        type Item = KwaversResult<SolverSnapshot>;
        
        fn next(&mut self) -> Option<Self::Item> {
            if self.current_step >= self.max_steps {
                return None;
            }
            
            // Note: step is private, so we'll return a snapshot instead
            let snapshot = SolverSnapshot {
                step: self.current_step,
                time: self.current_step as f64 * self.dt,
                fields: self.solver.fields.fields.clone(),
                max_pressure: self.solver.fields.fields
                    .index_axis(ndarray::Axis(0), PRESSURE_IDX)
                    .iter()
                    .map(|&p| p.abs())
                    .fold(0.0, f64::max),
            };
            
            self.current_step += 1;
            Some(Ok(snapshot))
        }
    }
    
    #[derive(Clone)]
    pub struct SolverSnapshot {
        pub step: usize,
        pub time: f64,
        pub fields: Array4<f64>,
        pub max_pressure: f64,
    }
    
    /// Lazy field transformation pipeline
    pub struct LazyFieldPipeline<T> {
        source: LazyField<T>,
        transforms: Vec<Box<dyn Fn(&Array3<T>) -> Array3<T>>>,
    }
    
    impl<T: Clone + 'static> LazyFieldPipeline<T> {
        pub fn new<F>(source: F) -> Self
        where
            F: FnOnce() -> Array3<T> + 'static,
        {
            Self {
                source: LazyField::new(source),
                transforms: Vec::new(),
            }
        }
        
        /// Add a transformation to the pipeline
        pub fn transform<F>(mut self, f: F) -> Self
        where
            F: Fn(&Array3<T>) -> Array3<T> + 'static,
        {
            self.transforms.push(Box::new(f));
            self
        }
        
        /// Execute the pipeline and get the result
        pub fn execute(self) -> Array3<T> {
            let mut result = self.source.get();
            
            for transform in self.transforms {
                result = transform(&result);
            }
            
            result
        }
        
        /// Create a new lazy field from this pipeline
        pub fn to_lazy(self) -> LazyField<T> {
            LazyField::new(move || self.execute())
        }
    }
}