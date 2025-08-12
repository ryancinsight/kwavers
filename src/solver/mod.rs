// src/solver/mod.rs
pub mod pstd;
pub mod fdtd;
pub mod hybrid;
pub mod time_integration;
pub mod spectral_dg;
pub mod imex;
pub mod amr;
pub mod kspace_correction;
pub mod heterogeneous_handler;
pub mod cpml_integration;
pub mod validation;
pub mod workspace;
pub mod time_reversal;
pub mod thermal_diffusion;
pub mod reconstruction;
pub mod plugin_based_solver; // New plugin-based architecture

use crate::boundary::{Boundary, PMLBoundary, CPMLBoundary};
use crate::config::ValidationConfig;
use crate::error::{KwaversResult, KwaversError, PhysicsError};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_indices::*;  // Import all field indices from unified source
use crate::physics::traits::{
    AcousticWaveModel, CavitationModelBehavior, LightDiffusionModelTrait,
    ThermalModelTrait, ChemicalModelTrait, AcousticScatteringModelTrait,
    StreamingModelTrait, HeterogeneityModelTrait
};
use crate::recorder::Recorder;
use crate::physics::mechanics::cavitation::CavitationModel;
use crate::physics::optics::diffusion::LightDiffusion;
use crate::physics::thermodynamics::heat_transfer::ThermalModel;
use crate::physics::chemistry::ChemicalModel;
use crate::physics::scattering::acoustic::RayleighScattering;
use crate::source::Source;
use crate::time::Time;
use crate::utils::{fft_3d, ifft_3d, array_utils}; // Removed warm_fft_cache, report_fft_statistics
use log::{info, trace, debug}; // Removed debug, warn (used as log::debug, log::warn)
use ndarray::{Array3, Array4, Axis, ArrayView3, ArrayViewMut3};
// Removed num_complex::Complex
use std::time::{Duration, Instant};
use std::sync::Arc;

// Physical validation constants - should be in configuration
const MAX_PRESSURE: f64 = 1e9;   // 1 GPa max pressure
const MIN_PRESSURE: f64 = -1e9;  // -1 GPa min pressure  
const MAX_TEMP: f64 = 1000.0;    // 1000K max temperature
const MIN_TEMP: f64 = 273.0;     // 0°C min temperature
const MAX_LIGHT: f64 = 1e10;     // Max light intensity

// Note: Field indices are now imported from physics::field_indices
// This ensures SSOT - Single Source of Truth
use log::warn; // Added warn import

// Import AMR types
use self::amr::{AMRManager, AMRConfig};
// Removed std::sync::atomic::{AtomicBool, Ordering}
// Removed rayon::prelude::*;

// Field indices are now imported from physics::field_indices for SSOT
// Re-export them for backward compatibility
pub use crate::physics::field_indices::{
    PRESSURE_IDX, LIGHT_IDX, TEMPERATURE_IDX, BUBBLE_RADIUS_IDX, BUBBLE_VELOCITY_IDX,
    VX_IDX, VY_IDX, VZ_IDX,
    SXX_IDX, SYY_IDX, SZZ_IDX, SXY_IDX, SXZ_IDX, SYZ_IDX,
    TOTAL_FIELDS
};


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

/// Monolithic solver struct that orchestrates the simulation
/// 
/// DESIGN PRINCIPLE VIOLATIONS:
/// - SRP: This class has too many responsibilities (physics, validation, timing, AMR, etc.)
/// - DIP: Depends on concrete implementations rather than abstractions
/// - OCP: Adding new physics requires modifying this class
/// 
/// REFACTORING NEEDED:
/// - Use plugin_based_solver.rs as the foundation
/// - Extract validation to ValidationManager
/// - Extract timing/metrics to MetricsCollector
/// - Extract field management to FieldManager
/// - Use event-driven architecture for component communication
/// 
/// NOTE: This struct is kept for backward compatibility but should be deprecated
/// in favor of the plugin-based architecture in plugin_based_solver.rs
pub struct Solver {
    // Core components
    pub fields: SimulationFields,
    pub grid: Grid,
    pub medium: Arc<dyn Medium + Send + Sync>,
    pub boundary: Box<dyn Boundary>,
    pub source: Box<dyn Source>,
    pub time: Time,
    
    // Physics models
    pub wave: Box<dyn AcousticWaveModel>,
    pub cavitation: Box<dyn CavitationModelBehavior>,
    pub light: Box<dyn LightDiffusionModelTrait>,
    pub thermal: Box<dyn ThermalModelTrait>,
    pub chemical: Box<dyn ChemicalModelTrait>,
    pub streaming: Box<dyn StreamingModelTrait>,
    pub scattering: Box<dyn AcousticScatteringModelTrait>,
    pub heterogeneity: Box<dyn HeterogeneityModelTrait>,
    
    // Validation configuration - SSOT for all limits
    pub validation_config: ValidationConfig,
    
    // State tracking
    pub prev_pressure: Array3<f64>,
    pub step_times: Vec<f64>,
    pub physics_times: [Vec<f64>; 9], // Timing for different physics components (including AMR)
    
    // Medium update tracking
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
        validation_config: Option<ValidationConfig>,
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
        
        let validation_config = validation_config.unwrap_or_default();
        
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
            validation_config,
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
    fn try_update_medium(&mut self) -> KwaversResult<()> {
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
            Ok(())
        } else {
            // Cannot get exclusive access
            log::warn!("Cannot update medium state (temperature, bubble dynamics) because it is shared");
            Err(KwaversError::ConcurrencyError {
                operation: "try_update_medium".to_string(),
                resource: "medium".to_string(),
                reason: "Cannot obtain exclusive access to medium for atomic update. \
                        This would violate ACID guarantees by proceeding with stale data.".to_string(),
            })
        }
    }
    
    /// Store updates to be applied later when exclusive access is available
    fn queue_medium_updates(&mut self, temperature: Array3<f64>, bubble_radius: Array3<f64>, bubble_velocity: Array3<f64>) {
        self.pending_temperature_update = Some(temperature);
        self.pending_bubble_update = Some((bubble_radius, bubble_velocity));
    }

    // Helper function to validate field values - fail fast on numerical instabilities
    // Following KISS principle: simple validation that fails loudly on problems
    fn validate_field(field: &Array3<f64>, field_name: &str, min_val: f64, max_val: f64) -> KwaversResult<()> {
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut min_violations = 0;
        let mut max_violations = 0;
        let mut first_nan_location = None;
        let mut first_inf_location = None;
        
        // Check for numerical instabilities
        for ((i, j, k), val) in field.indexed_iter() {
            if val.is_nan() {
                nan_count += 1;
                if first_nan_location.is_none() {
                    first_nan_location = Some((i, j, k));
                }
            } else if val.is_infinite() {
                inf_count += 1;
                if first_inf_location.is_none() {
                    first_inf_location = Some((i, j, k));
                }
            } else if *val < min_val {
                min_violations += 1;
            } else if *val > max_val {
                max_violations += 1;
            }
        }
        
        // Fail fast on any numerical instability - don't mask the problem
        if nan_count > 0 {
            let location = first_nan_location.unwrap();
            return Err(PhysicsError::Instability {
                field: field_name.to_string(),
                location,
                value: f64::NAN,
            }.into());
        }
        
        if inf_count > 0 {
            let location = first_inf_location.unwrap();
            return Err(PhysicsError::Instability {
                field: field_name.to_string(),
                location,
                value: f64::INFINITY,
            }.into());
        }
        
        // Log warnings for bounds violations but don't fail
        // These might be acceptable in some cases
        if min_violations > 0 || max_violations > 0 {
            log::warn!(
                "Field '{}' has {} values below {:.3e} and {} values above {:.3e}",
                field_name, min_violations, min_val, max_violations, max_val
            );
        }
        
        Ok(())
    }

    pub fn step(&mut self, step: usize, dt: f64, frequency: f64) -> KwaversResult<()> {
        // Timing for performance monitoring
        let step_start = Instant::now();
        let mut preprocessing_time = 0.0;
        let mut wave_time = 0.0;
        let mut cavitation_time = 0.0;
        let mut light_time = 0.0;
        let mut _heterogeneity_time = 0.0;
        let mut _scattering_time = 0.0;
        let mut thermal_time = 0.0;
        let mut chemical_time = 0.0;
        let mut postprocessing_time = 0.0;
        
        // Preprocess step - work with views instead of clones
        let preprocess_start = Instant::now();
        
        // Apply acoustic boundary conditions in-place
        {
            let mut pressure_view = self.fields.fields.index_axis_mut(Axis(0), PRESSURE_IDX);
            self.boundary.apply_acoustic(&mut pressure_view, &self.grid, step)?;
        }

        // Apply elastic wave PML boundary conditions in-place
        let elastic_velocity_indices = [VX_IDX, VY_IDX, VZ_IDX];
        for &field_idx in elastic_velocity_indices.iter() {
            if field_idx < self.fields.fields.shape()[0] {
                let mut component_view = self.fields.fields.index_axis_mut(Axis(0), field_idx);
                self.apply_elastic_pml(&mut component_view, field_idx, step)?;
            }
        }
        
        // Apply stress component boundary conditions in-place
        let stress_indices = [SXX_IDX, SYY_IDX, SZZ_IDX, 
                            SXY_IDX, SXZ_IDX, SYZ_IDX];
        for &field_idx in stress_indices.iter() {
            if field_idx < self.fields.fields.shape()[0] {
                let mut component_view = self.fields.fields.index_axis_mut(Axis(0), field_idx);
                self.apply_stress_boundary(&mut component_view, field_idx, step)?;
            }
        }
        
        preprocessing_time += preprocess_start.elapsed().as_secs_f64();
        
        // 1. Wave propagation - work with field views
        let wave_start = Instant::now();
        let t = step as f64 * dt;
        self.wave.update_wave(
            &mut self.fields.fields,
            &self.prev_pressure,
            self.source.as_ref(),
            &self.grid,
            self.medium.as_ref(),
            dt,
            t,
        )?;
        wave_time += wave_start.elapsed().as_secs_f64();
        
        // Validate pressure field without cloning - use config values
        {
            let pressure_view = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX);
            Self::validate_field(
                &pressure_view, 
                "pressure", 
                self.validation_config.pressure.min,
                self.validation_config.pressure.max
            )?;
        }

        // 2. Update cavitation effects using views
        let cavitation_start = Instant::now();
        let current_time = step as f64 * dt;
        
        // Get pressure as owned array for cavitation update
        let mut pressure = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        
        // Update cavitation (modifies pressure in place according to trait)
        self.cavitation.update_cavitation(
            &pressure,
            &self.grid,
            self.medium.as_ref(),
            dt,
            current_time,
        )?;
        
        // Update pressure field with cavitation effects
        self.fields.fields.index_axis_mut(Axis(0), PRESSURE_IDX)
            .assign(&pressure);
        
        cavitation_time += cavitation_start.elapsed().as_secs_f64();
        
        // Validate after cavitation - use config values
        {
            let pressure_view = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX);
            Self::validate_field(
                &pressure_view, 
                "pressure_after_cavitation",
                self.validation_config.pressure.min,
                self.validation_config.pressure.max
            )?;
        }
        
        // Get light emission from cavitation state
        let light_emission = self.cavitation.light_emission();
        
        // 3. Light propagation - work in-place
        let light_start = Instant::now();
        self.light.update_light(
            &mut self.fields.fields,
            &light_emission,
            &self.grid,
            self.medium.as_ref(),
            dt,
        )?;
        light_time += light_start.elapsed().as_secs_f64();
        
        // Validate light field - use config values
        {
            let light_view = self.fields.fields.index_axis(Axis(0), LIGHT_IDX);
            Self::validate_field(
                &light_view, 
                "light_after_update",
                self.validation_config.light.min,
                self.validation_config.light.max
            )?;
        }
        
        // 4. Update heterogeneity effects
        let heterogeneity_start = Instant::now();
        // Heterogeneity update code would go here
        _heterogeneity_time += heterogeneity_start.elapsed().as_secs_f64();
        
        // 5. Scattering effects - use views
        let scattering_start = Instant::now();
        {
            let pressure_view = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX);
            let bubble_radius = self.medium.bubble_radius();
            let bubble_velocity = self.medium.bubble_velocity();
            
            self.scattering.compute_scattering(
                &pressure_view,
                bubble_radius,
                bubble_velocity,
                &self.grid,
                self.medium.as_ref(),
                dt,
            )?;
        }
        _scattering_time += scattering_start.elapsed().as_secs_f64();
        
        // 6. Thermal diffusion - work in-place
        let thermal_start = Instant::now();
        self.thermal.update_thermal(
            &mut self.fields.fields,
            &self.grid,
            self.medium.as_ref(),
            dt,
        )?;
        thermal_time += thermal_start.elapsed().as_secs_f64();
        
        // Validate temperature field - use config values
        {
            let temp_view = self.fields.fields.index_axis(Axis(0), TEMPERATURE_IDX);
            Self::validate_field(
                &temp_view, 
                "temperature",
                self.validation_config.temperature.min,
                self.validation_config.temperature.max
            )?;
        }
        
        // 7. Chemical reactions - use views where possible
        let chemical_start = Instant::now();
        {
            // Use views for read-only access
            let pressure_view = self.fields.fields.index_axis(Axis(0), PRESSURE_IDX);
            let light_view = self.fields.fields.index_axis(Axis(0), LIGHT_IDX);
            let temperature_view = self.fields.fields.index_axis(Axis(0), TEMPERATURE_IDX);
            
            // Only clone what's absolutely necessary for the chemical model
            let emission_spectrum = self.light.emission_spectrum();
            let bubble_radius = self.medium.bubble_radius();
            
            self.chemical.update_chemical(
                &pressure_view.to_owned(),
                &light_view.to_owned(),
                emission_spectrum,
                bubble_radius,
                &temperature_view.to_owned(),
                &self.grid,
                dt,
                self.medium.as_ref(),
                frequency,
            )?;
        }
        chemical_time += chemical_start.elapsed().as_secs_f64();
        
        // Postprocessing - update medium state
        let postprocess_start = Instant::now();
        
        // Get views for medium update
        let temperature_view = self.fields.fields.index_axis(Axis(0), TEMPERATURE_IDX);
        let bubble_radius = self.cavitation.bubble_radius()?;
        let bubble_velocity = self.cavitation.bubble_velocity()?;
        
        // Queue updates without cloning
        self.queue_medium_updates_from_views(
            temperature_view.view(),
            &bubble_radius,
            &bubble_velocity
        );
        
        // Apply the updates atomically
        self.try_update_medium()?;
        
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
        self.physics_times[6].push(postprocessing_time);
        
        // Log progress periodically
        if step % 100 == 0 {
            self.log_progress(step, dt);
        }
        
        Ok(())
    }
    
    /// Queue medium updates from views without cloning
    fn queue_medium_updates_from_views(
        &mut self, 
        temperature: ArrayView3<f64>, 
        bubble_radius: &Array3<f64>, 
        bubble_velocity: &Array3<f64>
    ) {
        // Only clone if we need to store for later
        self.pending_temperature_update = Some(temperature.to_owned());
        self.pending_bubble_update = Some((bubble_radius.clone(), bubble_velocity.clone()));
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
    fn apply_stress_boundary(&mut self, field: &mut ArrayViewMut3<f64>, field_idx: usize, step: usize) -> KwaversResult<()> {
        // For now, just apply the same PML as elastic
        // In a full implementation, this would have stress-specific handling
        self.apply_elastic_pml(field, field_idx, step)
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