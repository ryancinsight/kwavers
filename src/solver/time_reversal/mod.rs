//! Time-Reversal Image Reconstruction Module
//!
//! This module implements time-reversal reconstruction algorithms for ultrasound imaging,
//! providing methods for focusing acoustic waves back to their sources using recorded
//! boundary data.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for time-reversal operations
//! - **CUPID**: Composable with other solver components
//! - **DRY**: Reuses existing grid and solver infrastructure
//! - **KISS**: Simple interface for complex algorithms

use crate::{
    error::{KwaversError, KwaversResult},
    grid::Grid,
    solver::{Solver, PRESSURE_IDX},
    sensor::{SensorData},
    validation::{ValidationBuilder, ValidationManager},
    recorder::Recorder,
};
use ndarray::{Array3, Array4, Axis};
use rayon::prelude::*;
use std::collections::HashMap;
use log::{info, debug, warn};

/// Configuration for time-reversal reconstruction
#[derive(Debug, Clone)]
pub struct TimeReversalConfig {
    /// Whether to apply frequency filtering during reconstruction
    pub apply_frequency_filter: bool,
    
    /// Frequency range for filtering (Hz)
    pub frequency_range: Option<(f64, f64)>,
    
    /// Whether to use amplitude correction
    pub amplitude_correction: bool,
    
    /// Time window for reconstruction (seconds)
    pub time_window: Option<(f64, f64)>,
    
    /// Whether to apply spatial windowing
    pub spatial_windowing: bool,
    
    /// Number of iterations for iterative reconstruction
    pub iterations: usize,
    
    /// Convergence tolerance for iterative methods
    pub tolerance: f64,
    
    /// Whether to use GPU acceleration
    #[cfg(feature = "gpu")]
    pub use_gpu: bool,
}

impl Default for TimeReversalConfig {
    fn default() -> Self {
        Self {
            apply_frequency_filter: true,
            frequency_range: None,
            amplitude_correction: true,
            time_window: None,
            spatial_windowing: false,
            iterations: 1,
            tolerance: 1e-6,
            #[cfg(feature = "gpu")]
            use_gpu: true,
        }
    }
}

/// Time-reversal reconstruction manager
pub struct TimeReversalReconstructor {
    config: TimeReversalConfig,
    validation_manager: ValidationManager,
}

impl TimeReversalReconstructor {
    /// Create a new time-reversal reconstructor
    pub fn new(config: TimeReversalConfig) -> KwaversResult<Self> {
        // Validate configuration
        let validation_manager = ValidationBuilder::new()
            .add_check("iterations", |c: &TimeReversalConfig| {
                if c.iterations == 0 {
                    Err("Iterations must be at least 1".into())
                } else {
                    Ok(())
                }
            })
            .add_check("tolerance", |c: &TimeReversalConfig| {
                if c.tolerance <= 0.0 || c.tolerance >= 1.0 {
                    Err("Tolerance must be between 0 and 1".into())
                } else {
                    Ok(())
                }
            })
            .add_check("frequency_range", |c: &TimeReversalConfig| {
                if let Some((f_min, f_max)) = c.frequency_range {
                    if f_min >= f_max || f_min < 0.0 {
                        Err("Invalid frequency range".into())
                    } else {
                        Ok(())
                    }
                } else {
                    Ok(())
                }
            })
            .build();
        
        validation_manager.validate(&config)?;
        
        Ok(Self {
            config,
            validation_manager,
        })
    }
    
    /// Perform time-reversal reconstruction
    pub fn reconstruct(
        &self,
        sensor_data: &SensorData,
        grid: &Grid,
        solver: &mut Solver,
        recorder: &mut Recorder,
        frequency: f64,
    ) -> KwaversResult<Array3<f64>> {
        info!("Starting time-reversal reconstruction");
        
        // Validate inputs
        self.validate_inputs(sensor_data, grid)?;
        
        // Prepare time-reversed signals
        let reversed_signals = self.prepare_reversed_signals(sensor_data)?;
        
        // Initialize reconstruction field
        let mut reconstruction = Array3::<f64>::zeros(grid.shape());
        
        // Perform reconstruction iterations
        for iteration in 0..self.config.iterations {
            debug!("Time-reversal iteration {}/{}", iteration + 1, self.config.iterations);
            
            // Reset pressure field to zero
            solver.fields.fields.index_axis_mut(ndarray::Axis(0), PRESSURE_IDX).fill(0.0);
            
            // Apply time-reversed signals as sources
            self.apply_reversed_sources(&reversed_signals, solver)?;
            
            // Propagate backwards in time
            let iteration_result = self.propagate_backwards(
                grid,
                solver,
                recorder,
                frequency,
                &reversed_signals,
            )?;
            
            // Accumulate reconstruction
            reconstruction = reconstruction + &iteration_result;
            
            // Check convergence for iterative methods
            if self.config.iterations > 1 {
                let convergence = self.check_convergence(&reconstruction, &iteration_result)?;
                if convergence < self.config.tolerance {
                    info!("Converged after {} iterations", iteration + 1);
                    break;
                }
            }
        }
        
        // Apply post-processing
        let reconstruction = self.post_process(reconstruction, grid)?;
        
        info!("Time-reversal reconstruction completed");
        Ok(reconstruction)
    }
    
    /// Validate input data
    fn validate_inputs(&self, sensor_data: &SensorData, grid: &Grid) -> KwaversResult<()> {
        if sensor_data.is_empty() {
            return Err(KwaversError::ValidationError(
                "Sensor data is empty".into()
            ));
        }
        
        // Check sensor positions are within grid
        for sensor in sensor_data.sensors() {
            let pos = sensor.position();
            if pos[0] >= grid.nx() || pos[1] >= grid.ny() || pos[2] >= grid.nz() {
                return Err(KwaversError::ValidationError(
                    format!("Sensor position {:?} is outside grid bounds", pos)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Prepare time-reversed signals
    fn prepare_reversed_signals(&self, sensor_data: &SensorData) -> KwaversResult<HashMap<usize, Vec<f64>>> {
        let mut reversed_signals = HashMap::new();
        
        for (sensor_id, data) in sensor_data.data_iter() {
            let mut reversed = data.to_vec();
            reversed.reverse();
            
            // Apply frequency filtering if configured
            if self.config.apply_frequency_filter {
                reversed = self.apply_frequency_filter(reversed)?;
            }
            
            // Apply amplitude correction if configured
            if self.config.amplitude_correction {
                reversed = self.apply_amplitude_correction(reversed)?;
            }
            
            reversed_signals.insert(*sensor_id, reversed);
        }
        
        Ok(reversed_signals)
    }
    
    /// Apply frequency filter to signal
    fn apply_frequency_filter(&self, signal: Vec<f64>) -> KwaversResult<Vec<f64>> {
        if let Some((f_min, f_max)) = self.config.frequency_range {
            // TODO: Implement FFT-based frequency filtering
            // For now, return the signal as-is
            warn!("Frequency filtering not yet implemented");
            Ok(signal)
        } else {
            Ok(signal)
        }
    }
    
    /// Apply amplitude correction
    fn apply_amplitude_correction(&self, signal: Vec<f64>) -> KwaversResult<Vec<f64>> {
        // Apply geometric spreading correction
        // TODO: Implement proper amplitude correction based on propagation distance
        let corrected: Vec<f64> = signal.iter()
            .enumerate()
            .map(|(i, &val)| {
                let time_factor = (i as f64 + 1.0).sqrt();
                val * time_factor
            })
            .collect();
        
        Ok(corrected)
    }
    
    /// Apply reversed sources to the solver
    fn apply_reversed_sources(
        &self,
        reversed_signals: &HashMap<usize, Vec<f64>>,
        solver: &mut Solver,
    ) -> KwaversResult<()> {
        // Clear existing sources
        solver.clear_sources();
        
        // Add time-reversed sources at sensor positions
        for (sensor_id, signal) in reversed_signals {
            // TODO: Create source at sensor position with reversed signal
            debug!("Adding reversed source for sensor {}", sensor_id);
        }
        
        Ok(())
    }
    
    /// Propagate waves backwards in time
    fn propagate_backwards(
        &self,
        grid: &Grid,
        solver: &mut Solver,
        recorder: &mut Recorder,
        frequency: f64,
        reversed_signals: &HashMap<usize, Vec<f64>>,
    ) -> KwaversResult<Array3<f64>> {
        let mut max_amplitude_field = Array3::<f64>::zeros(grid.shape());
        let time_steps = reversed_signals.values()
            .map(|v| v.len())
            .max()
            .unwrap_or(0);
        
        // Propagate for the duration of the reversed signals
        for step in 0..time_steps {
            // Update sources with current time step values
            for (sensor_id, signal) in reversed_signals {
                if step < signal.len() {
                    // TODO: Update source amplitude for this time step
                    let _amplitude = signal[step];
                }
            }
            
            // Advance solver one step
            solver.step(step, solver.time.dt, frequency)?;
            
            // Track maximum amplitude at each point
            let pressure = solver.fields.fields.index_axis(ndarray::Axis(0), PRESSURE_IDX);
            max_amplitude_field.par_iter_mut()
                .zip(pressure.par_iter())
                .for_each(|(max_val, &current_val)| {
                    *max_val = max_val.max(current_val.abs());
                });
            
            // Record if needed
            if step % 10 == 0 {
                recorder.record(&solver.fields, &solver.grid, step)?;
            }
        }
        
        Ok(max_amplitude_field)
    }
    
    /// Check convergence between iterations
    fn check_convergence(
        &self,
        current: &Array3<f64>,
        previous: &Array3<f64>,
    ) -> KwaversResult<f64> {
        let diff = current - previous;
        let norm_diff = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let norm_current = current.iter().map(|&x| x * x).sum::<f64>().sqrt();
        
        if norm_current > 0.0 {
            Ok(norm_diff / norm_current)
        } else {
            Ok(0.0)
        }
    }
    
    /// Post-process the reconstruction
    fn post_process(&self, mut reconstruction: Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
        // Apply spatial windowing if configured
        if self.config.spatial_windowing {
            reconstruction = self.apply_spatial_window(reconstruction, grid)?;
        }
        
        // Normalize the reconstruction
        let max_val = reconstruction.iter()
            .map(|&x| x.abs())
            .fold(0.0, f64::max);
        
        if max_val > 0.0 {
            reconstruction.mapv_inplace(|x| x / max_val);
        }
        
        Ok(reconstruction)
    }
    
    /// Apply spatial windowing function
    fn apply_spatial_window(&self, mut field: Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = (grid.nx(), grid.ny(), grid.nz());
        
        // Apply Tukey window in each dimension
        let alpha = 0.1; // Taper parameter
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let wx = tukey_window(i, nx, alpha);
                    let wy = tukey_window(j, ny, alpha);
                    let wz = tukey_window(k, nz, alpha);
                    
                    field[[i, j, k]] *= wx * wy * wz;
                }
            }
        }
        
        Ok(field)
    }
}

/// Tukey window function
fn tukey_window(i: usize, n: usize, alpha: f64) -> f64 {
    let x = i as f64 / (n - 1) as f64;
    
    if x < alpha / 2.0 {
        0.5 * (1.0 + (2.0 * std::f64::consts::PI * x / alpha).cos())
    } else if x > 1.0 - alpha / 2.0 {
        0.5 * (1.0 + (2.0 * std::f64::consts::PI * (1.0 - x) / alpha).cos())
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_time_reversal_config() {
        let config = TimeReversalConfig::default();
        assert_eq!(config.iterations, 1);
        assert!(config.apply_frequency_filter);
        assert!(config.amplitude_correction);
    }
    
    #[test]
    fn test_tukey_window() {
        let n = 100;
        let alpha = 0.1;
        
        // Test edge values
        assert!((tukey_window(0, n, alpha) - 0.0).abs() < 1e-10);
        assert!((tukey_window(n - 1, n, alpha) - 0.0).abs() < 1e-10);
        
        // Test middle value
        assert!((tukey_window(n / 2, n, alpha) - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_config_validation() {
        // Valid config
        let config = TimeReversalConfig {
            iterations: 5,
            tolerance: 1e-4,
            ..Default::default()
        };
        assert!(TimeReversalReconstructor::new(config).is_ok());
        
        // Invalid iterations
        let config = TimeReversalConfig {
            iterations: 0,
            ..Default::default()
        };
        assert!(TimeReversalReconstructor::new(config).is_err());
        
        // Invalid tolerance
        let config = TimeReversalConfig {
            tolerance: 1.5,
            ..Default::default()
        };
        assert!(TimeReversalReconstructor::new(config).is_err());
    }
}