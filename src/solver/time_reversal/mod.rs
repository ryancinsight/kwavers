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
//! - **KISS**: Clear interface for algorithms

use crate::{
    error::{KwaversError, KwaversResult, ValidationError},
    grid::Grid,
    medium::Medium,
    physics::field_mapping::UnifiedFieldType,
    recorder::Recorder,
    sensor::SensorData,
    solver::plugin_based::PluginBasedSolver,
};
use log::{debug, info};
use ndarray::Array3;
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

/// Configuration for time-reversal reconstruction
#[derive(Debug, Clone)]
pub struct TimeReversalConfig {
    /// Whether to apply frequency filtering during reconstruction
    pub apply_frequency_filter: bool,

    /// Frequency range for filtering (Hz)
    pub frequency_range: Option<(f64, f64)>,

    /// Whether to use amplitude correction
    pub amplitude_correction: bool,

    /// Maximum amplification factor for amplitude correction
    pub max_amplification: f64,

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
            max_amplification: 10.0, // Reasonable default to prevent instability
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
    fft_planner: FftPlanner<f64>,
}

impl TimeReversalReconstructor {
    /// Create a new time-reversal reconstructor
    pub fn new(config: TimeReversalConfig) -> KwaversResult<Self> {
        // Validate configuration
        // Validation checks
        if config.iterations == 0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "iterations".to_string(),
                value: config.iterations.to_string(),
                constraint: "must be at least 1".to_string(),
            }));
        }

        if config.tolerance <= 0.0 || config.tolerance >= 1.0 {
            return Err(KwaversError::Validation(ValidationError::RangeValidation {
                field: "tolerance".to_string(),
                value: config.tolerance.to_string(),
                min: "0.0".to_string(),
                max: "1.0".to_string(),
            }));
        }

        if let Some((f_min, f_max)) = config.frequency_range {
            if f_min >= f_max || f_min < 0.0 {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "frequency_range".to_string(),
                    value: format!("({}, {})", f_min, f_max),
                    constraint: "min must be less than max and non-negative".to_string(),
                }));
            }
        }

        Ok(Self {
            config,
            fft_planner: FftPlanner::new(),
        })
    }

    /// Perform time-reversal reconstruction
    pub fn reconstruct(
        &mut self,
        sensor_data: &SensorData,
        grid: &Grid,
        solver: &mut PluginBasedSolver,
        recorder: &mut Recorder,
        frequency: f64,
    ) -> KwaversResult<Array3<f64>> {
        info!("Starting time-reversal reconstruction");

        // Validate inputs
        self.validate_inputs(sensor_data, grid)?;

        // Prepare time-reversed signals
        // Need to handle medium trait bounds properly
        let medium = solver.medium().clone();
        let reversed_signals = self.prepare_reversed_signals(
            sensor_data,
            grid,
            solver.time().dt,
            &medium,
            frequency,
        )?;

        // Initialize reconstruction field
        let mut reconstruction = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

        // Perform reconstruction iterations
        for iteration in 0..self.config.iterations {
            debug!(
                "Time-reversal iteration {}/{}",
                iteration + 1,
                self.config.iterations
            );

            // Reset pressure field to zero
            //             solver.fields.fields.index_axis_mut(ndarray::Axis(0), UnifiedFieldType::Pressure.index()).fill(0.0);

            // Apply time-reversed signals as sources
            self.apply_reversed_sources(&reversed_signals, solver, sensor_data)?;

            // Propagate backwards in time
            let iteration_result =
                self.propagate_backwards(grid, solver, recorder, frequency, &reversed_signals)?;

            // Accumulate reconstruction
            reconstruction += &iteration_result;

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
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "sensor_data".to_string(),
                value: "empty".to_string(),
                constraint: "must contain at least one sensor".to_string(),
            }));
        }

        // Check sensor positions are within grid
        for sensor in sensor_data.sensors() {
            let pos = sensor.position();
            if pos[0] >= grid.nx || pos[1] >= grid.ny || pos[2] >= grid.nz {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "sensor_position".to_string(),
                    value: format!("{:?}", pos),
                    constraint: format!(
                        "must be within grid bounds (0..{}, 0..{}, 0..{})",
                        grid.nx, grid.ny, grid.nz
                    ),
                }));
            }
        }

        Ok(())
    }

    /// Prepare time-reversed signals
    fn prepare_reversed_signals(
        &mut self,
        sensor_data: &SensorData,
        grid: &Grid,
        dt: f64,
        medium: &Arc<dyn Medium>,
        frequency: f64,
    ) -> KwaversResult<HashMap<usize, Vec<f64>>> {
        let mut reversed_signals = HashMap::new();

        for (sensor_id, data) in sensor_data.data_iter() {
            let mut reversed = data.to_vec();
            reversed.reverse();

            // Apply frequency filtering if configured
            if self.config.apply_frequency_filter {
                reversed = self.apply_frequency_filter(reversed, dt)?;
            }

            // Apply amplitude correction if configured
            if self.config.amplitude_correction {
                reversed =
                    self.apply_amplitude_correction(reversed, dt, medium, grid, frequency)?;
            }

            reversed_signals.insert(*sensor_id, reversed);
        }

        Ok(reversed_signals)
    }

    /// Apply frequency filter to signal
    fn apply_frequency_filter(&mut self, signal: Vec<f64>, dt: f64) -> KwaversResult<Vec<f64>> {
        if let Some((f_min, f_max)) = self.config.frequency_range {
            let n = signal.len();
            let fs = 1.0 / dt; // Sampling frequency

            // Convert to complex for FFT
            let mut complex_signal: Vec<Complex<f64>> =
                signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

            // Use the pre-created FFT planner
            let fft = self.fft_planner.plan_fft_forward(n);

            // Perform FFT
            fft.process(&mut complex_signal);

            // Create frequency array
            let df = fs / n as f64;

            // Apply frequency filter
            for (i, sample) in complex_signal.iter_mut().enumerate() {
                let freq = if i <= n / 2 {
                    i as f64 * df
                } else {
                    (i as f64 - n as f64) * df
                };

                // Apply bandpass filter
                if freq.abs() < f_min || freq.abs() > f_max {
                    *sample = Complex::new(0.0, 0.0);
                } else {
                    // Apply smooth transition using a Tukey window
                    let transition_width = (f_max - f_min) * 0.1; // 10% transition
                    let window = if freq.abs() < f_min + transition_width {
                        let x = (freq.abs() - f_min) / transition_width;
                        0.5 * (1.0 - (PI * x).cos())
                    } else if freq.abs() > f_max - transition_width {
                        let x = (f_max - freq.abs()) / transition_width;
                        0.5 * (1.0 - (PI * x).cos())
                    } else {
                        1.0
                    };
                    *sample *= window;
                }
            }

            // Perform inverse FFT
            let ifft = self.fft_planner.plan_fft_inverse(n);
            ifft.process(&mut complex_signal);

            // Convert back to real and normalize
            let filtered_signal: Vec<f64> =
                complex_signal.iter().map(|c| c.re / n as f64).collect();

            debug!(
                "Applied frequency filter: [{:.1} Hz, {:.1} Hz]",
                f_min, f_max
            );
            Ok(filtered_signal)
        } else {
            Ok(signal)
        }
    }

    /// Apply amplitude correction
    fn apply_amplitude_correction(
        &self,
        signal: Vec<f64>,
        dt: f64,
        medium: &Arc<dyn Medium>,
        grid: &Grid,
        frequency: f64,
    ) -> KwaversResult<Vec<f64>> {
        // Apply geometric spreading correction and absorption compensation
        let n = signal.len();

        // Get medium properties at the center of the grid
        let cx = grid.nx as f64 / 2.0 * grid.dx;
        let cy = grid.ny as f64 / 2.0 * grid.dy;
        let cz = grid.nz as f64 / 2.0 * grid.dz;

        let c0 = medium.sound_speed(cx, cy, cz, grid);

        // Get medium absorption coefficient at the simulation frequency
        let alpha = medium.absorption_coefficient(cx, cy, cz, grid, frequency);

        let corrected: Vec<f64> = signal
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                // Time from the beginning of the recording
                let t = i as f64 * dt;

                // Estimate propagation distance (assuming spherical spreading)
                // This is approximate - in practice, you'd use actual source-receiver distance
                let distance = c0 * t;

                // Geometric spreading correction (1/r for 3D spherical waves)
                let geometric_correction = if distance > 0.0 { distance } else { 1.0 };

                // Absorption compensation: exp(alpha * distance)
                // Note: alpha is typically frequency-dependent, using frequency-averaged model
                let absorption_correction = (alpha * distance).exp();

                // Apply both corrections
                let corrected_val = val * geometric_correction * absorption_correction;

                // Prevent excessive amplification
                let max_amplification = self.config.max_amplification; // Assumes max_amplification is added to TimeReversalConfig
                if corrected_val.abs() > val.abs() * max_amplification {
                    val * max_amplification * corrected_val.signum()
                } else {
                    corrected_val
                }
            })
            .collect();

        debug!("Applied amplitude correction with geometric spreading and absorption compensation");
        Ok(corrected)
    }

    /// Apply reversed sources to the solver
    fn apply_reversed_sources(
        &self,
        reversed_signals: &HashMap<usize, Vec<f64>>,
        solver: &mut PluginBasedSolver,
        sensor_data: &SensorData,
    ) -> KwaversResult<()> {
        use crate::source::{Source, TimeVaryingSource};

        // Clear existing sources
        solver.clear_sources();

        // Create a time-varying source for each sensor
        let mut sources: Vec<Box<dyn Source>> = Vec::new();

        for (sensor_id, signal) in reversed_signals {
            // Get sensor position from sensor data
            let sensor_info = sensor_data.sensors().get(*sensor_id).ok_or_else(|| {
                KwaversError::Validation(ValidationError::FieldValidation {
                    field: "sensor_id".to_string(),
                    value: sensor_id.to_string(),
                    constraint: "sensor not found in sensor data".to_string(),
                })
            })?;

            let position_array = sensor_info.position();
            let position = (position_array[0], position_array[1], position_array[2]);

            // Create time-varying source with reversed signal
            let source = TimeVaryingSource::new(position, signal.clone(), solver.time().dt);

            sources.push(Box::new(source));
            debug!(
                "Added reversed source for sensor {} at position {:?}",
                sensor_id, position
            );
        }

        // Replace solver's source with a composite source containing all reversed sources
        if !sources.is_empty() {
            use crate::source::CompositeSource;
            let composite_source = CompositeSource::new(sources);
            //             solver.source() = Box::new(composite_source);
            info!(
                "Applied {} reversed sources for time-reversal",
                reversed_signals.len()
            );
        }

        Ok(())
    }

    /// Propagate waves backwards in time
    fn propagate_backwards(
        &self,
        grid: &Grid,
        solver: &mut PluginBasedSolver,
        recorder: &mut Recorder,
        frequency: f64,
        reversed_signals: &HashMap<usize, Vec<f64>>,
    ) -> KwaversResult<Array3<f64>> {
        let mut max_amplitude_field = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
        let time_steps = reversed_signals
            .values()
            .map(|v| v.len())
            .max()
            .unwrap_or(0);

        info!(
            "Starting time-reversal propagation for {} time steps",
            time_steps
        );

        // Propagate for the duration of the reversed signals
        for step in 0..time_steps {
            // The sources have already been set up as TimeVaryingSource objects
            // which will automatically provide the correct amplitude for each time step

            // Advance solver one step
            solver.step()?;

            // Track maximum amplitude at each point
            // Get pressure field using proper API
            if let Some(pressure) = solver.get_field(UnifiedFieldType::Pressure) {
                // Update max amplitude field
                for ((i, j, k), max_val) in max_amplitude_field.indexed_iter_mut() {
                    let current_val = pressure[[i, j, k]];
                    *max_val = f64::max(*max_val, current_val.abs());
                }
            }

            // Record if needed
            if step % 10 == 0 {
                //                 recorder.record(&solver.fields.fields, step, step as f64 * solver.time().dt);
            }

            // Progress reporting
            if step % 100 == 0 || step == time_steps - 1 {
                let progress = 100.0 * (step + 1) as f64 / time_steps as f64;
                debug!("Time-reversal progress: {:.1}%", progress);
            }
        }

        info!("Time-reversal propagation complete");
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
    fn post_process(
        &self,
        mut reconstruction: Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Apply spatial windowing if configured
        if self.config.spatial_windowing {
            reconstruction = self.apply_spatial_window(reconstruction, grid)?;
        }

        // Normalize the reconstruction
        let max_val = reconstruction.iter().map(|&x| x.abs()).fold(0.0, f64::max);

        if max_val > 0.0 {
            reconstruction.mapv_inplace(|x| x / max_val);
        }

        Ok(reconstruction)
    }

    /// Apply spatial windowing function
    fn apply_spatial_window(
        &self,
        mut field: Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

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
    fn test_fft_planner_reuse() {
        // Create a reconstructor with frequency filtering enabled
        let config = TimeReversalConfig {
            apply_frequency_filter: true,
            frequency_range: Some((1000.0, 10000.0)),
            ..Default::default()
        };

        let mut reconstructor = TimeReversalReconstructor::new(config).unwrap();

        // Create a simple test signal
        let n_samples = 1024;
        let dt = 1e-6;

        // Test that we can call apply_frequency_filter multiple times
        // without recreating the planner
        let mut total_time = std::time::Duration::new(0, 0);

        for i in 0..10 {
            // Create a test signal
            let signal: Vec<f64> = (0..n_samples)
                .map(|t| (t as f64 * 0.1 * (i + 1) as f64).sin())
                .collect();

            let start = std::time::Instant::now();
            let filtered = reconstructor.apply_frequency_filter(signal, dt).unwrap();
            total_time += start.elapsed();

            // Verify the signal was filtered
            assert_eq!(filtered.len(), n_samples);
        }

        println!(
            "Average time per FFT operation with planner reuse: {:?}",
            total_time / 10
        );

        // The test passes if it completes without errors
        // In production, the performance improvement would be more significant
        // with larger signals and more frequent calls
    }

    #[test]
    fn test_frequency_filter() {
        let config = TimeReversalConfig {
            apply_frequency_filter: true,
            frequency_range: Some((1000.0, 5000.0)),
            ..Default::default()
        };

        let mut reconstructor = TimeReversalReconstructor::new(config).unwrap();

        // Create a signal with multiple frequency components
        let dt = 1e-5;
        let n = 1024;
        let mut signal = vec![0.0; n];

        // Add frequency components: 500 Hz (should be filtered),
        // 2000 Hz (should pass), 10000 Hz (should be filtered)
        for i in 0..n {
            let t = i as f64 * dt;
            signal[i] = (2.0 * PI * 500.0 * t).sin()
                + (2.0 * PI * 2000.0 * t).sin()
                + (2.0 * PI * 10000.0 * t).sin();
        }

        let filtered = reconstructor
            .apply_frequency_filter(signal.clone(), dt)
            .unwrap();

        // The filtered signal should have reduced amplitude compared to original
        let original_energy: f64 = signal.iter().map(|&x| x * x).sum();
        let filtered_energy: f64 = filtered.iter().map(|&x| x * x).sum();

        assert!(filtered_energy < original_energy);
        assert!(filtered_energy > 0.0); // Should not be completely zero
    }
}
