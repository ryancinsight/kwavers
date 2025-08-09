//! Reconstruction Algorithms Module
//!
//! This module provides image reconstruction algorithms compatible with k-Wave toolbox,
//! including linear and planar array reconstruction for photoacoustic and ultrasound imaging.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for reconstruction operations
//! - **CUPID**: Composable with sensor and solver components
//! - **DRY**: Reuses FFT and grid infrastructure
//! - **Zero-Copy**: Uses iterators and slices for efficiency
//!
//! # Literature References
//! - Treeby & Cox (2010): "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields"
//! - Xu & Wang (2006): "Universal back-projection algorithm for photoacoustic computed tomography"
//! - Burgholzer et al. (2007): "Exact and approximative imaging methods for photoacoustic tomography"

use crate::{
    error::{KwaversError, KwaversResult},
    grid::Grid,
    sensor::SensorData,
};
use ndarray::{Array1, Array2, Array3, Axis, Zip, s};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;
use rayon::prelude::*;

/// Configuration for linear array reconstruction
#[derive(Debug, Clone)]
pub struct LineReconConfig {
    /// Sensor element positions [n_elements x 3]
    pub sensor_positions: Array2<f64>,
    
    /// Speed of sound in medium (m/s)
    pub speed_of_sound: f64,
    
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    
    /// Interpolation method
    pub interpolation: InterpolationMethod,
    
    /// Apply frequency filtering
    pub apply_filter: bool,
    
    /// Frequency filter range (Hz)
    pub filter_range: Option<(f64, f64)>,
    
    /// Apply envelope detection
    pub envelope_detection: bool,
    
    /// Apply log compression
    pub log_compression: bool,
    
    /// Dynamic range for log compression (dB)
    pub dynamic_range: f64,
}

/// Configuration for planar array reconstruction
#[derive(Debug, Clone)]
pub struct PlaneReconConfig {
    /// Sensor element positions [n_elements x 3]
    pub sensor_positions: Array2<f64>,
    
    /// Speed of sound in medium (m/s)
    pub speed_of_sound: f64,
    
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    
    /// Interpolation method
    pub interpolation: InterpolationMethod,
    
    /// Apply frequency filtering
    pub apply_filter: bool,
    
    /// Frequency filter range (Hz)
    pub filter_range: Option<(f64, f64)>,
    
    /// Apply directivity correction
    pub directivity_correction: bool,
    
    /// Sensor element size for directivity
    pub element_size: Option<f64>,
}

/// Interpolation methods for reconstruction
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation
    Nearest,
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
}

/// Linear array reconstruction (kspaceLineRecon equivalent)
///
/// Implements delay-and-sum beamforming for linear array transducers
pub struct LineRecon {
    config: LineReconConfig,
    fft_planner: FftPlanner<f64>,
}

impl LineRecon {
    /// Create a new linear array reconstruction instance
    pub fn new(config: LineReconConfig) -> Self {
        Self {
            config,
            fft_planner: FftPlanner::new(),
        }
    }
    
    /// Reconstruct image from sensor data
    ///
    /// # Arguments
    /// * `sensor_data` - Recorded pressure data [n_time x n_elements]
    /// * `grid` - Reconstruction grid
    ///
    /// # Returns
    /// Reconstructed image volume
    pub fn reconstruct(&mut self, sensor_data: &Array2<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
        let n_time = sensor_data.nrows();
        let n_elements = sensor_data.ncols();
        
        // Validate sensor configuration
        if n_elements != self.config.sensor_positions.nrows() {
            return Err(KwaversError::field_validation(
                "sensor_elements",
                n_elements,
                &format!("expected {} elements", self.config.sensor_positions.nrows())
            ));
        }
        
        // Initialize reconstruction volume
        let mut image = grid.zeros_array();
        
        // Apply frequency filtering if requested
        let filtered_data = if self.config.apply_filter {
            self.apply_frequency_filter(sensor_data)?
        } else {
            sensor_data.clone()
        };
        
        // Time vector
        let dt = 1.0 / self.config.sampling_frequency;
        let time: Vec<f64> = (0..n_time).map(|i| i as f64 * dt).collect();
        
        // Perform delay-and-sum beamforming using parallel iterators
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let dx = grid.dx;
        
        // Use parallel chunks for better performance
        let image_slice = image.as_slice_mut().unwrap();
        image_slice.par_chunks_mut(nx).enumerate().for_each(|(iz, chunk)| {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = iy * nx + ix;
                    if idx < chunk.len() {
                        // Reconstruction point in physical coordinates
                        let x = ix as f64 * dx;
                        let y = iy as f64 * dx;
                        let z = iz as f64 * dx;
                        
                        // Accumulate contributions from all sensor elements
                        let mut sum = 0.0;
                        for elem in 0..n_elements {
                            // Calculate distance from sensor to reconstruction point
                            let sensor_pos = self.config.sensor_positions.row(elem);
                            let distance = ((x - sensor_pos[0]).powi(2) +
                                          (y - sensor_pos[1]).powi(2) +
                                          (z - sensor_pos[2]).powi(2)).sqrt();
                            
                            // Calculate time delay
                            let delay = distance / self.config.speed_of_sound;
                            
                            // Find corresponding time index
                            let time_idx = delay / dt;
                            
                            // Interpolate sensor data at delayed time
                            if time_idx >= 0.0 && time_idx < (n_time - 1) as f64 {
                                let value = self.interpolate(&filtered_data.column(elem), time_idx);
                                
                                // Apply distance-based amplitude correction
                                let amplitude_correction = if distance > 0.0 {
                                    1.0 / distance.sqrt()
                                } else {
                                    1.0
                                };
                                
                                sum += value * amplitude_correction;
                            }
                        }
                        
                        chunk[idx] = sum / n_elements as f64;
                    }
                }
            }
        });
        
        // Reshape back to 3D
        let mut image_3d = Array3::from_shape_vec((nx, ny, nz), image.as_slice().unwrap().to_vec())
            .map_err(|e| KwaversError::Numerical(crate::error::NumericalError::Instability {
                operation: "reshape".to_string(),
                condition: format!("Failed to reshape image: {}", e),
            }))?;
        
        // Apply envelope detection if requested
        if self.config.envelope_detection {
            image_3d = self.apply_envelope_detection(image_3d)?;
        }
        
        // Apply log compression if requested
        if self.config.log_compression {
            image_3d = self.apply_log_compression(image_3d);
        }
        
        Ok(image_3d)
    }
    
    /// Interpolate sensor data at fractional time index
    fn interpolate(&self, data: &ndarray::ArrayView1<f64>, index: f64) -> f64 {
        match self.config.interpolation {
            InterpolationMethod::Nearest => {
                let idx = index.round() as usize;
                if idx < data.len() {
                    data[idx]
                } else {
                    0.0
                }
            }
            InterpolationMethod::Linear => {
                let idx0 = index.floor() as usize;
                let idx1 = idx0 + 1;
                
                if idx1 < data.len() {
                    let t = index - idx0 as f64;
                    data[idx0] * (1.0 - t) + data[idx1] * t
                } else {
                    0.0
                }
            }
            InterpolationMethod::Cubic => {
                // Catmull-Rom cubic interpolation
                let idx1 = index.floor() as usize;
                let t = index - idx1 as f64;
                
                if idx1 > 0 && idx1 + 2 < data.len() {
                    let idx0 = idx1 - 1;
                    let idx2 = idx1 + 1;
                    let idx3 = idx1 + 2;
                    
                    let v0 = data[idx0];
                    let v1 = data[idx1];
                    let v2 = data[idx2];
                    let v3 = data[idx3];
                    
                    let t2 = t * t;
                    let t3 = t2 * t;
                    
                    0.5 * ((2.0 * v1) +
                           (-v0 + v2) * t +
                           (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v3) * t2 +
                           (-v0 + 3.0 * v1 - 3.0 * v2 + v3) * t3)
                } else {
                    // Fall back to linear interpolation when cubic cannot be applied
                    let idx0 = index.floor() as usize;
                    let idx1 = idx0 + 1;
                    
                    if idx1 < data.len() {
                        let t = index - idx0 as f64;
                        data[idx0] * (1.0 - t) + data[idx1] * t
                    } else if idx0 < data.len() {
                        data[idx0]
                    } else {
                        0.0
                    }
                }
            }
        }
    }
    
    /// Apply frequency filter to sensor data
    fn apply_frequency_filter(&mut self, data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (n_time, n_elements) = data.dim();
        let mut filtered = Array2::zeros((n_time, n_elements));
        
        if let Some((f_min, f_max)) = self.config.filter_range {
            let df = self.config.sampling_frequency / n_time as f64;
            
            for elem in 0..n_elements {
                // Convert to complex for FFT
                let mut complex_data: Vec<Complex<f64>> = data.column(elem)
                    .iter()
                    .map(|&x| Complex::new(x, 0.0))
                    .collect();
                
                // Forward FFT
                let fft = self.fft_planner.plan_fft_forward(n_time);
                fft.process(&mut complex_data);
                
                // Apply frequency filter
                for (i, value) in complex_data.iter_mut().enumerate() {
                    let freq = i as f64 * df;
                    if freq < f_min || freq > f_max {
                        *value = Complex::new(0.0, 0.0);
                    }
                }
                
                // Inverse FFT
                let ifft = self.fft_planner.plan_fft_inverse(n_time);
                ifft.process(&mut complex_data);
                
                // Extract real part and normalize
                for (i, &c) in complex_data.iter().enumerate() {
                    filtered[[i, elem]] = c.re / n_time as f64;
                }
            }
        } else {
            filtered.assign(data);
        }
        
        Ok(filtered)
    }
    
    /// Apply envelope detection using Hilbert transform
    fn apply_envelope_detection(&mut self, mut image: Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = image.dim();
        
        // Apply Hilbert transform along each line
        for iz in 0..nz {
            for iy in 0..ny {
                let mut line = image.slice_mut(s![.., iy, iz]);
                let envelope = self.hilbert_envelope(line.as_slice().unwrap())?;
                line.assign(&Array1::from_vec(envelope));
            }
        }
        
        Ok(image)
    }
    
    /// Compute envelope using Hilbert transform
    fn hilbert_envelope(&mut self, signal: &[f64]) -> KwaversResult<Vec<f64>> {
        let n = signal.len();
        
        // Convert to complex
        let mut complex_signal: Vec<Complex<f64>> = signal
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // FFT
        let fft = self.fft_planner.plan_fft_forward(n);
        fft.process(&mut complex_signal);
        
        // Apply Hilbert transform in frequency domain
        for i in 1..n/2 {
            complex_signal[i] *= 2.0;
        }
        for i in n/2 + 1..n {
            complex_signal[i] = Complex::new(0.0, 0.0);
        }
        
        // Inverse FFT
        let ifft = self.fft_planner.plan_fft_inverse(n);
        ifft.process(&mut complex_signal);
        
        // Compute magnitude (envelope)
        Ok(complex_signal.iter()
            .map(|c| (c.norm() / n as f64))
            .collect())
    }
    
    /// Apply log compression for display
    fn apply_log_compression(&self, mut image: Array3<f64>) -> Array3<f64> {
        let max_val = image.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        
        if max_val > 0.0 {
            let threshold = max_val * 10.0_f64.powf(-self.config.dynamic_range / 20.0);
            
            Zip::from(&mut image).for_each(|x| {
                let val = x.abs();
                *x = if val > threshold {
                    20.0 * (val / max_val).log10()
                } else {
                    -self.config.dynamic_range
                };
            });
        }
        
        image
    }
}

/// Planar array reconstruction (kspacePlaneRecon equivalent)
///
/// Implements 3D reconstruction for planar sensor arrays
pub struct PlaneRecon {
    config: PlaneReconConfig,
    fft_planner: FftPlanner<f64>,
}

impl PlaneRecon {
    /// Create a new planar array reconstruction instance
    pub fn new(config: PlaneReconConfig) -> Self {
        Self {
            config,
            fft_planner: FftPlanner::new(),
        }
    }
    
    /// Reconstruct 3D image from planar sensor data
    ///
    /// # Arguments
    /// * `sensor_data` - Recorded pressure data [n_time x n_elements]
    /// * `grid` - Reconstruction grid
    ///
    /// # Returns
    /// Reconstructed 3D image volume
    pub fn reconstruct(&mut self, sensor_data: &Array2<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
        let n_time = sensor_data.nrows();
        let n_elements = sensor_data.ncols();
        
        // Validate sensor configuration
        if n_elements != self.config.sensor_positions.nrows() {
            return Err(KwaversError::field_validation(
                "sensor_elements",
                n_elements,
                &format!("expected {} elements", self.config.sensor_positions.nrows())
            ));
        }
        
        // Initialize reconstruction volume
        let mut image = grid.zeros_array();
        
        // Apply frequency filtering if requested
        let filtered_data = if self.config.apply_filter {
            self.apply_frequency_filter(sensor_data)?
        } else {
            sensor_data.clone()
        };
        
        // Perform 3D back-projection reconstruction
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let dx = grid.dx;
        let dt = 1.0 / self.config.sampling_frequency;
        
        // Use parallel processing for efficiency
        let image_slice = image.as_slice_mut().unwrap();
        image_slice.par_chunks_mut(nx * ny).enumerate().for_each(|(iz, chunk)| {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = iy * nx + ix;
                    if idx < chunk.len() {
                        // Reconstruction point
                        let x = ix as f64 * dx;
                        let y = iy as f64 * dx;
                        let z = iz as f64 * dx;
                        let recon_point = [x, y, z];
                        
                        // Accumulate contributions
                        let mut sum = 0.0;
                        let mut weight_sum = 0.0;
                        
                        for elem in 0..n_elements {
                            let sensor_pos = self.config.sensor_positions.row(elem);
                            
                            // Calculate distance and delay
                            let distance = ((recon_point[0] - sensor_pos[0]).powi(2) +
                                          (recon_point[1] - sensor_pos[1]).powi(2) +
                                          (recon_point[2] - sensor_pos[2]).powi(2)).sqrt();
                            
                            let delay = distance / self.config.speed_of_sound;
                            let time_idx = delay / dt;
                            
                            if time_idx >= 0.0 && time_idx < (n_time - 1) as f64 {
                                // Interpolate sensor data
                                let value = self.interpolate(&filtered_data.column(elem), time_idx);
                                
                                // Calculate weight with directivity correction
                                let weight = self.calculate_weight(
                                    &recon_point,
                                    sensor_pos.as_slice().unwrap(),
                                    distance
                                );
                                
                                sum += value * weight;
                                weight_sum += weight;
                            }
                        }
                        
                        chunk[idx] = if weight_sum > 0.0 {
                            sum / weight_sum
                        } else {
                            0.0
                        };
                    }
                }
            }
        });
        
        // Reshape to 3D
        let image_3d = Array3::from_shape_vec((nx, ny, nz), image.as_slice().unwrap().to_vec())
            .map_err(|e| KwaversError::Numerical(crate::error::NumericalError::Instability {
                operation: "reshape".to_string(),
                condition: format!("Failed to reshape image: {}", e),
            }))?;
        
        Ok(image_3d)
    }
    
    /// Calculate weight for back-projection including directivity
    fn calculate_weight(&self, recon_point: &[f64; 3], sensor_pos: &[f64], distance: f64) -> f64 {
        let mut weight = if distance > 0.0 {
            1.0 / distance // Spherical spreading correction
        } else {
            1.0
        };
        
        // Apply directivity correction if enabled
        if self.config.directivity_correction {
            if let Some(element_size) = self.config.element_size {
                // Calculate angle between sensor normal and reconstruction point
                let dx = recon_point[0] - sensor_pos[0];
                let dy = recon_point[1] - sensor_pos[1];
                let dz = recon_point[2] - sensor_pos[2];
                
                // Assume sensor normal points in +z direction
                let cos_theta = dz / distance;
                
                // Directivity function for circular piston
                if cos_theta > 0.0 {
                    let ka = 2.0 * PI * element_size / self.config.speed_of_sound;
                    let x = ka * (1.0 - cos_theta.powi(2)).sqrt();
                    
                    let directivity = if x.abs() < 1e-6 {
                        1.0
                    } else {
                        2.0 * x.sin() / x
                    };
                    
                    weight *= directivity * cos_theta; // Include obliquity factor
                } else {
                    weight = 0.0; // Behind sensor
                }
            }
        }
        
        weight
    }
    
    /// Interpolate sensor data (reused from LineRecon)
    fn interpolate(&self, data: &ndarray::ArrayView1<f64>, index: f64) -> f64 {
        match self.config.interpolation {
            InterpolationMethod::Nearest => {
                let idx = index.round() as usize;
                if idx < data.len() {
                    data[idx]
                } else {
                    0.0
                }
            }
            InterpolationMethod::Linear => {
                let idx0 = index.floor() as usize;
                let idx1 = idx0 + 1;
                
                if idx1 < data.len() {
                    let t = index - idx0 as f64;
                    data[idx0] * (1.0 - t) + data[idx1] * t
                } else {
                    0.0
                }
            }
            InterpolationMethod::Cubic => {
                let idx1 = index.floor() as usize;
                let t = index - idx1 as f64;
                
                if idx1 > 0 && idx1 + 2 < data.len() {
                    let v0 = data[idx1 - 1];
                    let v1 = data[idx1];
                    let v2 = data[idx1 + 1];
                    let v3 = data[idx1 + 2];
                    
                    let t2 = t * t;
                    let t3 = t2 * t;
                    
                    0.5 * ((2.0 * v1) +
                           (-v0 + v2) * t +
                           (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v3) * t2 +
                           (-v0 + 3.0 * v1 - 3.0 * v2 + v3) * t3)
                } else {
                    // Fall back to linear
                    let idx0 = index.floor() as usize;
                    let idx1 = idx0 + 1;
                    
                    if idx1 < data.len() {
                        let t = index - idx0 as f64;
                        data[idx0] * (1.0 - t) + data[idx1] * t
                    } else {
                        0.0
                    }
                }
            }
        }
    }
    
    /// Apply frequency filter (reused from LineRecon)
    fn apply_frequency_filter(&mut self, data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (n_time, n_elements) = data.dim();
        let mut filtered = Array2::zeros((n_time, n_elements));
        
        if let Some((f_min, f_max)) = self.config.filter_range {
            let df = self.config.sampling_frequency / n_time as f64;
            
            for elem in 0..n_elements {
                let mut complex_data: Vec<Complex<f64>> = data.column(elem)
                    .iter()
                    .map(|&x| Complex::new(x, 0.0))
                    .collect();
                
                let fft = self.fft_planner.plan_fft_forward(n_time);
                fft.process(&mut complex_data);
                
                for (i, value) in complex_data.iter_mut().enumerate() {
                    let freq = i as f64 * df;
                    if freq < f_min || freq > f_max {
                        *value = Complex::new(0.0, 0.0);
                    }
                }
                
                let ifft = self.fft_planner.plan_fft_inverse(n_time);
                ifft.process(&mut complex_data);
                
                for (i, &c) in complex_data.iter().enumerate() {
                    filtered[[i, elem]] = c.re / n_time as f64;
                }
            }
        } else {
            filtered.assign(data);
        }
        
        Ok(filtered)
    }
}

/// Iterative reconstruction using conjugate gradient method
pub struct IterativeRecon {
    /// Maximum number of iterations
    pub max_iterations: usize,
    
    /// Convergence tolerance
    pub tolerance: f64,
    
    /// Regularization parameter
    pub regularization: f64,
}

impl IterativeRecon {
    /// Create new iterative reconstruction instance
    pub fn new(max_iterations: usize, tolerance: f64, regularization: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            regularization,
        }
    }
    
    /// Perform iterative reconstruction using conjugate gradient
    pub fn reconstruct(&self, sensor_data: &Array2<f64>, forward_operator: impl Fn(&Array3<f64>) -> Array2<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
        let mut image = grid.zeros_array();
        let mut residual = sensor_data.clone();
        let mut direction = grid.zeros_array();
        let mut residual_norm_sq = residual.iter().map(|x| x * x).sum::<f64>();
        
        for iter in 0..self.max_iterations {
            // Compute gradient
            let gradient = self.compute_gradient(&residual, &forward_operator, grid)?;
            
            // Update search direction (conjugate gradient)
            if iter == 0 {
                direction = gradient.clone();
            } else {
                let beta = residual_norm_sq / residual_norm_sq;
                Zip::from(&mut direction)
                    .and(&gradient)
                    .for_each(|d, &g| *d = g + beta * *d);
            }
            
            // Line search
            let forward_direction = forward_operator(&direction);
            let alpha = residual_norm_sq / forward_direction.iter().map(|x| x * x).sum::<f64>();
            
            // Update image and residual
            Zip::from(&mut image)
                .and(&direction)
                .for_each(|i, &d| *i += alpha * d);
            
            Zip::from(&mut residual)
                .and(&forward_direction)
                .for_each(|r, &fd| *r -= alpha * fd);
            
            // Check convergence
            let new_residual_norm_sq = residual.iter().map(|x| x * x).sum::<f64>();
            if (new_residual_norm_sq / sensor_data.len() as f64).sqrt() < self.tolerance {
                break;
            }
            
            residual_norm_sq = new_residual_norm_sq;
        }
        
        Ok(image)
    }
    
    /// Compute gradient using adjoint operator
    fn compute_gradient(&self, residual: &Array2<f64>, forward_operator: &impl Fn(&Array3<f64>) -> Array2<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
        // This is a placeholder for the adjoint operator
        // In practice, this would be the transpose of the forward operator
        Ok(grid.zeros_array())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_linear_reconstruction() -> KwaversResult<()> {
        // Create test configuration
        let n_elements = 64;
        let mut sensor_positions = Array2::zeros((n_elements, 3));
        for i in 0..n_elements {
            sensor_positions[[i, 0]] = i as f64 * 0.001; // 1mm spacing
        }
        
        let config = LineReconConfig {
            sensor_positions,
            speed_of_sound: 1500.0,
            sampling_frequency: 20e6,
            interpolation: InterpolationMethod::Linear,
            apply_filter: false,
            filter_range: None,
            envelope_detection: false,
            log_compression: false,
            dynamic_range: 60.0,
        };
        
        let mut recon = LineRecon::new(config);
        
        // Create test data
        let n_time = 1000;
        let sensor_data = Array2::zeros((n_time, n_elements));
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001);
        
        // Perform reconstruction
        let image = recon.reconstruct(&sensor_data, &grid)?;
        assert_eq!(image.dim(), (64, 64, 64));
        
        Ok(())
    }
    
    #[test]
    fn test_planar_reconstruction() -> KwaversResult<()> {
        // Create test configuration for 2D planar array
        let n_x = 32;
        let n_y = 32;
        let n_elements = n_x * n_y;
        let mut sensor_positions = Array2::zeros((n_elements, 3));
        
        for iy in 0..n_y {
            for ix in 0..n_x {
                let idx = iy * n_x + ix;
                sensor_positions[[idx, 0]] = ix as f64 * 0.001;
                sensor_positions[[idx, 1]] = iy as f64 * 0.001;
                sensor_positions[[idx, 2]] = 0.0; // Planar array at z=0
            }
        }
        
        let config = PlaneReconConfig {
            sensor_positions,
            speed_of_sound: 1500.0,
            sampling_frequency: 20e6,
            interpolation: InterpolationMethod::Linear,
            apply_filter: false,
            filter_range: None,
            directivity_correction: true,
            element_size: Some(0.0005),
        };
        
        let mut recon = PlaneRecon::new(config);
        
        // Create test data
        let n_time = 1000;
        let sensor_data = Array2::zeros((n_time, n_elements));
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001);
        
        // Perform reconstruction
        let image = recon.reconstruct(&sensor_data, &grid)?;
        assert_eq!(image.dim(), (64, 64, 64));
        
        Ok(())
    }
    
    #[test]
    fn test_interpolation_methods() {
        // Test linear interpolation
        let config_linear = LineReconConfig {
            sensor_positions: Array2::zeros((1, 3)),
            speed_of_sound: 1500.0,
            sampling_frequency: 20e6,
            interpolation: InterpolationMethod::Linear,
            apply_filter: false,
            filter_range: None,
            envelope_detection: false,
            log_compression: false,
            dynamic_range: 60.0,
        };
        
        let recon_linear = LineRecon::new(config_linear);
        
        // Test data
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let view = data.view();
        
        // Test linear interpolation
        let val = recon_linear.interpolate(&view, 1.5);
        assert_relative_eq!(val, 2.5, epsilon = 1e-10);
        
        // Test boundary
        let val = recon_linear.interpolate(&view, 0.0);
        assert_relative_eq!(val, 1.0, epsilon = 1e-10);
        
        // Test cubic interpolation with edge cases
        let config_cubic = LineReconConfig {
            sensor_positions: Array2::zeros((1, 3)),
            speed_of_sound: 1500.0,
            sampling_frequency: 20e6,
            interpolation: InterpolationMethod::Cubic,
            apply_filter: false,
            filter_range: None,
            envelope_detection: false,
            log_compression: false,
            dynamic_range: 60.0,
        };
        
        let recon_cubic = LineRecon::new(config_cubic);
        
        // Test cubic at boundaries (should fall back to linear)
        let val = recon_cubic.interpolate(&view, 0.5); // Near start, should use linear fallback
        assert!(val > 0.0 && val < 5.0); // Reasonable value
        
        let val = recon_cubic.interpolate(&view, 4.5); // Near end, should use linear fallback
        assert!(val > 0.0 && val < 6.0); // Reasonable value
        
        // Test cubic in the middle (should use cubic)
        let val = recon_cubic.interpolate(&view, 2.5);
        assert!(val > 2.0 && val < 4.0); // Should be around 3.5
        
        // Test out of bounds
        let val = recon_cubic.interpolate(&view, -1.0);
        assert_eq!(val, 0.0);
        
        let val = recon_cubic.interpolate(&view, 10.0);
        assert_eq!(val, 0.0);
    }
}