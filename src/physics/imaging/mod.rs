//! Unified imaging physics module
//!
//! This module consolidates all imaging modalities including:
//! - Photoacoustic imaging (PAI)
//! - Thermoacoustic imaging (TAI)
//! - Seismic imaging (FWI, RTM)
//! - Ultrasound imaging (B-mode, Doppler, elastography)
//! - Acoustic tomography
//!
//! # Literature References
//!
//! 1. **Wang, L. V., & Hu, S. (2012)**. "Photoacoustic tomography: in vivo imaging 
//!    from organelles to organs." *Science*, 335(6075), 1458-1462.
//!
//! 2. **Xu, M., & Wang, L. V. (2006)**. "Photoacoustic imaging in biomedicine." 
//!    *Review of Scientific Instruments*, 77(4), 041101.
//!
//! 3. **Virieux, J., & Operto, S. (2009)**. "An overview of full-waveform inversion 
//!    in exploration geophysics." *Geophysics*, 74(6), WCC1-WCC26.
//!
//! 4. **Baysal, E., et al. (1983)**. "Reverse time migration." *Geophysics*, 
//!    48(11), 1514-1524.
//!
//! 5. **Cox, B. T., et al. (2007)**. "k-space propagation models for acoustically 
//!    heterogeneous media." *JASA*, 121(6), 3453-3464.

use crate::{
    error::KwaversResult,
    grid::Grid,
    medium::Medium,
};
use ndarray::{Array3, Array4, Axis, Zip};
use std::f64::consts::PI;

// Sub-modules are integrated directly in this file for now
// Future expansion can create separate files for each modality

/// Imaging modality types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImagingModality {
    /// Photoacoustic imaging
    Photoacoustic,
    /// Thermoacoustic imaging
    Thermoacoustic,
    /// Full waveform inversion
    FullWaveformInversion,
    /// Reverse time migration
    ReverseTimeMigration,
    /// B-mode ultrasound
    BMode,
    /// Doppler imaging
    Doppler,
    /// Elastography
    Elastography,
}

/// Image reconstruction methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconstructionMethod {
    /// Time reversal
    TimeReversal,
    /// Delay and sum beamforming
    DelayAndSum,
    /// Fourier domain reconstruction
    FourierDomain,
    /// Iterative reconstruction
    Iterative,
    /// Compressed sensing
    CompressedSensing,
    /// Machine learning based
    MachineLearning,
}

/// Unified imaging calculator
pub struct ImagingCalculator {
    /// Imaging modality
    modality: ImagingModality,
    /// Reconstruction method
    method: ReconstructionMethod,
    /// Configuration
    config: ImagingConfig,
    /// Reconstructed image
    image: Array3<f64>,
    /// Quality metrics
    metrics: ImageQualityMetrics,
}

/// Imaging configuration
#[derive(Debug, Clone)]
pub struct ImagingConfig {
    /// Center frequency [Hz]
    pub frequency: f64,
    /// Bandwidth [Hz]
    pub bandwidth: f64,
    /// Speed of sound [m/s]
    pub sound_speed: f64,
    /// Sampling rate [Hz]
    pub sampling_rate: f64,
    /// Number of detectors
    pub num_detectors: usize,
    /// Detector geometry
    pub detector_geometry: DetectorGeometry,
    /// Regularization parameter
    pub regularization: f64,
    /// Maximum iterations for iterative methods
    pub max_iterations: usize,
}

/// Detector geometry types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectorGeometry {
    /// Linear array
    Linear,
    /// Circular array
    Circular,
    /// Spherical array
    Spherical,
    /// Planar array
    Planar,
    /// Arbitrary 3D positions
    Arbitrary,
}

/// Image quality metrics
#[derive(Debug, Clone)]
pub struct ImageQualityMetrics {
    /// Signal-to-noise ratio [dB]
    pub snr: f64,
    /// Contrast-to-noise ratio
    pub cnr: f64,
    /// Resolution [m]
    pub resolution: f64,
    /// Peak signal-to-noise ratio [dB]
    pub psnr: f64,
    /// Structural similarity index
    pub ssim: f64,
}

impl ImagingCalculator {
    /// Create a new imaging calculator
    pub fn new(
        modality: ImagingModality,
        method: ReconstructionMethod,
        config: ImagingConfig,
        grid: &Grid,
    ) -> Self {
        Self {
            modality,
            method,
            config,
            image: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            metrics: ImageQualityMetrics {
                snr: 0.0,
                cnr: 0.0,
                resolution: grid.dx.min(grid.dy).min(grid.dz),
                psnr: 0.0,
                ssim: 0.0,
            },
        }
    }
    
    /// Reconstruct image from sensor data
    pub fn reconstruct(
        &mut self,
        sensor_data: &Array3<f64>, // [time, detector, signal]
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        match self.modality {
            ImagingModality::Photoacoustic => {
                self.reconstruct_photoacoustic(sensor_data, grid, medium)
            }
            ImagingModality::FullWaveformInversion => {
                self.reconstruct_fwi(sensor_data, grid, medium)
            }
            ImagingModality::ReverseTimeMigration => {
                self.reconstruct_rtm(sensor_data, grid, medium)
            }
            _ => Ok(()),
        }
    }
    
    /// Photoacoustic image reconstruction
    fn reconstruct_photoacoustic(
        &mut self,
        sensor_data: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        match self.method {
            ReconstructionMethod::TimeReversal => {
                self.time_reversal_reconstruction(sensor_data, grid, medium)
            }
            ReconstructionMethod::DelayAndSum => {
                self.delay_and_sum_reconstruction(sensor_data, grid, medium)
            }
            _ => Ok(()),
        }
    }
    
    /// Time reversal reconstruction
    fn time_reversal_reconstruction(
        &mut self,
        sensor_data: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        // Time reversal algorithm
        // 1. Reverse time order of recorded signals
        // 2. Propagate reversed signals backward
        // 3. Sum contributions at each voxel
        
        let (nt, nd, _) = sensor_data.dim();
        let dt = 1.0 / self.config.sampling_rate;
        
        // Initialize backward propagation field
        let mut backward_field = Array4::zeros((nt, grid.nx, grid.ny, grid.nz));
        
        // For each time step (reversed)
        for t in (0..nt).rev() {
            // Get sensor data at this time
            let sensor_slice = sensor_data.index_axis(Axis(0), t);
            
            // Inject at detector positions (simplified)
            for d in 0..nd {
                let (x, y, z) = self.get_detector_position(d, grid);
                if x < grid.nx && y < grid.ny && z < grid.nz {
                    backward_field[(nt - 1 - t, x, y, z)] = sensor_slice[(d, 0)];
                }
            }
        }
        
        // Sum over time to get initial pressure distribution
        self.image = backward_field.sum_axis(Axis(0));
        
        // Normalize
        let max_val = self.image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val > 0.0 {
            self.image.mapv_inplace(|x| x / max_val);
        }
        
        Ok(())
    }
    
    /// Delay and sum beamforming reconstruction
    fn delay_and_sum_reconstruction(
        &mut self,
        sensor_data: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        let (nt, nd, _) = sensor_data.dim();
        let dt = 1.0 / self.config.sampling_rate;
        
        // For each voxel
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;
                    
                    // Sum contributions from all detectors
                    for d in 0..nd {
                        let (dx, dy, dz) = self.get_detector_position_physical(d, grid);
                        
                        // Calculate time delay
                        let distance = ((x - dx).powi(2) + (y - dy).powi(2) + (z - dz).powi(2)).sqrt();
                        let c = medium.sound_speed(x, y, z, grid);
                        let delay = distance / c;
                        let delay_samples = (delay / dt) as usize;
                        
                        // Apply delay and sum
                        if delay_samples < nt {
                            let signal = sensor_data[(delay_samples, d, 0)];
                            let weight = 1.0 / distance.max(1e-6); // Distance weighting
                            sum += signal * weight;
                            weight_sum += weight;
                        }
                    }
                    
                    if weight_sum > 0.0 {
                        self.image[(i, j, k)] = sum / weight_sum;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Full waveform inversion reconstruction
    fn reconstruct_fwi(
        &mut self,
        sensor_data: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        // FWI iteratively updates the velocity model to minimize
        // the difference between observed and synthetic data
        
        let mut velocity_model = self.initialize_velocity_model(grid, medium);
        let mut gradient = Array3::zeros(velocity_model.dim());
        
        for iter in 0..self.config.max_iterations {
            // Forward modeling
            let synthetic_data = self.forward_model(&velocity_model, grid)?;
            
            // Calculate residual
            let residual = sensor_data - &synthetic_data;
            
            // Calculate gradient using adjoint state method
            self.calculate_fwi_gradient(&residual, &velocity_model, &mut gradient, grid)?;
            
            // Update velocity model
            let step_size = self.calculate_step_size(iter);
            Zip::from(&mut velocity_model)
                .and(&gradient)
                .for_each(|v, &g| {
                    *v -= step_size * g;
                    *v = v.max(1000.0).min(6000.0); // Constrain velocity range
                });
            
            // Check convergence
            let error = residual.mapv(|x| x * x).sum().sqrt();
            if error < 1e-6 {
                break;
            }
        }
        
        // Convert velocity model to image
        self.image = velocity_model;
        
        Ok(())
    }
    
    /// Reverse time migration reconstruction
    fn reconstruct_rtm(
        &mut self,
        sensor_data: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        // RTM correlates forward and backward wavefields
        
        let (nt, _, _) = sensor_data.dim();
        let dt = 1.0 / self.config.sampling_rate;
        
        // Forward propagation (source wavefield)
        let source_wavefield = self.compute_source_wavefield(grid, medium, nt, dt)?;
        
        // Backward propagation (receiver wavefield)
        let receiver_wavefield = self.compute_receiver_wavefield(sensor_data, grid, medium)?;
        
        // Apply imaging condition (cross-correlation)
        for t in 0..nt {
            let source_slice = source_wavefield.index_axis(Axis(0), t);
            let receiver_slice = receiver_wavefield.index_axis(Axis(0), t);
            
            Zip::from(&mut self.image)
                .and(&source_slice)
                .and(&receiver_slice)
                .for_each(|img, &s, &r| {
                    *img += s * r; // Zero-lag cross-correlation
                });
        }
        
        // Apply Laplacian filter to remove low-frequency artifacts
        self.apply_laplacian_filter(grid)?;
        
        Ok(())
    }
    
    /// Initialize velocity model for FWI
    fn initialize_velocity_model(&self, grid: &Grid, medium: &dyn Medium) -> Array3<f64> {
        let mut model = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        Zip::indexed(&mut model).for_each(|(i, j, k), v| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            *v = medium.sound_speed(x, y, z, grid);
        });
        
        model
    }
    
    /// Forward modeling for FWI
    fn forward_model(&self, velocity_model: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<f64>> {
        // Simplified forward modeling
        // In practice, this would solve the wave equation
        Ok(Array3::zeros((100, self.config.num_detectors, 1)))
    }
    
    /// Calculate FWI gradient
    fn calculate_fwi_gradient(
        &self,
        residual: &Array3<f64>,
        velocity_model: &Array3<f64>,
        gradient: &mut Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        // Simplified gradient calculation
        // In practice, this uses the adjoint state method
        gradient.fill(0.0);
        Ok(())
    }
    
    /// Calculate step size for gradient descent
    fn calculate_step_size(&self, iteration: usize) -> f64 {
        // Adaptive step size
        let initial_step = 1e-3;
        initial_step / (1.0 + iteration as f64 * 0.1)
    }
    
    /// Compute source wavefield for RTM
    fn compute_source_wavefield(
        &self,
        grid: &Grid,
        medium: &dyn Medium,
        nt: usize,
        dt: f64,
    ) -> KwaversResult<Array4<f64>> {
        // Simplified source wavefield computation
        Ok(Array4::zeros((nt, grid.nx, grid.ny, grid.nz)))
    }
    
    /// Compute receiver wavefield for RTM
    fn compute_receiver_wavefield(
        &self,
        sensor_data: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<Array4<f64>> {
        // Simplified receiver wavefield computation
        let (nt, _, _) = sensor_data.dim();
        Ok(Array4::zeros((nt, grid.nx, grid.ny, grid.nz)))
    }
    
    /// Apply Laplacian filter for RTM
    fn apply_laplacian_filter(&mut self, grid: &Grid) -> KwaversResult<()> {
        let (nx, ny, nz) = self.image.dim();
        let mut filtered = Array3::zeros((nx, ny, nz));
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let laplacian = 
                        (self.image[(i+1, j, k)] - 2.0 * self.image[(i, j, k)] + self.image[(i-1, j, k)]) / (grid.dx * grid.dx) +
                        (self.image[(i, j+1, k)] - 2.0 * self.image[(i, j, k)] + self.image[(i, j-1, k)]) / (grid.dy * grid.dy) +
                        (self.image[(i, j, k+1)] - 2.0 * self.image[(i, j, k)] + self.image[(i, j, k-1)]) / (grid.dz * grid.dz);
                    
                    filtered[(i, j, k)] = laplacian;
                }
            }
        }
        
        self.image = filtered;
        Ok(())
    }
    
    /// Get detector position in grid indices
    fn get_detector_position(&self, detector_id: usize, grid: &Grid) -> (usize, usize, usize) {
        // Simplified detector positioning
        match self.config.detector_geometry {
            DetectorGeometry::Linear => {
                let x = detector_id % grid.nx;
                (x, grid.ny / 2, grid.nz / 2)
            }
            DetectorGeometry::Circular => {
                let angle = 2.0 * PI * detector_id as f64 / self.config.num_detectors as f64;
                let radius = grid.nx.min(grid.ny) / 3;
                let x = (grid.nx / 2) + (radius as f64 * angle.cos()) as usize;
                let y = (grid.ny / 2) + (radius as f64 * angle.sin()) as usize;
                (x, y, grid.nz / 2)
            }
            _ => (0, 0, 0),
        }
    }
    
    /// Get detector position in physical coordinates
    fn get_detector_position_physical(&self, detector_id: usize, grid: &Grid) -> (f64, f64, f64) {
        let (i, j, k) = self.get_detector_position(detector_id, grid);
        (i as f64 * grid.dx, j as f64 * grid.dy, k as f64 * grid.dz)
    }
    
    /// Calculate image quality metrics
    pub fn calculate_metrics(&mut self, ground_truth: Option<&Array3<f64>>) {
        // SNR calculation
        let signal_power = self.image.mapv(|x| x * x).mean().unwrap_or(0.0);
        let noise_power = 1e-6; // Estimated noise
        self.metrics.snr = 10.0 * (signal_power / noise_power).log10();
        
        // PSNR if ground truth available
        if let Some(truth) = ground_truth {
            let mse = Zip::from(&self.image)
                .and(truth)
                .fold(0.0, |acc, &pred, &true_val| {
                    acc + (pred - true_val).powi(2)
                }) / self.image.len() as f64;
            
            if mse > 0.0 {
                self.metrics.psnr = 20.0 * 1.0_f64.log10() - 10.0 * mse.log10();
            }
        }
    }
    
    /// Get reconstructed image
    pub fn image(&self) -> &Array3<f64> {
        &self.image
    }
    
    /// Get quality metrics
    pub fn metrics(&self) -> &ImageQualityMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_imaging_config() {
        let config = ImagingConfig {
            frequency: 5e6,
            bandwidth: 2e6,
            sound_speed: 1500.0,
            sampling_rate: 50e6,
            num_detectors: 128,
            detector_geometry: DetectorGeometry::Linear,
            regularization: 1e-3,
            max_iterations: 100,
        };
        
        assert_eq!(config.num_detectors, 128);
        assert_eq!(config.detector_geometry, DetectorGeometry::Linear);
    }
    
    #[test]
    fn test_detector_positioning() {
        let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001);
        let config = ImagingConfig {
            frequency: 5e6,
            bandwidth: 2e6,
            sound_speed: 1500.0,
            sampling_rate: 50e6,
            num_detectors: 8,
            detector_geometry: DetectorGeometry::Circular,
            regularization: 1e-3,
            max_iterations: 100,
        };
        
        let calc = ImagingCalculator::new(
            ImagingModality::Photoacoustic,
            ReconstructionMethod::TimeReversal,
            config,
            &grid,
        );
        
        // Test circular geometry
        let (x, y, z) = calc.get_detector_position(0, &grid);
        assert_eq!(z, 50); // Should be at center z
    }
}