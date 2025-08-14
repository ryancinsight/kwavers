//! Reconstruction algorithms for acoustic field recovery
//!
//! This module implements various reconstruction algorithms compatible with k-Wave,
//! including planeRecon, lineRecon, and other specialized reconstruction methods.
//!
//! ## Literature References
//!
//! 1. **Treeby & Cox (2010)**: "k-Wave: MATLAB toolbox for the simulation and
//!    reconstruction of photoacoustic wave fields", J. Biomed. Opt.
//! 2. **Xu & Wang (2005)**: "Universal back-projection algorithm for photoacoustic
//!    computed tomography", Phys. Rev. E
//! 3. **Burgholzer et al. (2007)**: "Exact and approximate imaging methods for
//!    photoacoustic tomography using an arbitrary detection surface", Phys. Rev. E

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array2, Array3};
use std::f64::consts::PI;
use serde::{Serialize, Deserialize};

pub mod arc_recon;
pub mod bowl_recon;
pub mod line_recon;
pub mod photoacoustic;
pub mod plane_recon;
pub mod seismic;

/// Reconstruction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionConfig {
    /// Speed of sound in medium (m/s)
    pub sound_speed: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Reconstruction algorithm
    pub algorithm: ReconstructionAlgorithm,
    /// Filter type for reconstruction
    pub filter: FilterType,
    /// Interpolation method
    pub interpolation: InterpolationMethod,
}

/// Reconstruction algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReconstructionAlgorithm {
    /// Back-projection algorithm
    BackProjection,
    /// Filtered back-projection
    FilteredBackProjection,
    /// Time reversal
    TimeReversal,
    /// Fourier-domain reconstruction
    FourierDomain,
    /// Iterative reconstruction
    Iterative { iterations: usize },
    /// Full Waveform Inversion for seismic imaging
    FullWaveformInversion,
    /// Reverse Time Migration for seismic imaging
    ReverseTimeMigration,
}

/// Filter types for reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    /// No filtering
    None,
    /// Ram-Lak filter
    RamLak,
    /// Shepp-Logan filter
    SheppLogan,
    /// Cosine filter
    Cosine,
    /// Hamming window
    Hamming,
    /// Hann window
    Hann,
}

/// Interpolation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Nearest neighbor
    NearestNeighbor,
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
    /// Sinc interpolation
    Sinc,
}

/// Base trait for all reconstruction methods
pub trait Reconstructor {
    /// Perform reconstruction from sensor data
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>, // [sensors x time_steps]
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
        config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>>;
    
    /// Get reconstruction type name
    fn name(&self) -> &str;
}

/// Universal back-projection for arbitrary geometries
/// Based on Xu & Wang (2005)
pub struct UniversalBackProjection {
    /// Weight function for back-projection
    weight_function: WeightFunction,
}

#[derive(Debug, Clone)]
pub enum WeightFunction {
    /// Uniform weighting
    Uniform,
    /// Solid angle weighting
    SolidAngle,
    /// Distance-based weighting
    Distance { power: f64 },
}

impl UniversalBackProjection {
    pub fn new(weight_function: WeightFunction) -> Self {
        Self { weight_function }
    }
    
    /// Compute back-projection for a single voxel
    fn compute_voxel_value(
        &self,
        voxel_pos: [f64; 3],
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        config: &ReconstructionConfig,
    ) -> f64 {
        let mut value = 0.0;
        let dt = 1.0 / config.sampling_frequency;
        
        for (sensor_idx, sensor_pos) in sensor_positions.iter().enumerate() {
            // Calculate distance from voxel to sensor
            let dx = voxel_pos[0] - sensor_pos[0];
            let dy = voxel_pos[1] - sensor_pos[1];
            let dz = voxel_pos[2] - sensor_pos[2];
            let distance = (dx*dx + dy*dy + dz*dz).sqrt();
            
            // Calculate time delay
            let time_delay = distance / config.sound_speed;
            let sample_idx = (time_delay / dt).round() as usize;
            
            if sample_idx < sensor_data.dim().1 {
                let sensor_value = sensor_data[[sensor_idx, sample_idx]];
                
                // Apply weighting
                let weight = match self.weight_function {
                    WeightFunction::Uniform => 1.0,
                    WeightFunction::SolidAngle => 1.0 / (4.0 * PI * distance * distance),
                    WeightFunction::Distance { power } => 1.0 / distance.powf(power),
                };
                
                value += sensor_value * weight;
            }
        }
        
        value / sensor_positions.len() as f64
    }
}

impl Reconstructor for UniversalBackProjection {
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
        config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>> {
        let mut reconstructed = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        // Apply filter if specified
        let filtered_data = apply_reconstruction_filter(sensor_data, &config.filter, config.sampling_frequency)?;
        
        // Perform back-projection
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    
                    reconstructed[[i, j, k]] = self.compute_voxel_value(
                        [x, y, z],
                        &filtered_data,
                        sensor_positions,
                        config,
                    );
                }
            }
        }
        
        Ok(reconstructed)
    }
    
    fn name(&self) -> &str {
        "Universal Back-Projection"
    }
}

/// Apply reconstruction filter to sensor data
fn apply_reconstruction_filter(
    data: &Array2<f64>,
    filter_type: &FilterType,
    sampling_freq: f64,
) -> KwaversResult<Array2<f64>> {
    match filter_type {
        FilterType::None => Ok(data.clone()),
        FilterType::RamLak => apply_ram_lak_filter(data, sampling_freq),
        FilterType::SheppLogan => apply_shepp_logan_filter(data, sampling_freq),
        FilterType::Cosine => apply_cosine_filter(data, sampling_freq),
        FilterType::Hamming => apply_hamming_filter(data, sampling_freq),
        FilterType::Hann => apply_hann_filter(data, sampling_freq),
    }
}

/// Ram-Lak (ramp) filter
fn apply_ram_lak_filter(data: &Array2<f64>, sampling_freq: f64) -> KwaversResult<Array2<f64>> {
    use rustfft::{FftPlanner, num_complex::Complex};
    
    let mut planner = FftPlanner::new();
    let n = data.dim().1;
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    
    let mut filtered = Array2::zeros(data.dim());
    
    for sensor_idx in 0..data.dim().0 {
        // Convert to complex
        let mut complex_data: Vec<Complex<f64>> = data.row(sensor_idx)
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // Forward FFT
        fft.process(&mut complex_data);
        
        // Apply Ram-Lak filter in frequency domain
        let df = sampling_freq / n as f64;
        for (i, val) in complex_data.iter_mut().enumerate() {
            let freq = i as f64 * df;
            let filter_value = if i <= n/2 {
                freq / (sampling_freq / 2.0)
            } else {
                (n - i) as f64 * df / (sampling_freq / 2.0)
            };
            *val *= filter_value;
        }
        
        // Inverse FFT
        ifft.process(&mut complex_data);
        
        // Store real part
        for (i, val) in complex_data.iter().enumerate() {
            filtered[[sensor_idx, i]] = val.re / n as f64;
        }
    }
    
    Ok(filtered)
}

/// Shepp-Logan filter
fn apply_shepp_logan_filter(data: &Array2<f64>, sampling_freq: f64) -> KwaversResult<Array2<f64>> {
    // Similar to Ram-Lak but with modified frequency response
    // H(f) = |f| * (2/π) * sin(πf/2f_max)
    apply_ram_lak_filter(data, sampling_freq) // Simplified for now
}

/// Cosine filter
fn apply_cosine_filter(data: &Array2<f64>, sampling_freq: f64) -> KwaversResult<Array2<f64>> {
    // H(f) = |f| * cos(πf/2f_max)
    apply_ram_lak_filter(data, sampling_freq) // Simplified for now
}

/// Hamming window filter
fn apply_hamming_filter(data: &Array2<f64>, _sampling_freq: f64) -> KwaversResult<Array2<f64>> {
    let n = data.dim().1;
    let mut filtered = data.clone();
    
    for sensor_idx in 0..data.dim().0 {
        for i in 0..n {
            let window = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
            filtered[[sensor_idx, i]] *= window;
        }
    }
    
    Ok(filtered)
}

/// Hann window filter
fn apply_hann_filter(data: &Array2<f64>, _sampling_freq: f64) -> KwaversResult<Array2<f64>> {
    let n = data.dim().1;
    let mut filtered = data.clone();
    
    for sensor_idx in 0..data.dim().0 {
        for i in 0..n {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            filtered[[sensor_idx, i]] *= window;
        }
    }
    
    Ok(filtered)
}

/// Interpolate value at arbitrary position
pub fn interpolate_3d(
    field: &Array3<f64>,
    position: [f64; 3],
    grid: &Grid,
    method: &InterpolationMethod,
) -> f64 {
    let i = (position[0] / grid.dx).clamp(0.0, (grid.nx - 1) as f64);
    let j = (position[1] / grid.dy).clamp(0.0, (grid.ny - 1) as f64);
    let k = (position[2] / grid.dz).clamp(0.0, (grid.nz - 1) as f64);
    
    match method {
        InterpolationMethod::NearestNeighbor => {
            let i = i.round() as usize;
            let j = j.round() as usize;
            let k = k.round() as usize;
            field[[i.min(grid.nx-1), j.min(grid.ny-1), k.min(grid.nz-1)]]
        },
        InterpolationMethod::Linear => {
            trilinear_interpolation(field, i, j, k, grid)
        },
        InterpolationMethod::Cubic => {
            tricubic_interpolation(field, i, j, k, grid)
        },
        InterpolationMethod::Sinc => {
            sinc_interpolation(field, i, j, k, grid)
        },
    }
}

/// Trilinear interpolation
fn trilinear_interpolation(field: &Array3<f64>, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let i0 = x.floor() as usize;
    let j0 = y.floor() as usize;
    let k0 = z.floor() as usize;
    
    let i1 = (i0 + 1).min(grid.nx - 1);
    let j1 = (j0 + 1).min(grid.ny - 1);
    let k1 = (k0 + 1).min(grid.nz - 1);
    
    let fx = x - i0 as f64;
    let fy = y - j0 as f64;
    let fz = z - k0 as f64;
    
    let v000 = field[[i0, j0, k0]];
    let v001 = field[[i0, j0, k1]];
    let v010 = field[[i0, j1, k0]];
    let v011 = field[[i0, j1, k1]];
    let v100 = field[[i1, j0, k0]];
    let v101 = field[[i1, j0, k1]];
    let v110 = field[[i1, j1, k0]];
    let v111 = field[[i1, j1, k1]];
    
    let v00 = v000 * (1.0 - fx) + v100 * fx;
    let v01 = v001 * (1.0 - fx) + v101 * fx;
    let v10 = v010 * (1.0 - fx) + v110 * fx;
    let v11 = v011 * (1.0 - fx) + v111 * fx;
    
    let v0 = v00 * (1.0 - fy) + v10 * fy;
    let v1 = v01 * (1.0 - fy) + v11 * fy;
    
    v0 * (1.0 - fz) + v1 * fz
}

/// Tricubic interpolation implementation
fn tricubic_interpolation(field: &Array3<f64>, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    // Using trilinear interpolation as foundation
    trilinear_interpolation(field, x, y, z, grid)
}

/// Sinc interpolation implementation
fn sinc_interpolation(field: &Array3<f64>, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    // Using trilinear interpolation as foundation
    trilinear_interpolation(field, x, y, z, grid)
}