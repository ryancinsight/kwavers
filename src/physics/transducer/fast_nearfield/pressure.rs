//! Pressure Field Calculation using FNM
//!
//! Implements efficient pressure field computation using basis function decomposition
//! and FFT-based k-space convolution for O(n log n) complexity.
//!
//! ## References
//!
//! - McGough (2004): O(n) pressure calculation algorithm
//! - Kelly & McGough (2006): Transient field extension
//! - Zeng & McGough (2008): Angular spectrum approach evaluation

use super::basis::BasisFunctions;
use super::FnmConfiguration;
use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array2, Array3};
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Pressure field calculator using FNM
pub struct PressureFieldCalculator {
    /// Sound speed (m/s)
    sound_speed: f64,
    /// Medium density (kg/m³)
    density: f64,
    /// Configuration
    config: FnmConfiguration,
    /// FFT planner for k-space convolution
    fft_planner: FftPlanner<f64>,
}

// Manual Debug implementation since FftPlanner doesn't implement Debug
impl std::fmt::Debug for PressureFieldCalculator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PressureFieldCalculator")
            .field("sound_speed", &self.sound_speed)
            .field("density", &self.density)
            .field("config", &self.config)
            .field("fft_planner", &"<FftPlanner>")
            .finish()
    }
}

impl PressureFieldCalculator {
    /// Create new pressure field calculator
    ///
    /// # Arguments
    ///
    /// * `config` - FNM configuration
    pub fn new(config: &FnmConfiguration) -> KwaversResult<Self> {
        Ok(Self {
            sound_speed: 1500.0,  // Default water sound speed
            density: 1000.0,       // Default water density
            config: config.clone(),
            fft_planner: FftPlanner::new(),
        })
    }

    /// Compute pressure field with FFT-based k-space convolution
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `frequency` - Ultrasound frequency (Hz)
    /// * `basis` - Precomputed basis functions
    ///
    /// # Returns
    ///
    /// Complex pressure field (Pa)
    ///
    /// # Algorithm
    ///
    /// 1. Decompose field using basis functions
    /// 2. Transform to k-space using FFT (O(n log n))
    /// 3. Apply Green's function convolution
    /// 4. Inverse FFT back to spatial domain
    ///
    /// Complexity: O(n log n) with FFT-based convolution
    ///
    /// # References
    ///
    /// Zeng & McGough (2008): Angular spectrum method with FFT acceleration
    pub fn compute_pressure_fft(
        &mut self,
        grid: &Grid,
        frequency: f64,
        basis: &BasisFunctions,
    ) -> KwaversResult<Array3<Complex<f64>>> {
        let (nx, ny, nz) = grid.dimensions();
        
        // Wave number
        let k = 2.0 * PI * frequency / self.sound_speed;
        
        // Compute k-space grid
        let _kx_max = PI / grid.dx;
        let _ky_max = PI / grid.dy;
        
        // Initialize result
        let mut pressure = Array3::zeros((nx, ny, nz));
        
        // Process each z-plane using angular spectrum method
        for z_idx in 0..nz {
            let z = z_idx as f64 * grid.dz;
            
            // Create source plane at z=0 (simplified for foundation)
            let mut source_plane = Array2::<Complex<f64>>::zeros((nx, ny));
            
            // Apply basis modulation to source
            for i in 0..nx {
                for j in 0..ny {
                    let x = i as f64 * grid.dx - (nx as f64 * grid.dx) / 2.0;
                    let y = j as f64 * grid.dy - (ny as f64 * grid.dy) / 2.0;
                    let r = (x * x + y * y).sqrt();
                    
                    // Basis-modulated source
                    let basis_weight = if r > self.config.singularity_tolerance {
                        basis.evaluate(0, (x / (r + grid.dx)).clamp(-1.0, 1.0))
                    } else {
                        1.0
                    };
                    
                    source_plane[[i, j]] = Complex::new(basis_weight, 0.0);
                }
            }
            
            // Propagate to current z using angular spectrum
            let propagated = self.angular_spectrum_propagation(
                &source_plane,
                z,
                k,
                grid.dx,
                grid.dy,
            )?;
            
            // Store in 3D pressure field
            for i in 0..nx {
                for j in 0..ny {
                    pressure[[i, j, z_idx]] = propagated[[i, j]];
                }
            }
        }
        
        Ok(pressure)
    }

    /// Angular spectrum propagation using FFT
    ///
    /// Propagates a 2D source plane to distance z using the angular spectrum method
    ///
    /// # Arguments
    ///
    /// * `source` - Source plane pressure distribution
    /// * `z` - Propagation distance (m)
    /// * `k` - Wave number (rad/m)
    /// * `dx` - X grid spacing (m)
    /// * `dy` - Y grid spacing (m)
    ///
    /// # References
    ///
    /// Zeng & McGough (2008): Accurate and efficient angular spectrum implementation
    fn angular_spectrum_propagation(
        &mut self,
        source: &Array2<Complex<f64>>,
        z: f64,
        k: f64,
        dx: f64,
        dy: f64,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        let (nx, ny) = source.dim();
        
        // Prepare for FFT
        let mut fft_buffer: Vec<Complex<f64>> = source.iter().copied().collect();
        
        // Create 2D FFT (apply 1D FFT along each dimension)
        let fft_x = self.fft_planner.plan_fft_forward(nx);
        let fft_y = self.fft_planner.plan_fft_forward(ny);
        
        // FFT along x-dimension
        for j in 0..ny {
            let start = j * nx;
            let end = start + nx;
            fft_x.process(&mut fft_buffer[start..end]);
        }
        
        // Transpose for y-dimension FFT
        let mut transposed = vec![Complex::new(0.0, 0.0); nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                transposed[i * ny + j] = fft_buffer[j * nx + i];
            }
        }
        
        // FFT along y-dimension
        for i in 0..nx {
            let start = i * ny;
            let end = start + ny;
            fft_y.process(&mut transposed[start..end]);
        }
        
        // Apply propagation in k-space
        let kx_step = 2.0 * PI / (nx as f64 * dx);
        let ky_step = 2.0 * PI / (ny as f64 * dy);
        
        for i in 0..nx {
            for j in 0..ny {
                let kx = if i < nx / 2 {
                    i as f64 * kx_step
                } else {
                    (i as f64 - nx as f64) * kx_step
                };
                
                let ky = if j < ny / 2 {
                    j as f64 * ky_step
                } else {
                    (j as f64 - ny as f64) * ky_step
                };
                
                // Angular spectrum transfer function: H(kx, ky) = exp(i * kz * z)
                let k_perp_sq = kx * kx + ky * ky;
                let kz = if k_perp_sq < k * k {
                    Complex::new((k * k - k_perp_sq).sqrt(), 0.0)
                } else {
                    // Evanescent wave: use imaginary kz
                    Complex::new(0.0, -(k_perp_sq - k * k).sqrt())
                };
                
                let transfer = Complex::new(0.0, kz.re * z).exp() * Complex::new(kz.im * z, 0.0).exp();
                
                transposed[i * ny + j] *= transfer;
            }
        }
        
        // Inverse FFT
        let ifft_y = self.fft_planner.plan_fft_inverse(ny);
        let ifft_x = self.fft_planner.plan_fft_inverse(nx);
        
        // IFFT along y-dimension
        for i in 0..nx {
            let start = i * ny;
            let end = start + ny;
            ifft_y.process(&mut transposed[start..end]);
        }
        
        // Transpose back
        for i in 0..nx {
            for j in 0..ny {
                fft_buffer[j * nx + i] = transposed[i * ny + j] / (ny as f64);
            }
        }
        
        // IFFT along x-dimension
        for j in 0..ny {
            let start = j * nx;
            let end = start + nx;
            ifft_x.process(&mut fft_buffer[start..end]);
        }
        
        // Normalize and convert to Array2
        let mut result = Array2::zeros((nx, ny));
        for i in 0..nx {
            for j in 0..ny {
                result[[i, j]] = fft_buffer[j * nx + i] / (nx as f64);
            }
        }
        
        Ok(result)
    }

    /// Compute pressure field with O(n) complexity
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `frequency` - Ultrasound frequency (Hz)
    /// * `basis` - Precomputed basis functions
    ///
    /// # Returns
    ///
    /// Complex pressure field (Pa)
    ///
    /// # Algorithm
    ///
    /// 1. Decompose field calculation using basis functions
    /// 2. Compute basis contributions efficiently
    /// 3. Combine using linear superposition
    ///
    /// Complexity: O(n) where n = number of grid points
    pub fn compute_pressure(
        &self,
        grid: &Grid,
        frequency: f64,
        basis: &BasisFunctions,
    ) -> KwaversResult<Array3<Complex<f64>>> {
        let (nx, ny, nz) = grid.dimensions();
        let mut pressure = Array3::zeros((nx, ny, nz));

        // Wave number
        let k = 2.0 * PI * frequency / self.sound_speed;

        // Reference amplitude (simplified)
        let amplitude = self.density * self.sound_speed; // Characteristic acoustic impedance

        // Compute pressure using basis function decomposition
        // Full implementation would use FFT-based convolution for O(n log n) complexity
        // This simplified version demonstrates the structure
        
        for i in 0..nx {
            for j in 0..ny {
                for kk in 0..nz {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, kk);
                    
                    // Distance from origin (simplified focal point)
                    let r = (x * x + y * y + z * z).sqrt();
                    
                    if r < self.config.singularity_tolerance {
                        pressure[[i, j, kk]] = Complex::new(amplitude, 0.0);
                        continue;
                    }

                    // Simplified spherical wave with basis modulation
                    // Full implementation: sum over basis contributions with proper Green's function
                    let phase = k * r;
                    let magnitude = amplitude / r;
                    
                    // Modulate with basis function (simplified)
                    let basis_weight = if basis.count() > 0 {
                        basis.evaluate(0, (x / r).clamp(-1.0, 1.0))
                    } else {
                        1.0
                    };
                    
                    pressure[[i, j, kk]] = Complex::new(
                        magnitude * phase.cos() * basis_weight,
                        magnitude * phase.sin() * basis_weight,
                    );
                }
            }
        }

        Ok(pressure)
    }

    /// Compute spatial impulse response
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `basis` - Precomputed basis functions
    ///
    /// # Returns
    ///
    /// Real-valued spatial impulse response
    pub fn compute_sir(
        &self,
        grid: &Grid,
        basis: &BasisFunctions,
    ) -> KwaversResult<Array3<f64>> {
        // Use default frequency for SIR calculation
        let frequency = self.config.frequency;
        let pressure = self.compute_pressure(grid, frequency, basis)?;
        
        // Convert to real-valued SIR (magnitude)
        let (nx, ny, nz) = pressure.dim();
        let mut sir = Array3::zeros((nx, ny, nz));
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    sir[[i, j, k]] = pressure[[i, j, k]].norm();
                }
            }
        }
        
        Ok(sir)
    }

    /// Set sound speed
    pub fn set_sound_speed(&mut self, c: f64) {
        self.sound_speed = c;
    }

    /// Set medium density
    pub fn set_density(&mut self, rho: f64) {
        self.density = rho;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_creation() {
        let config = FnmConfiguration::default();
        let result = PressureFieldCalculator::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pressure_computation() {
        let config = FnmConfiguration::default();
        let calculator = PressureFieldCalculator::new(&config).unwrap();
        let basis = BasisFunctions::new(16).unwrap();
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();

        let result = calculator.compute_pressure(&grid, 5.0e6, &basis);
        assert!(result.is_ok());

        let pressure = result.unwrap();
        assert_eq!(pressure.dim(), (20, 20, 20));
        
        // Check that some pressure values are non-zero
        let max_magnitude = pressure.iter()
            .map(|c| c.norm())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_magnitude > 0.0, "Pressure field should have non-zero values");
    }

    #[test]
    fn test_sir_computation() {
        let config = FnmConfiguration::default();
        let calculator = PressureFieldCalculator::new(&config).unwrap();
        let basis = BasisFunctions::new(16).unwrap();
        let grid = Grid::new(15, 15, 15, 0.001, 0.001, 0.001).unwrap();

        let result = calculator.compute_sir(&grid, &basis);
        assert!(result.is_ok());

        let sir = result.unwrap();
        assert_eq!(sir.dim(), (15, 15, 15));
        
        // SIR should be non-negative
        assert!(sir.iter().all(|&x| x >= 0.0), "SIR values should be non-negative");
    }
}
