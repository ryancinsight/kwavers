// src/fft/fft3d.rs
use crate::fft::fft_core::{precompute_twiddles, reverse_bits, FftDirection, is_power_of_two, next_power_of_two_usize, log2_ceil};
use crate::grid::Grid;
use ndarray::{Array3, s};
use num_complex::Complex;
use rayon::prelude::*;
use log::{debug, trace};
use std::sync::Arc;

/// Optimized 3D FFT implementation with cache-friendly algorithms
#[derive(Debug, Clone)]
pub struct Fft3d {
    nx: usize,
    ny: usize,
    nz: usize,
    padded_nx: usize,
    padded_ny: usize,
    padded_nz: usize,
    twiddles_x: Arc<Vec<Complex<f64>>>,
    twiddles_y: Arc<Vec<Complex<f64>>>,
    twiddles_z: Arc<Vec<Complex<f64>>>,
    bit_reverse_indices_x: Arc<Vec<usize>>,
    bit_reverse_indices_y: Arc<Vec<usize>>,
    bit_reverse_indices_z: Arc<Vec<usize>>,
}

impl Fft3d {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let padded_nx = next_power_of_two_usize(nx);
        let padded_ny = next_power_of_two_usize(ny);
        let padded_nz = next_power_of_two_usize(nz);
        
        // Precompute twiddle factors for better performance
        let twiddles_x = Arc::new(precompute_twiddles(padded_nx, FftDirection::Forward));
        let twiddles_y = Arc::new(precompute_twiddles(padded_ny, FftDirection::Forward));
        let twiddles_z = Arc::new(precompute_twiddles(padded_nz, FftDirection::Forward));
        
        // Precompute bit-reverse indices for faster reordering
        let bit_reverse_indices_x = Arc::new(Self::precompute_bit_reverse_indices(padded_nx));
        let bit_reverse_indices_y = Arc::new(Self::precompute_bit_reverse_indices(padded_ny));
        let bit_reverse_indices_z = Arc::new(Self::precompute_bit_reverse_indices(padded_nz));
        
        debug!(
            "Initialized optimized Fft3d: padded {}x{}x{} (original: {}x{}x{})",
            padded_nx, padded_ny, padded_nz, nx, ny, nz
        );
        
        Self {
            nx,
            ny,
            nz,
            padded_nx,
            padded_ny,
            padded_nz,
            twiddles_x,
            twiddles_y,
            twiddles_z,
            bit_reverse_indices_x,
            bit_reverse_indices_y,
            bit_reverse_indices_z,
        }
    }
    
    /// Precompute bit-reverse indices for a given size
    fn precompute_bit_reverse_indices(n: usize) -> Vec<usize> {
        let log2_n = log2_ceil(n) as u32;
        (0..n).map(|i| reverse_bits(i as u32, log2_n) as usize).collect()
    }

    pub fn process(&mut self, field: &mut Array3<Complex<f64>>, grid: &Grid) {
        debug!(
            "Computing optimized 3D FFT: size {}x{}x{}",
            grid.nx, grid.ny, grid.nz
        );
        assert_eq!(
            field.dim(),
            (grid.nx, grid.ny, grid.nz),
            "Field dimensions must match grid"
        );

        // Create padded field if needed
        if (grid.nx != self.padded_nx) || (grid.ny != self.padded_ny) || (grid.nz != self.padded_nz) {
            trace!("Padding field for FFT from {}x{}x{} to {}x{}x{}", 
                   grid.nx, grid.ny, grid.nz, self.padded_nx, self.padded_ny, self.padded_nz);
            
            let mut padded = Array3::zeros((self.padded_nx, self.padded_ny, self.padded_nz));
            padded.slice_mut(s![0..grid.nx, 0..grid.ny, 0..grid.nz]).assign(field);
            
            // Perform in-place FFT on padded field
            self.fft_in_place(&mut padded);
            
            // Copy back to original field
            field.assign(&padded.slice(s![0..grid.nx, 0..grid.ny, 0..grid.nz]));
        } else {
            // If no padding needed, perform FFT directly on the input field
            self.fft_in_place(field);
        }
    }

    fn fft_in_place(&self, field: &mut Array3<Complex<f64>>) {
        let (nx, ny, nz) = (self.padded_nx, self.padded_ny, self.padded_nz);
        
        // Apply bit reversal permutation
        self.apply_bit_reversal(field);
        
        // Process each dimension using sequential approach instead of parallel
        // to avoid borrowing issues
        
        // FFT along x-axis
        for j in 0..ny {
            let twiddles = &self.twiddles_x;
            for k in 0..nz {
                // Create a temporary buffer for the x-slice
                let mut temp = vec![Complex::new(0.0, 0.0); nx];
                
                // Copy data to temp buffer
                for i in 0..nx {
                    temp[i] = field[[i, j, k]];
                }
                
                // Perform FFT on temp buffer
                self.butterfly_1d_optimized(&mut temp, twiddles, nx);
                
                // Copy back to field
                for i in 0..nx {
                    field[[i, j, k]] = temp[i];
                }
            }
        }
        
        // FFT along y-axis
        for i in 0..nx {
            let twiddles = &self.twiddles_y;
            for k in 0..nz {
                // Create a temporary buffer for the y-slice
                let mut temp = vec![Complex::new(0.0, 0.0); ny];
                
                // Copy data to temp buffer (y-axis is not contiguous in memory)
                for j in 0..ny {
                    temp[j] = field[[i, j, k]];
                }
                
                // Perform FFT on temp buffer
                self.butterfly_1d_optimized(&mut temp, twiddles, ny);
                
                // Copy back to field
                for j in 0..ny {
                    field[[i, j, k]] = temp[j];
                }
            }
        }
        
        // FFT along z-axis
        for i in 0..nx {
            let twiddles = &self.twiddles_z;
            for j in 0..ny {
                // Create a temporary buffer for the z-slice
                let mut temp = vec![Complex::new(0.0, 0.0); nz];
                
                // Copy data to temp buffer (z-axis is not contiguous in memory)
                for k in 0..nz {
                    temp[k] = field[[i, j, k]];
                }
                
                // Perform FFT on temp buffer
                self.butterfly_1d_optimized(&mut temp, twiddles, nz);
                
                // Copy back to field
                for k in 0..nz {
                    field[[i, j, k]] = temp[k];
                }
            }
        }
    }
    
    /// Apply bit reversal permutation to the 3D field
    fn apply_bit_reversal(&self, field: &mut Array3<Complex<f64>>) {
        let (nx, ny, nz) = (self.padded_nx, self.padded_ny, self.padded_nz);
        
        // Use precomputed bit-reverse indices
        let indices_x = &self.bit_reverse_indices_x;
        let indices_y = &self.bit_reverse_indices_y;
        let indices_z = &self.bit_reverse_indices_z;
        
        // Apply bit reversal in a cache-friendly way
        for i in 0..nx {
            let i_rev = indices_x[i];
            if i < i_rev {
                for j in 0..ny {
                    for k in 0..nz {
                        field.swap([i, j, k], [i_rev, j, k]);
                    }
                }
            }
        }
        
        for j in 0..ny {
            let j_rev = indices_y[j];
            if j < j_rev {
                for i in 0..nx {
                    for k in 0..nz {
                        field.swap([i, j, k], [i, j_rev, k]);
                    }
                }
            }
        }
        
        for k in 0..nz {
            let k_rev = indices_z[k];
            if k < k_rev {
                for i in 0..nx {
                    for j in 0..ny {
                        field.swap([i, j, k], [i, j, k_rev]);
                    }
                }
            }
        }
    }

    /// Optimized 1D butterfly FFT algorithm that works on slices for better cache locality
    #[inline]
    fn butterfly_1d_optimized(&self, data: &mut [Complex<f64>], twiddles: &[Complex<f64>], n: usize) {
        let mut len = 1;
        while len < n {
            let half_len = len;
            len *= 2;
            
            // Process each butterfly stage
            for k in (0..n).step_by(len) {
                // Use SIMD-friendly loop with fewer branches
                for j in 0..half_len {
                    let idx = k + j;
                    let idx2 = idx + half_len;
                    
                    // Avoid bounds checking in release mode
                    debug_assert!(idx < n && idx2 < n && j * (n / len) < twiddles.len());
                    
                    // Compute butterfly operation
                    let t = twiddles[j * (n / len)] * data[idx2];
                    let u = data[idx];
                    data[idx] = u + t;
                    data[idx2] = u - t;
                }
            }
        }
    }
}