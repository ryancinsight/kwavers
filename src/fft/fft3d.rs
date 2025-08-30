// src/fft/fft3d.rs
use crate::fft::fft_core::{
    apply_bit_reversal_3d, butterfly_1d, log2_ceil, next_power_of_two_usize, precompute_twiddles,
    reverse_bits, FftDirection,
};
use crate::grid::Grid;
use log::debug;
use ndarray::Array3;
use num_complex::Complex;
use std::sync::Arc;

/// 3D FFT implementation with cache-friendly algorithms
#[derive(Debug, Clone))]
pub struct Fft3d {
    padded_nx: usize,
    padded_ny: usize,
    padded_nz: usize,
    twiddles_x: Arc<Vec<Complex<f64>>>,
    twiddles_y: Arc<Vec<Complex<f64>>>,
    twiddles_z: Arc<Vec<Complex<f64>>>,
    bit_reverse_indices_x: Arc<Vec<usize>>,
    bit_reverse_indices_y: Arc<Vec<usize>>,
    bit_reverse_indices_z: Arc<Vec<usize>>,
    // Reusable workspace buffers to avoid allocations
    workspace_x: Vec<Complex<f64>>,
    workspace_y: Vec<Complex<f64>>,
    workspace_z: Vec<Complex<f64>>,
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
            padded_nx,
            padded_ny,
            padded_nz,
            twiddles_x,
            twiddles_y,
            twiddles_z,
            bit_reverse_indices_x,
            bit_reverse_indices_y,
            bit_reverse_indices_z,
            workspace_x: vec![Complex::new(0.0, 0.0); padded_nx],
            workspace_y: vec![Complex::new(0.0, 0.0); padded_ny],
            workspace_z: vec![Complex::new(0.0, 0.0); padded_nz],
        }
    }

    /// Precompute bit-reverse indices for a given size
    fn precompute_bit_reverse_indices(n: usize) -> Vec<usize> {
        let log2_n = log2_ceil(n);
        (0..n)
            .map(|i| reverse_bits(i as u32, log2_n) as usize)
            .collect()
    }

    pub fn process(&mut self, field: &mut Array3<Complex<f64>>, grid: &Grid) {
        debug!(
            "Computing optimized 3D FFT: size {}x{}x{}",
            grid.nx, grid.ny, grid.nz
        );

        // Create padded field
        let mut padded = Array3::zeros((self.padded_nx, self.padded_ny, self.padded_nz));
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    padded[[i, j, k] = field[[i, j, k];
                }
            }
        }

        // Perform in-place FFT on padded field
        self.fft_in_place(&mut padded);

        // Copy back to original field
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    field[[i, j, k] = padded[[i, j, k];
                }
            }
        }
    }

    fn fft_in_place(&mut self, field: &mut Array3<Complex<f64>>) {
        let (nx, ny, nz) = (self.padded_nx, self.padded_ny, self.padded_nz);

        // Apply bit reversal permutation
        apply_bit_reversal_3d(
            field,
            &self.bit_reverse_indices_x,
            &self.bit_reverse_indices_y,
            &self.bit_reverse_indices_z,
        );

        // Process each dimension using sequential approach instead of parallel
        // to avoid borrowing issues

        // FFT along x-axis
        for j in 0..ny {
            let twiddles = &self.twiddles_x;
            for k in 0..nz {
                // Use reusable buffer for the x-slice
                let workspace = &mut self.workspace_x;

                // Copy data to workspace buffer
                for i in 0..nx {
                    workspace[i] = field[[i, j, k];
                }

                // Perform FFT on workspace buffer
                butterfly_1d(workspace, twiddles, nx);

                // Copy back to field
                for i in 0..nx {
                    field[[i, j, k] = workspace[i];
                }
            }
        }

        // FFT along y-axis
        for i in 0..nx {
            let twiddles = &self.twiddles_y;
            for k in 0..nz {
                // Use reusable buffer for the y-slice
                let workspace = &mut self.workspace_y;

                // Copy data to workspace buffer (y-axis is not contiguous in memory)
                for j in 0..ny {
                    workspace[j] = field[[i, j, k];
                }

                // Perform FFT on workspace buffer
                butterfly_1d(workspace, twiddles, ny);

                // Copy back to field
                for j in 0..ny {
                    field[[i, j, k] = workspace[j];
                }
            }
        }

        // FFT along z-axis
        for i in 0..nx {
            let twiddles = &self.twiddles_z;
            for j in 0..ny {
                // Use reusable buffer for the z-slice
                let workspace = &mut self.workspace_z;

                // Copy data to workspace buffer (z-axis is not contiguous in memory)
                for k in 0..nz {
                    workspace[k] = field[[i, j, k];
                }

                // Perform FFT on workspace buffer
                butterfly_1d(workspace, twiddles, nz);

                // Copy back to field
                for k in 0..nz {
                    field[[i, j, k] = workspace[k];
                }
            }
        }
    }
}
