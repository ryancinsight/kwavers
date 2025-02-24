// src/fft/fft3d.rs
use crate::fft::fft_core::{precompute_twiddles, reverse_bits, FftDirection};
use crate::grid::Grid;
use ndarray::{Array3, Axis, s};
use num_complex::Complex;
use rayon::prelude::*;
use log::debug;

#[derive(Debug, Clone)]
pub struct Fft3d {
    nx: usize,
    ny: usize,
    nz: usize,
    padded_nx: usize,
    padded_ny: usize,
    padded_nz: usize,
    twiddles_x: Vec<Complex<f64>>,
    twiddles_y: Vec<Complex<f64>>,
    twiddles_z: Vec<Complex<f64>>,
}

impl Fft3d {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let padded_nx = nx.next_power_of_two();
        let padded_ny = ny.next_power_of_two();
        let padded_nz = nz.next_power_of_two();
        let twiddles_x = precompute_twiddles(padded_nx, FftDirection::Forward);
        let twiddles_y = precompute_twiddles(padded_ny, FftDirection::Forward);
        let twiddles_z = precompute_twiddles(padded_nz, FftDirection::Forward);
        debug!(
            "Initialized custom Fft3d: padded {}x{}x{} (original: {}x{}x{})",
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
        }
    }

    pub fn process(&mut self, field: &mut Array3<Complex<f64>>, grid: &Grid) {
        debug!(
            "Computing custom 3D FFT: size {}x{}x{}",
            grid.nx, grid.ny, grid.nz
        );
        assert_eq!(
            field.dim(),
            (grid.nx, grid.ny, grid.nz),
            "Field dimensions must match grid"
        );

        // Pad field to power-of-two sizes
        let mut padded_field = Array3::zeros((self.padded_nx, self.padded_ny, self.padded_nz));
        padded_field
            .slice_mut(s![0..self.nx, 0..self.ny, 0..self.nz])
            .assign(field);

        // In-place 3D FFT
        self.fft_in_place(&mut padded_field);

        // Copy back to original field
        field.assign(&padded_field.slice(s![0..self.nx, 0..self.ny, 0..self.nz]));
    }

    fn fft_in_place(&self, field: &mut Array3<Complex<f64>>) {
        let (nx, ny, nz) = (self.padded_nx, self.padded_ny, self.padded_nz);

        // Bit reversal along all dimensions
        for i in 0..nx {
            let i_rev = reverse_bits(i as u32, (nx as f64).log2() as u32) as usize;
            for j in 0..ny {
                let j_rev = reverse_bits(j as u32, (ny as f64).log2() as u32) as usize;
                for k in 0..nz {
                    let k_rev = reverse_bits(k as u32, (nz as f64).log2() as u32) as usize;
                    if i < i_rev || (i == i_rev && j < j_rev) || (i == i_rev && j == j_rev && k < k_rev) {
                        field.swap([i, j, k], [i_rev, j_rev, k_rev]);
                    }
                }
            }
        }

        // FFT along x-axis
        for j in 0..ny {
            for k in 0..nz {
                self.butterfly_1d(&mut field.slice_mut(s![.., j, k]), &self.twiddles_x, nx);
            }
        }

        // FFT along y-axis
        for i in 0..nx {
            for k in 0..nz {
                self.butterfly_1d(&mut field.slice_mut(s![i, .., k]), &self.twiddles_y, ny);
            }
        }

        // FFT along z-axis
        for i in 0..nx {
            for j in 0..ny {
                self.butterfly_1d(&mut field.slice_mut(s![i, j, ..]), &self.twiddles_z, nz);
            }
        }
    }

    fn butterfly_1d(&self, data: &mut ndarray::ArrayViewMut1<Complex<f64>>, twiddles: &[Complex<f64>], n: usize) {
        let mut len = 1;
        while len < n {
            let half_len = len;
            len *= 2;
            for k in (0..n).step_by(len) {
                for j in 0..half_len {
                    let idx = k + j;
                    let t = twiddles[j * (n / len)] * data[idx + half_len];
                    let u = data[idx];
                    data[idx] = u + t;
                    data[idx + half_len] = u - t;
                }
            }
        }
    }
}