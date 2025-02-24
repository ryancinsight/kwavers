// src/utils/mod.rs
use crate::grid::Grid;
use crate::fft::{Fft3d, Ifft3d};
use ndarray::{Array3, Array4, Axis};
use num_complex::Complex;
use log::debug;

pub fn fft_3d(fields: &Array4<f64>, field_idx: usize, grid: &Grid) -> Array3<Complex<f64>> {
    debug!("Computing custom 3D FFT for field {}", field_idx);
    let mut field_freq = fields
        .index_axis(Axis(0), field_idx)
        .mapv(|x| Complex::new(x, 0.0))
        .into_dimensionality::<ndarray::Ix3>()
        .unwrap();
    let mut fft = Fft3d::new(grid.nx, grid.ny, grid.nz);
    fft.process(&mut field_freq, grid);
    field_freq
}

pub fn ifft_3d(field: &Array3<Complex<f64>>, grid: &Grid) -> Array3<f64> {
    debug!("Computing custom 3D IFFT");
    let mut field_freq = field.clone();
    let mut ifft = Ifft3d::new(grid.nx, grid.ny, grid.nz);
    ifft.process(&mut field_freq, grid)
}

pub fn laplacian(fields: &Array4<f64>, field_idx: usize, grid: &Grid) -> Result<Array3<f64>, &'static str> {
    debug!("Computing Laplacian for field {}", field_idx);
    let mut field_fft = fft_3d(fields, field_idx, grid);
    let k2 = grid.k_squared();

    field_fft
        .indexed_iter_mut()
        .zip(k2.iter())
        .for_each(|(((i, j, k), f), &k2_val)| {
            *f *= Complex::new(-k2_val, 0.0);
        });

    Ok(ifft_3d(&field_fft, grid))
}

pub fn derivative(fields: &Array4<f64>, field_idx: usize, grid: &Grid, axis: usize) -> Result<Array3<f64>, &'static str> {
    if axis > 2 { return Err("Axis must be 0 (x), 1 (y), or 2 (z)"); }
    let mut field_fft = fft_3d(fields, field_idx, grid);
    let k = match axis {
        0 => grid.kx(),
        1 => grid.ky(),
        2 => grid.kz(),
        _ => unreachable!(),
    };

    field_fft
        .axis_iter_mut(Axis(axis))
        .enumerate()
        .for_each(|(idx, mut slice)| {
            slice.mapv_inplace(|c| c * Complex::new(0.0, k[idx]));
        });

    Ok(ifft_3d(&field_fft, grid))
}