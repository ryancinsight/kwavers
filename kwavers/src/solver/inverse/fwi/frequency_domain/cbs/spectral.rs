//! Periodic spectral shifted Green operator for CBS.
//!
//! # Contract
//!
//! On a uniform periodic grid, the shifted reference Helmholtz operator is
//! diagonal in the Fourier basis:
//!
//! ```text
//! (∇² + k0² + iε) u = q
//! û(k) = q̂(k) / (k0² + iε - |k|²)
//! ```
//!
//! The inverse transform in `math::fft` is normalized, so the implementation is
//! the exact pseudospectral inverse for the discrete periodic grid.

use super::absorbing::{absorbing_weights, AbsorbingBoundary};
use super::grid::GridSpec;
use crate::math::fft::{fft_3d_complex_into, ifft_3d_complex_inplace, Complex64};
use ndarray::Array3;
use std::f64::consts::TAU;

#[cfg(test)]
pub(super) fn apply_shifted_green_spectral(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    source_density: &[Complex64],
) -> Vec<Complex64> {
    apply_shifted_green_spectral_with_boundary(
        grid,
        reference_wavenumber,
        epsilon,
        source_density,
        AbsorbingBoundary::disabled(),
    )
}

#[cfg(test)]
pub(super) fn apply_shifted_green_spectral_adjoint(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    field: &[Complex64],
) -> Vec<Complex64> {
    apply_shifted_green_spectral_adjoint_with_boundary(
        grid,
        reference_wavenumber,
        epsilon,
        field,
        AbsorbingBoundary::disabled(),
    )
}

pub(super) fn apply_shifted_green_spectral_with_boundary(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    source_density: &[Complex64],
    absorbing_boundary: AbsorbingBoundary,
) -> Vec<Complex64> {
    apply_spectral_multiplier(
        grid,
        reference_wavenumber,
        epsilon,
        source_density,
        false,
        absorbing_boundary,
    )
}

pub(super) fn apply_shifted_green_spectral_adjoint_with_boundary(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    field: &[Complex64],
    absorbing_boundary: AbsorbingBoundary,
) -> Vec<Complex64> {
    apply_spectral_multiplier(
        grid,
        reference_wavenumber,
        epsilon,
        field,
        true,
        absorbing_boundary,
    )
}

fn apply_spectral_multiplier(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    values: &[Complex64],
    adjoint: bool,
    absorbing_boundary: AbsorbingBoundary,
) -> Vec<Complex64> {
    let weights =
        absorbing_weights(grid, absorbing_boundary).expect("validated absorbing boundary");
    let mut real_space =
        Array3::from_shape_vec(grid.dimensions, values.to_vec()).expect("validated CBS shape");
    for (value, &weight) in real_space.iter_mut().zip(weights.iter()) {
        *value *= weight;
    }
    let mut spectrum = Array3::zeros(grid.dimensions);
    fft_3d_complex_into(&real_space, &mut spectrum);

    let (nx, ny, nz) = grid.dimensions;
    for ix in 0..nx {
        let kx = angular_mode(ix, nx, grid.spacing_m);
        for iy in 0..ny {
            let ky = angular_mode(iy, ny, grid.spacing_m);
            for iz in 0..nz {
                let kz = angular_mode(iz, nz, grid.spacing_m);
                let denominator = Complex64::new(
                    reference_wavenumber
                        .mul_add(reference_wavenumber, -(kx * kx + ky * ky + kz * kz)),
                    epsilon,
                );
                let multiplier = if adjoint {
                    Complex64::new(1.0, 0.0) / denominator.conj()
                } else {
                    Complex64::new(1.0, 0.0) / denominator
                };
                spectrum[[ix, iy, iz]] *= multiplier;
            }
        }
    }

    real_space.assign(&spectrum);
    ifft_3d_complex_inplace(&mut real_space);
    for (value, &weight) in real_space.iter_mut().zip(weights.iter()) {
        *value *= weight;
    }
    real_space.iter().copied().collect()
}

fn angular_mode(index: usize, count: usize, spacing_m: f64) -> f64 {
    let signed_index = if index <= count / 2 {
        index as f64
    } else {
        index as f64 - count as f64
    };
    TAU * signed_index / (count as f64 * spacing_m)
}
