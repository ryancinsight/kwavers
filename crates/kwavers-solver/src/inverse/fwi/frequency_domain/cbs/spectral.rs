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
use super::temporal::pstd_leapfrog_symbol;
use kwavers_math::fft::{fft_3d_complex_into, ifft_3d_complex_inplace, Complex64};
use leto::Array3;
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
        SpectralApplication::Forward,
        SpectralSymbol::ContinuousHelmholtz,
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
        SpectralApplication::Adjoint,
        SpectralSymbol::ContinuousHelmholtz,
        absorbing_boundary,
    )
}

pub(super) fn apply_shifted_green_pstd_spectral_with_boundary(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    source_density: &[Complex64],
    time_step_s: f64,
    reference_sound_speed_m_s: f64,
    absorbing_boundary: AbsorbingBoundary,
) -> Vec<Complex64> {
    apply_spectral_multiplier(
        grid,
        reference_wavenumber,
        epsilon,
        source_density,
        SpectralApplication::Forward,
        SpectralSymbol::PstdLeapfrog {
            time_step_s,
            reference_sound_speed_m_s,
        },
        absorbing_boundary,
    )
}

pub(super) fn apply_shifted_green_pstd_spectral_adjoint_with_boundary(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    field: &[Complex64],
    time_step_s: f64,
    reference_sound_speed_m_s: f64,
    absorbing_boundary: AbsorbingBoundary,
) -> Vec<Complex64> {
    apply_spectral_multiplier(
        grid,
        reference_wavenumber,
        epsilon,
        field,
        SpectralApplication::Adjoint,
        SpectralSymbol::PstdLeapfrog {
            time_step_s,
            reference_sound_speed_m_s,
        },
        absorbing_boundary,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpectralApplication {
    Forward,
    Adjoint,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SpectralSymbol {
    ContinuousHelmholtz,
    PstdLeapfrog {
        time_step_s: f64,
        reference_sound_speed_m_s: f64,
    },
}

fn apply_spectral_multiplier(
    grid: GridSpec,
    reference_wavenumber: f64,
    epsilon: f64,
    values: &[Complex64],
    application: SpectralApplication,
    symbol: SpectralSymbol,
    absorbing_boundary: AbsorbingBoundary,
) -> Vec<Complex64> {
    let weights =
        absorbing_weights(grid, absorbing_boundary).expect("validated absorbing boundary");
    let (nx, ny, nz) = grid.dimensions;
    let mut real_space =
        Array3::from_shape_vec([nx, ny, nz], values.to_vec()).expect("validated CBS shape");
    for (value, &weight) in real_space.iter_mut().zip(weights.iter()) {
        *value *= weight;
    }
    let mut spectrum = Array3::<Complex64>::from_elem([nx, ny, nz], Complex64::default());
    fft_3d_complex_into(&real_space, &mut spectrum);

    let (nx, ny, nz) = grid.dimensions;
    for ix in 0..nx {
        let kx = angular_mode(ix, nx, grid.spacing_m);
        for iy in 0..ny {
            let ky = angular_mode(iy, ny, grid.spacing_m);
            for iz in 0..nz {
                let kz = angular_mode(iz, nz, grid.spacing_m);
                let denominator =
                    spectral_denominator(symbol, reference_wavenumber, epsilon, kx, ky, kz);
                let multiplier = if application == SpectralApplication::Adjoint {
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

fn spectral_denominator(
    symbol: SpectralSymbol,
    reference_wavenumber: f64,
    epsilon: f64,
    kx: f64,
    ky: f64,
    kz: f64,
) -> Complex64 {
    let grid_wavenumber_squared = kx.mul_add(kx, ky.mul_add(ky, kz * kz));
    let real = match symbol {
        SpectralSymbol::ContinuousHelmholtz => {
            reference_wavenumber.mul_add(reference_wavenumber, -grid_wavenumber_squared)
        }
        SpectralSymbol::PstdLeapfrog {
            time_step_s,
            reference_sound_speed_m_s,
        } => pstd_leapfrog_symbol(
            reference_wavenumber,
            grid_wavenumber_squared.sqrt(),
            time_step_s,
            reference_sound_speed_m_s,
        ),
    };
    Complex64::new(real, epsilon)
}

fn angular_mode(index: usize, count: usize, spacing_m: f64) -> f64 {
    let signed_index = if index <= count / 2 {
        index as f64
    } else {
        index as f64 - count as f64
    };
    TAU * signed_index / (count as f64 * spacing_m)
}
