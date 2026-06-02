//! k-space operator precomputation for GPU PSTD construction.
//!
//! Computes kappa, source_kappa, and 1-D complex shift arrays from grid
//! dimensions and physical parameters. All outputs are f32-packed for
//! direct upload to GPU buffers.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::shift_operators::{generate_kappa, generate_shift_1d, generate_source_kappa};

/// Precompute kappa, source_kappa, and packed 1-D shift arrays for k-space.
///
/// Returns `(kappa_f32, source_kappa_f32, shifts_all)` where `shifts_all`
/// is packed as:
/// `[x_pos_re, x_pos_im, x_neg_re, x_neg_im (nx each),
///   y_pos_re, y_pos_im, y_neg_re, y_neg_im (ny each),
///   z_pos_re, z_pos_im, z_neg_re, z_neg_im (nz each)]`
pub(in crate::forward::pstd::gpu_pstd) fn precompute_kspace_shifts(
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    c_ref: f64,
    dt: f64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let kappa_3d = generate_kappa(nx, ny, nz, dx, dy, dz, c_ref, dt);
    let kappa_f32: Vec<f32> = kappa_3d.iter().map(|&v| v as f32).collect();

    let source_kappa_3d = generate_source_kappa(nx, ny, nz, dx, dy, dz, c_ref, dt);
    let source_kappa_f32: Vec<f32> = source_kappa_3d.iter().map(|&v| v as f32).collect();

    let dk_x = TWO_PI / (nx as f64 * dx);
    let dk_y = TWO_PI / (ny as f64 * dy);
    let dk_z = TWO_PI / (nz as f64 * dz);
    let (sx_pos, sx_neg) = generate_shift_1d(nx, dk_x, dx);
    let (sy_pos, sy_neg) = generate_shift_1d(ny, dk_y, dy);
    let (sz_pos, sz_neg) = generate_shift_1d(nz, dk_z, dz);

    let mut shifts_all: Vec<f32> = Vec::with_capacity(4 * (nx + ny + nz));
    for c in &sx_pos {
        shifts_all.push(c.re as f32);
    }
    for c in &sx_pos {
        shifts_all.push(c.im as f32);
    }
    for c in &sx_neg {
        shifts_all.push(c.re as f32);
    }
    for c in &sx_neg {
        shifts_all.push(c.im as f32);
    }
    for c in &sy_pos {
        shifts_all.push(c.re as f32);
    }
    for c in &sy_pos {
        shifts_all.push(c.im as f32);
    }
    for c in &sy_neg {
        shifts_all.push(c.re as f32);
    }
    for c in &sy_neg {
        shifts_all.push(c.im as f32);
    }
    for c in &sz_pos {
        shifts_all.push(c.re as f32);
    }
    for c in &sz_pos {
        shifts_all.push(c.im as f32);
    }
    for c in &sz_neg {
        shifts_all.push(c.re as f32);
    }
    for c in &sz_neg {
        shifts_all.push(c.im as f32);
    }

    (kappa_f32, source_kappa_f32, shifts_all)
}
