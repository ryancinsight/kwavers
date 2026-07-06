//! Trilinear 3-D resampling of a [`FocalKernel`] onto a new isotropic
//! grid spacing.
//!
//! ## Method
//!
//! For each output voxel at integer index `(i', j', k')`, compute the
//! corresponding input-grid coordinate `(x, y, z) = (i', j', k') *
//! out_dx / in_dx` and trilinearly interpolate from the 8 surrounding
//! input voxels. Output voxels whose input footprint falls outside the
//! source array clamp to the nearest in-range voxel.
//!
//! ## Why not cubic?
//!
//! Cubic-spline (Catmull-Rom or B-spline) resampling preserves the
//! focal peak ~5 % better than trilinear under heavy downsampling, but
//! costs ~8× more arithmetic per voxel and risks Gibbs overshoot on
//! sharp pressure gradients. Since the planner's downstream
//! `kernel_focal_envelope` normalizes by the global maximum, absolute
//! peak preservation is recovered downstream — what the planner cares
//! about is *shape* fidelity, which trilinear handles adequately for
//! the 0.5–1.5× downsampling ratios typical in this pipeline.
//!
//! Cubic can be added behind a separate function if a future use case
//! requires it.

use ndarray::Array3;

use super::kernel::FocalKernel;

/// Trilinearly resample `kernel.field` onto a new isotropic grid
/// spacing `target_dx_m`. The focal voxel index in the output kernel
/// is `round(input_focus_idx * input_dx / output_dx)` — the focus
/// physical position is preserved (within sub-voxel rounding).
///
/// If `(target_dx_m - kernel.dx_m).abs() < 1e-9` the input is returned
/// as a clone (no resampling needed).
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[must_use]
pub fn resample_trilinear(kernel: &FocalKernel, target_dx_m: f64) -> FocalKernel {
    debug_assert!(
        target_dx_m > 0.0,
        "resample_trilinear: target_dx_m must be > 0"
    );
    if (target_dx_m - kernel.dx_m).abs() < 1e-9 {
        return kernel.clone();
    }
    let zoom = kernel.dx_m / target_dx_m;
    let (nxi, nyi, nzi) = kernel.field.dim();
    let nxo = ((nxi as f64) * zoom).round().max(1.0) as usize;
    let nyo = ((nyi as f64) * zoom).round().max(1.0) as usize;
    let nzo = ((nzi as f64) * zoom).round().max(1.0) as usize;

    let mut out = Array3::<f64>::zeros((nxo, nyo, nzo));
    let inv_zoom = 1.0 / zoom;
    let inp = &kernel.field;

    // Each output voxel reads eight immutable input voxels and writes one
    // disjoint output voxel, so the traversal is race-free under Moirai.
    crate::parallel::for_each_indexed_mut(out.view_mut(), |(io, jo, ko), out_val| {
        let xi = (io as f64) * inv_zoom;
        let x0 = xi.floor() as isize;
        let fx = xi - (x0 as f64);
        let x0u = x0.clamp(0, (nxi - 1) as isize) as usize;
        let x1u = (x0 + 1).clamp(0, (nxi - 1) as isize) as usize;

        let yj = (jo as f64) * inv_zoom;
        let y0 = yj.floor() as isize;
        let fy = yj - (y0 as f64);
        let y0u = y0.clamp(0, (nyi - 1) as isize) as usize;
        let y1u = (y0 + 1).clamp(0, (nyi - 1) as isize) as usize;

        let zk = (ko as f64) * inv_zoom;
        let z0 = zk.floor() as isize;
        let fz = zk - (z0 as f64);
        let z0u = z0.clamp(0, (nzi - 1) as isize) as usize;
        let z1u = (z0 + 1).clamp(0, (nzi - 1) as isize) as usize;

        // 8-corner trilinear interpolation
        let c000 = inp[[x0u, y0u, z0u]];
        let c100 = inp[[x1u, y0u, z0u]];
        let c010 = inp[[x0u, y1u, z0u]];
        let c110 = inp[[x1u, y1u, z0u]];
        let c001 = inp[[x0u, y0u, z1u]];
        let c101 = inp[[x1u, y0u, z1u]];
        let c011 = inp[[x0u, y1u, z1u]];
        let c111 = inp[[x1u, y1u, z1u]];

        let c00 = c000.mul_add(1.0 - fx, c100 * fx);
        let c10 = c010.mul_add(1.0 - fx, c110 * fx);
        let c01 = c001.mul_add(1.0 - fx, c101 * fx);
        let c11 = c011.mul_add(1.0 - fx, c111 * fx);
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        *out_val = c0 * (1.0 - fz) + c1 * fz;
    });

    let new_focus = (
        ((kernel.focus_idx.0 as f64) * zoom)
            .round()
            .clamp(0.0, (nxo - 1) as f64) as usize,
        ((kernel.focus_idx.1 as f64) * zoom)
            .round()
            .clamp(0.0, (nyo - 1) as f64) as usize,
        ((kernel.focus_idx.2 as f64) * zoom)
            .round()
            .clamp(0.0, (nzo - 1) as f64) as usize,
    );

    FocalKernel::new(
        out,
        target_dx_m,
        new_focus,
        kernel.f0,
        kernel.pnp_realised,
        kernel.source_pa,
        kernel.fwhm_lat_m,
        kernel.fwhm_ax_m,
    )
}
