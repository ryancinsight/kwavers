//! Embed a [`FocalKernel`] into a target grid centred on the planner's
//! focal voxel, zero-filling regions outside the kernel footprint.

use ndarray::Array3;

use super::kernel::FocalKernel;

/// Embed `kernel.field` into a `target_shape` grid such that the
/// kernel's focal voxel lands at `target_focus_idx`. Voxels outside
/// the kernel's footprint are zero-filled.
///
/// The output is the per-voxel peak rarefactional pressure (Pa) with
/// the kernel's amplitude scale unchanged — the caller is responsible
/// for any subsequent normalization (e.g. dividing by the global max
/// to obtain a unit envelope) and for applying tissue absorption.
///
/// # Panics
///
/// Does not panic. Out-of-range index arithmetic is performed with
/// signed math and clamped before slicing.
#[must_use]
pub fn place_kernel_at_focus(
    kernel: &FocalKernel,
    target_shape: (usize, usize, usize),
    target_focus_idx: (usize, usize, usize),
) -> Array3<f64> {
    let (tnx, tny, tnz) = target_shape;
    let (knx, kny, knz) = kernel.field.dim();
    let (fkx, fky, fkz) = kernel.focus_idx;
    let (ftx, fty, ftz) = target_focus_idx;

    let mut out = Array3::<f64>::zeros((tnx, tny, tnz));

    // Kernel-coordinate range that maps into the target grid. We use
    // signed arithmetic to handle the case where the kernel extends
    // past either boundary of the target grid.
    let fkx_i = fkx as isize;
    let fky_i = fky as isize;
    let fkz_i = fkz as isize;
    let ftx_i = ftx as isize;
    let fty_i = fty as isize;
    let ftz_i = ftz as isize;
    let tnx_i = tnx as isize;
    let tny_i = tny as isize;
    let tnz_i = tnz as isize;

    let src_x0 = (fkx_i - ftx_i).max(0);
    let src_y0 = (fky_i - fty_i).max(0);
    let src_z0 = (fkz_i - ftz_i).max(0);
    let src_x1 = (fkx_i + (tnx_i - ftx_i)).min(knx as isize);
    let src_y1 = (fky_i + (tny_i - fty_i)).min(kny as isize);
    let src_z1 = (fkz_i + (tnz_i - ftz_i)).min(knz as isize);

    if src_x1 <= src_x0 || src_y1 <= src_y0 || src_z1 <= src_z0 {
        return out;
    }

    let dst_x0 = (src_x0 - fkx_i + ftx_i) as usize;
    let dst_y0 = (src_y0 - fky_i + fty_i) as usize;
    let dst_z0 = (src_z0 - fkz_i + ftz_i) as usize;
    let dst_x1 = dst_x0 + (src_x1 - src_x0) as usize;
    let dst_y1 = dst_y0 + (src_y1 - src_y0) as usize;
    let dst_z1 = dst_z0 + (src_z1 - src_z0) as usize;

    let src_view = kernel.field.slice(ndarray::s![
        src_x0 as usize..src_x1 as usize,
        src_y0 as usize..src_y1 as usize,
        src_z0 as usize..src_z1 as usize,
    ]);
    out.slice_mut(ndarray::s![dst_x0..dst_x1, dst_y0..dst_y1, dst_z0..dst_z1])
        .assign(&src_view);

    out
}
