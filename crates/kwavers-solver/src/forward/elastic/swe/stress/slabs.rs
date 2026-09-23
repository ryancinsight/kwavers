//! The velocity kick of an elastic field by its stress divergence,
//! evaluated a slab of x-planes at a time through a stress window once its
//! fields outgrow the caches (ADR 133).

use super::super::scratch::ElasticStepScratch;
use super::super::types::ElasticWaveField;
use super::divergence::{
    derivatives, stress_components, validate_stress_divergence_shapes, DensityScale, VelocityKick,
};
use core::num::NonZeroUsize;
use core::ops::Range;
use kwavers_core::arena::{cache_capacity_bytes, last_level_cache_bytes};
use kwavers_grid::Grid;
use std::sync::OnceLock;

use leto::{Array3, ArrayView3};
use leto_ops::{Axis, FiniteDifference3D, PlaneWindow, PlaneWindowMut};

/// The half-step kick of `field`'s velocities by the acceleration its
/// displacement gives: `v + half_dt · (∇·σ) / ρ`, per component.
///
/// The scale and the kick ride the divergence pass. Scaling the divergence
/// in a pass of its own costs three grids written and read back; storing the
/// acceleration for the kick to read costs three more, which past the
/// caches is a write-allocate, a write-back and a read per field. Here each
/// velocity is read and written once. Values are unchanged: the divergence
/// is summed in x, y, z order, scaled, and added at `half_dt` in the order
/// the separate passes took, and every intermediate is a value they stored
/// and reloaded without rounding.
///
/// `scratch`'s divergence and acceleration fields are untouched here; the
/// body-force route still writes and reads them, since
/// `(divergence + force) / rho` is not the same rounding as scaling the
/// divergence alone.
///
/// Scratch whose stress fields are not C-contiguous evaluates whole-grid,
/// since the window slides its planes as contiguous runs.
///
/// # Panics
///
/// Panics if any field or scratch shape differs from the grid's.
pub(crate) fn stress_kick_into(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &mut ElasticWaveField,
    kick: &VelocityKick<'_>,
    scratch: &mut ElasticStepScratch,
) {
    let slab = if window_slides(scratch) {
        slab_planes(grid, &kick.scale)
    } else {
        NonZeroUsize::new(grid.nx).unwrap_or(NonZeroUsize::MIN)
    };
    stress_kick_in_slabs(grid, lambda, mu, field, kick, scratch, slab);
}

/// Whether the stress window can slide in `scratch`: it moves planes as
/// contiguous runs, which C-contiguous stress fields hold.
fn window_slides(scratch: &ElasticStepScratch) -> bool {
    [
        &scratch.sxx,
        &scratch.syy,
        &scratch.szz,
        &scratch.sxy,
        &scratch.sxz,
        &scratch.syz,
    ]
    .iter()
    .all(|stress| stress.as_slice().is_some())
}

/// Fields a slab keeps in flight per plane with a uniform density: the six
/// stresses of the window, the three displacement components, the Lamé pair
/// and the three velocities. A density field adds one.
const LIVE_FIELDS: usize = 14;

/// How many x-planes each slab of the kick evaluation covers.
///
/// Whole-grid while the evaluation's live fields fit the caches an even
/// split across the workers keeps resident (`cache_capacity_bytes`): then
/// every intermediate is still resident when it is read back, and slabs
/// only add passes. Past them, the largest slab whose window and the planes
/// around it fit the shared last-level cache, and no more than the worker
/// count, since each pass hands one plane to a task.
///
/// Measured through `swe_kick_slab_sweep` (release, fastest of 20
/// paired repeats, a 285K: 24 workers, 60 MiB held by an even split, a
/// 36 MiB last level): at 64, 72 and 76 cubed every slab height is slower
/// than whole-grid -- the last two past the last level alone -- at 80 cubed
/// the best is level with it, at 88 cubed 16-plane slabs run 1568-1609 us
/// against 2017-2098, at 96 cubed 24-plane slabs run
/// 1857-1904 us against 2742-3113 (1.5x), and at 128 cubed 16-plane slabs
/// run 5294-5634 us against 9712-10677 (1.8x). The rule gives 24 planes at
/// 96 cubed and 16 at 128, the measured best of 8, 12, 16, 24 and 32 at
/// each; which of its two limits binds is what moves the optimum between
/// them.
///
/// A platform reporting no cache size evaluates whole-grid, the route that
/// wins whenever the fields fit.
fn slab_planes(grid: &Grid, scale: &DensityScale<'_>) -> NonZeroUsize {
    let live = LIVE_FIELDS + usize::from(matches!(scale, DensityScale::Field(_)));
    let caches = cache_capacity_bytes()
        .zip(last_level_cache_bytes())
        .map(|(total, last_level)| Caches { total, last_level });
    slab_height(grid.nx, grid.ny * grid.nz, live, caches, workers())
}

/// The cache capacities the slab rule reads, in bytes.
struct Caches {
    /// What an even split across the workers keeps resident.
    total: usize,
    /// The shared last level.
    last_level: usize,
}

/// [`slab_planes`]'s rule on its inputs: a grid of `planes` x-planes of
/// `plane_cells` cells each, `live` fields in flight, the platform's caches
/// if it reports them, and `workers` threads.
fn slab_height(
    planes: usize,
    plane_cells: usize,
    live: usize,
    caches: Option<Caches>,
    workers: usize,
) -> NonZeroUsize {
    let whole = NonZeroUsize::new(planes).unwrap_or(NonZeroUsize::MIN);
    let Some(Caches { total, last_level }) = caches else {
        return whole;
    };
    let per_plane = live * plane_cells * size_of::<f64>();
    if per_plane.saturating_mul(planes) <= total {
        return whole;
    }
    let fitting = (last_level / per_plane.max(1)).saturating_sub(2 * STENCIL_REACH);
    NonZeroUsize::new(fitting.min(workers))
        .unwrap_or(NonZeroUsize::MIN)
        .min(whole)
}

/// The threads a pass can spread over, read once: the count does not change
/// while the process runs, and every evaluation asks.
fn workers() -> usize {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    *WORKERS.get_or_init(|| std::thread::available_parallelism().map_or(1, usize::from))
}

/// Planes a fourth-order derivative reaches on either side of its own.
///
/// A slab's kick at planes `start..end` read stresses on
/// `start - STENCIL_REACH..end + STENCIL_REACH`, so those stress planes must
/// be held before the slab's divergence runs.
const STENCIL_REACH: usize = 2;

/// [`stress_kick_into`], evaluated `slab` x-planes at a time through
/// a stress window.
///
/// The window is the leading planes of `scratch`'s six stress fields: each
/// slab holds there the stress planes its divergence reads, starting at grid
/// plane `start - STENCIL_REACH`. The planes the previous slab already
/// computed and this one still needs slide to the front, and only the rest
/// are computed, so no stress plane is computed twice. The same few
/// megabytes are rewritten for every slab, so they stay in cache: the stress
/// never makes the round trip through DRAM that a grid-sized intermediate
/// does, where every line written is first read back from planes last
/// touched a whole pass earlier.
///
/// A `slab` of `nx` or more is one slab, and the window is then the whole of
/// each stress field: the whole-grid evaluation. Either way every plane of
/// every velocity is the value the whole-grid evaluation writes, since
/// leto's windowed passes take the grid's stencil at each grid plane.
///
/// After a slabbed evaluation the stress fields hold the last slab's window,
/// not the grid's stress.
///
/// # Panics
///
/// Panics if any field or scratch shape differs from the grid's, or if
/// `slab` is under the grid's plane count and a stress field is not
/// C-contiguous.
pub(crate) fn stress_kick_in_slabs(
    grid: &Grid,
    lambda: &Array3<f64>,
    mu: &Array3<f64>,
    field: &mut ElasticWaveField,
    kick: &VelocityKick<'_>,
    scratch: &mut ElasticStepScratch,
    slab: NonZeroUsize,
) {
    validate_stress_divergence_shapes(grid, lambda, mu, field, scratch);
    let op = derivatives(grid);
    let ElasticStepScratch {
        sxx,
        syy,
        szz,
        sxy,
        sxz,
        syz,
        ..
    } = scratch;
    let ElasticWaveField {
        ux,
        uy,
        uz,
        vx,
        vy,
        vz,
        ..
    } = field;
    let mut stresses = [sxx, syy, szz, sxy, sxz, syz];
    let [nx, ny, nz] = [grid.nx, grid.ny, grid.nz];
    let plane = ny * nz;
    let mut held = 0..0;
    for start in (0..nx).step_by(slab.get()) {
        let end = start.saturating_add(slab.get()).min(nx);
        let needed = start.saturating_sub(STENCIL_REACH)..end.saturating_add(STENCIL_REACH).min(nx);
        let kept = needed.start.max(held.start)..held.end;
        if kept.start < kept.end && kept.start > held.start {
            let from = (kept.start - held.start) * plane..(kept.end - held.start) * plane;
            for stress in &mut stresses {
                stress
                    .as_slice_mut()
                    .expect("invariant: scratch fields are C-contiguous")
                    .copy_within(from.clone(), 0);
            }
        }
        let fresh = kept.end.max(needed.start)..needed.end;
        let window = [(0, needed.len(), 1), (0, ny, 1), (0, nz, 1)];
        stress_components(
            &op,
            fresh,
            lambda,
            mu,
            [&*ux, &*uy, &*uz],
            stresses.each_mut().map(|stress| {
                stress
                    .slice_mut(&window)
                    .expect("invariant: the window lies within the stress field")
            }),
            needed.start,
        );
        kick_velocities(
            &op,
            start..end,
            stresses.each_ref().map(|stress| {
                stress
                    .slice(&window)
                    .expect("invariant: the window lies within the stress field")
            }),
            needed.start,
            kick,
            [&mut *vx, &mut *vy, &mut *vz],
        );
        held = needed;
    }
}

/// The kick of the velocities on the grid planes `planes` by the divergence
/// of the stress tensor, whose fields hold the grid planes from `origin` on,
/// scaled by the density.
///
/// One pass for all three. Taken separately they read nine stress lanes over
/// six distinct fields -- `sxy`, `sxz` and `syz` each feed two of the three
/// -- so each repeated field crossed the bus twice. Here every field is read
/// once per output lane: at 96 cubed that is 70 MB against 105, and one pass
/// rather than three.
fn kick_velocities(
    op: &FiniteDifference3D<f64>,
    planes: Range<usize>,
    [sxx, syy, szz, sxy, sxz, syz]: [ArrayView3<'_, f64>; 6],
    origin: usize,
    kick: &VelocityKick<'_>,
    [vx, vy, vz]: [&mut Array3<f64>; 3],
) {
    let grid_planes = vx.shape()[0];
    let half_dt = kick.half_dt;
    let (mut x_out, mut y_out, mut z_out) = (vx.view_mut(), vy.view_mut(), vz.view_mut());
    let held = |stress| PlaneWindow::new(stress, origin);
    let terms = [
        (Axis::X, held(sxx)),
        (Axis::Y, held(sxy)),
        (Axis::Z, held(sxz)),
        (Axis::X, held(sxy)),
        (Axis::Y, held(syy)),
        (Axis::Z, held(syz)),
        (Axis::X, held(sxz)),
        (Axis::Y, held(syz)),
        (Axis::Z, held(szz)),
    ];
    let destinations = [
        PlaneWindowMut::whole(&mut x_out),
        PlaneWindowMut::whole(&mut y_out),
        PlaneWindowMut::whole(&mut z_out),
    ];
    match kick.scale {
        DensityScale::UniformReciprocal(reciprocal) => op.map_axis_derivatives_in_windows(
            grid_planes,
            planes,
            terms,
            [],
            destinations,
            |[xx, xy, xz, yx, yy, yz, zx, zy, zz], [], [x, y, z]| {
                [
                    x + half_dt * (((xx + xy) + xz) * reciprocal),
                    y + half_dt * (((yx + yy) + yz) * reciprocal),
                    z + half_dt * (((zx + zy) + zz) * reciprocal),
                ]
            },
        ),
        DensityScale::Field(density) => op.map_axis_derivatives_in_windows(
            grid_planes,
            planes,
            terms,
            [PlaneWindow::whole(density)],
            destinations,
            |[xx, xy, xz, yx, yy, yz, zx, zy, zz], [rho], [x, y, z]| {
                [
                    x + half_dt * (((xx + xy) + xz) / rho),
                    y + half_dt * (((yx + yy) + yz) / rho),
                    z + half_dt * (((zx + zy) + zz) / rho),
                ]
            },
        ),
    }
    .expect("invariant: validated elastic fields share the grid shape");
}

#[cfg(test)]
mod tests;
