//! Separable Gaussian smoothing passes and the projection row-norm cache.
//!
//! Extracted from `pipeline.rs`: the four smoothing passes, the routine that
//! sequences them, and the row-norm cache are one operation family, and
//! `pipeline.rs` had grown past the 500-line target carrying that family's
//! documentation inside it. The struct's own documentation stayed behind.

use crate::reconstruction::acoustic_projection::AcousticProjectionGeometry;
use kwavers_core::error::KwaversResult;
use leto::Array3;
use moirai_parallel::{for_each_chunk_mut_enumerated_with, map_collect_index_with, Adaptive};

/// Separable 3-point Gaussian-weighted smoothing (σ in grid points).
///
/// ## Implementation
///
/// Three sequential 1-D passes (X → Y → Z) using a 2-buffer ping-pong
/// pattern: one `Array3` allocation (the write buffer; the caller's image
/// is consumed and reused as the first read buffer), with `Array3::assign`
/// (memcpy, no allocation) used to propagate edge-plane values between
/// passes.  Each pass interior is computed in parallel via Moirai chunk
/// dispatch.
///
/// ## Correctness invariant
///
/// Edge planes in each dimension (index 0 and dim−1) are not written by
/// their respective pass, preserving the same boundary semantics as the
/// original sequential implementation.
///
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
pub(crate) fn apply_smoothing(image: Array3<f64>, sigma: f64) -> KwaversResult<Array3<f64>> {
    if sigma <= 0.0 {
        return Ok(image);
    }
    let [nx, ny, nz] = image.shape();
    if nx < 3 || ny < 3 || nz < 3 {
        return Ok(image);
    }
    let wn = (-0.5 / (sigma * sigma)).exp();
    let norm_w = 2.0f64.mul_add(wn, 1.0);
    let w0 = 1.0 / norm_w;
    let wn = wn / norm_w;

    // One allocation total (down from two): the caller's image becomes the
    // first read buffer and the write buffer starts zeroed, with edge
    // planes propagated by `assign` before each pass.
    let mut a = image;
    let mut b = Array3::zeros((nx, ny, nz));

    // --- X pass: interior i ∈ [1, nx−2] ---
    b.assign(&a);
    smooth_x_pass(&a, &mut b, nx, ny, nz, wn, w0);
    std::mem::swap(&mut a, &mut b);
    // Propagate current result (in a) into b so j/k edge planes are correct.
    b.assign(&a);

    // --- Y pass: interior j ∈ [1, ny−2] ---
    smooth_y_pass(&a, &mut b, ny, nz, wn, w0);
    std::mem::swap(&mut a, &mut b);
    b.assign(&a);

    // --- Z pass: interior k ∈ [1, nz−2] ---
    smooth_z_pass(&a, &mut b, nz, wn, w0);
    Ok(b)
}

/// Compute the per-sensor squared row norm `‖A_row_s‖²` for the acoustic projection matrix.
///
/// ## Theorem: Row-norm preconditioning in SIRT (Dines & Kak 1979 §III)
///
/// The SIRT update
/// ```text
/// x^(k+1) = x^(k) + λ · D_R · Aᵀ · (b − A·x^(k))
/// ```
/// requires the diagonal preconditioner `D_R`s` = 1/‖A_row_s‖²` so that each
/// sensor is scaled by its own energy; sensors with large geometric coverage do
/// not dominate the gradient update.
///
/// For the acoustic projection model `A[s,(i,j,k)] = exp(−2αf_c r) / r`:
/// ```text
/// ‖A_row_s‖² = Σ_{i,j,k} (exp(−2αf_c r_{s,v}) / r_{s,v})²
/// ```
///
/// This is computed once per geometry/grid-shape pair and cached in
/// `RealTimeSirtPipeline::row_norm_sq_cache`. Parallelised over sensors via Moirai.
/// The lower clamp at `f64::EPSILON` prevents a zero-denominator singularity
/// for far-field sensors whose stencil weight is below double-precision roundoff.
///
/// # References
/// - Dines KA, Kak AC (1979). "Ultrasonic attenuation tomography of soft tissues."
///   *Ultrasonic Imaging* 1(1):16–33, §III.
pub(crate) fn compute_row_norm_sq(
    geom: &AcousticProjectionGeometry,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let (dx, dy, dz) = geom.voxel_spacing;
    let alpha = geom.alpha_nepers_per_m_per_hz();
    let f_c = geom.center_frequency_hz;
    let zs = geom.element_z;

    map_collect_index_with::<Adaptive, _, _>(geom.element_x.len(), |sensor_idx| {
        let xs = geom.element_x[sensor_idx];
        let mut norm_sq = 0.0_f64;
        for i in 0..nx {
            let xv = i as f64 * dx;
            let dx2 = (xv - xs) * (xv - xs);
            for j in 0..ny {
                let yv = j as f64 * dy;
                let dxy2 = yv.mul_add(yv, dx2);
                for k in 0..nz {
                    let zv = k as f64 * dz;
                    let r = (zv - zs).mul_add(zv - zs, dxy2).sqrt().max(1e-6);
                    let weight = (-2.0 * alpha * f_c * r).exp() / r;
                    norm_sq += weight * weight;
                }
            }
        }
        // Clamp away from zero to prevent div-by-zero in D_R[s] = 1/‖A_row_s‖²
        norm_sq.max(f64::EPSILON)
    })
}

/// `a` and `b` are always C-contiguous here: [`RealTimeSirtPipeline::apply_smoothing`]
/// derives both from `image`, which is only ever built by `Array3::zeros` (C-contiguous
/// by construction, per leto `Layout::c_contiguous`) and thereafter mutated only by
/// scalar indexing or `.clone()`/`.assign()` (which preserve the destination's own
/// layout, never adopting the source's) — never transposed or reshaped.
pub(crate) fn smooth_x_pass(
    a: &Array3<f64>,
    b: &mut Array3<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    wn: f64,
    w0: f64,
) {
    let plane_len = ny * nz;
    let a_slice = a
        .as_slice()
        .expect("invariant: Array3 smoothing read buffer is C-contiguous (row-major)");
    let b_slice = b
        .as_slice_mut()
        .expect("invariant: Array3 smoothing write buffer is C-contiguous (row-major)");
    let interior = &mut b_slice[plane_len..(nx - 1) * plane_len];

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
        interior,
        plane_len,
        |chunk_idx, plane| {
            let i = chunk_idx + 1;
            let center_base = i * plane_len;
            let left_base = center_base - plane_len;
            let right_base = center_base + plane_len;

            for (offset, dst) in plane.iter_mut().enumerate() {
                *dst = wn * a_slice[left_base + offset]
                    + w0 * a_slice[center_base + offset]
                    + wn * a_slice[right_base + offset];
            }
        },
    );
}

/// See [`smooth_x_pass`]: `a`/`b` are the same chained `image` buffers, provably
/// C-contiguous.
pub(crate) fn smooth_y_pass(
    a: &Array3<f64>,
    b: &mut Array3<f64>,
    ny: usize,
    nz: usize,
    wn: f64,
    w0: f64,
) {
    let plane_len = ny * nz;
    let a_slice = a
        .as_slice()
        .expect("invariant: Array3 smoothing read buffer is C-contiguous (row-major)");
    let b_slice = b
        .as_slice_mut()
        .expect("invariant: Array3 smoothing write buffer is C-contiguous (row-major)");

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(b_slice, plane_len, |i, plane| {
        let plane_base = i * plane_len;
        for j in 1..ny - 1 {
            let row_base = j * nz;
            let left_base = plane_base + (j - 1) * nz;
            let center_base = plane_base + row_base;
            let right_base = plane_base + (j + 1) * nz;

            for k in 0..nz {
                plane[row_base + k] = wn * a_slice[left_base + k]
                    + w0 * a_slice[center_base + k]
                    + wn * a_slice[right_base + k];
            }
        }
    });
}

/// See [`smooth_x_pass`]: `a`/`b` are the same chained `image` buffers, provably
/// C-contiguous.
pub(crate) fn smooth_z_pass(a: &Array3<f64>, b: &mut Array3<f64>, nz: usize, wn: f64, w0: f64) {
    let a_slice = a
        .as_slice()
        .expect("invariant: Array3 smoothing read buffer is C-contiguous (row-major)");
    let b_slice = b
        .as_slice_mut()
        .expect("invariant: Array3 smoothing write buffer is C-contiguous (row-major)");

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(b_slice, nz, |row_idx, row| {
        let row_base = row_idx * nz;
        for k in 1..nz - 1 {
            row[k] = wn * a_slice[row_base + k - 1]
                + w0 * a_slice[row_base + k]
                + wn * a_slice[row_base + k + 1];
        }
    });
}
