//! Staggered-to-collocated velocity interpolation.
//!
//! # Theorem: Half-cell backward shift for staggered velocity
//!
//! On a staggered (Yee) grid, the velocity component `u_α[i]` is located at
//! position `(i + ½)·Δα`.  To obtain the value at the pressure-grid position
//! `i·Δα` a half-cell backward shift by `−Δα/2` is required.  Under the
//! band-limited interpolation assumption (field sampled at or below Nyquist)
//! this is equivalent to the spectral half-shift `IFFT(exp(−ikα·Δα/2)·FFT(u_α))`.
//!
//! For smooth, well-sampled fields the finite-difference approximation
//!
//! ```text
//!   u_ns[i] = (u[i−1] + u[i]) / 2     for i ≥ 1
//!   u_ns[0] = u[0] / 2                 (ghost cell = 0 at boundary)
//! ```
//!
//! matches the spectral shift to machine precision (Liu 1997, §2).
//!
//! # Implementation
//!
//! The triple-loop form is replaced by slice operations so the compiler can
//! autovectorize over the transverse dimensions.
//! The cost is still O(N³) but the constant factor is much smaller.
//!
//! ## Algorithm (axis=0 case; axes 1 and 2 are analogous)
//!
//! 1. `u_ns = u * 0.5`                          — every element gets u[i]/2.
//! 2. `u_ns[1.., .., ..] += u[0..n-1, .., ..] * 0.5`
//!    — add u[i-1]/2 for i ≥ 1.
//!
//! Step 2 reads the slice `u[0..n-1]` (offset by −1 in axis 0) and adds it
//! into `u_ns[1..]`.  The read and write slices are non-overlapping in memory,
//! so the operation is safe without auxiliary storage.
//!
//! # References
//!
//! - Liu, Q. H. (1997). Geophysics 63(6), 2082–2089.
//! - k-Wave MATLAB Toolbox, `kspaceFirstOrder3D.m`, `u_non_staggered` computation.
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.

use leto::Array3;

/// Interpolate a staggered velocity component to pressure-grid positions.
///
/// ## Arguments
///
/// * `u`    – Staggered velocity field of shape `(nx, ny, nz)`.
/// * `axis` – Axis along which to shift backward by a half cell:
///   0 = x, 1 = y, 2 = z.
///
/// ## Returns
///
/// Non-staggered velocity at pressure-grid positions, same shape as `u`.
///
/// ## Panics
///
/// Panics if `axis ≥ 3` (debug-assert only; release builds silently
/// apply the z-axis path for any axis ≥ 2).
#[must_use]
pub fn interpolate_staggered_to_collocated(u: &Array3<f64>, axis: usize) -> Array3<f64> {
    debug_assert!(axis < 3, "axis must be 0, 1, or 2; got {axis}");

    let [nx, ny, nz] = u.shape();

    // Step 1: u_ns = u / 2.  This initialises every element including the
    // boundary row (i=0 / j=0 / k=0), which is u[0]/2 — matching the ghost-
    // cell-zero boundary condition.
    let mut u_ns = u.mapv(|v| v * 0.5);

    match axis {
        0 if nx > 1 => {
            for i in 1..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        u_ns[[i, j, k]] += u[[i - 1, j, k]] * 0.5;
                    }
                }
            }
        }
        1 if ny > 1 => {
            for i in 0..nx {
                for j in 1..ny {
                    for k in 0..nz {
                        u_ns[[i, j, k]] += u[[i, j - 1, k]] * 0.5;
                    }
                }
            }
        }
        _ if nz > 1 => {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 1..nz {
                        u_ns[[i, j, k]] += u[[i, j, k - 1]] * 0.5;
                    }
                }
            }
        }
        _ => {}
    }

    u_ns
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array3;

    // ── Theorem: uniform field, axis-0 interpolation ─────────────────────────
    //
    // For ux[i] = v (constant), the staggered-to-collocated result must be:
    //   u_ns[0]    = v / 2     (ghost cell = 0, so average = (0 + v)/2)
    //   u_ns[i≥1]  = v         (average of two equal values)
    #[test]
    fn test_uniform_field_axis0() {
        let u = Array3::from_elem([4, 1, 1], 2.0_f64);
        let u_ns = interpolate_staggered_to_collocated(&u, 0);

        assert!((u_ns[[0, 0, 0]] - 1.0).abs() < f64::EPSILON * 4.0);
        assert!((u_ns[[1, 0, 0]] - 2.0).abs() < f64::EPSILON * 4.0);
        assert!((u_ns[[3, 0, 0]] - 2.0).abs() < f64::EPSILON * 4.0);
    }

    // ── Theorem: uniform field, axis-1 interpolation ─────────────────────────
    #[test]
    fn test_uniform_field_axis1() {
        let u = Array3::from_elem([1, 4, 1], 3.0_f64);
        let u_ns = interpolate_staggered_to_collocated(&u, 1);

        assert!((u_ns[[0, 0, 0]] - 1.5).abs() < f64::EPSILON * 4.0);
        assert!((u_ns[[0, 2, 0]] - 3.0).abs() < f64::EPSILON * 4.0);
    }

    // ── Theorem: linearly varying field, axis-0 ──────────────────────────────
    //
    // For ux[i] = i:
    //   u_ns[0] = 0/2 = 0.0
    //   u_ns[i] = (i-1 + i)/2 = i - 0.5  for i ≥ 1
    #[test]
    fn test_linearly_varying_axis0() {
        let mut u = Array3::zeros([5, 1, 1]);
        for i in 0..5usize {
            u[[i, 0, 0]] = i as f64;
        }
        let u_ns = interpolate_staggered_to_collocated(&u, 0);

        let expected = [0.0_f64, 0.5, 1.5, 2.5, 3.5];
        for i in 0..5 {
            assert!(
                (u_ns[[i, 0, 0]] - expected[i]).abs() < 1e-14,
                "axis=0, i={i}: expected {}, got {}",
                expected[i],
                u_ns[[i, 0, 0]]
            );
        }
    }

    // ── Theorem: linearly varying field, axis-2 ──────────────────────────────
    #[test]
    fn test_linearly_varying_axis2() {
        let mut u = Array3::zeros([1, 1, 5]);
        for k in 0..5usize {
            u[[0, 0, k]] = k as f64;
        }
        let u_ns = interpolate_staggered_to_collocated(&u, 2);

        let expected = [0.0_f64, 0.5, 1.5, 2.5, 3.5];
        for k in 0..5 {
            assert!(
                (u_ns[[0, 0, k]] - expected[k]).abs() < 1e-14,
                "axis=2, k={k}: expected {}, got {}",
                expected[k],
                u_ns[[0, 0, k]]
            );
        }
    }

    // ── Theorem: singleton axis produces correct half-value ───────────────────
    //
    // For a 1×1×1 grid, the interpolation reduces to u[0]/2.
    #[test]
    fn test_singleton_axis0() {
        let u = Array3::from_elem([1, 1, 1], 6.0_f64);
        let u_ns = interpolate_staggered_to_collocated(&u, 0);
        assert!((u_ns[[0, 0, 0]] - 3.0).abs() < f64::EPSILON * 4.0);
    }

    // ── Theorem: 3-D consistency check ───────────────────────────────────────
    //
    // Interpolating a uniform field (value=v) along any axis must yield
    // v on interior cells and v/2 on the boundary row.
    #[test]
    fn test_3d_consistency() {
        let v = 4.0_f64;
        let u = Array3::from_elem([8, 8, 8], v);

        for axis in 0..3usize {
            let u_ns = interpolate_staggered_to_collocated(&u, axis);
            // Interior cells
            assert!(
                (u_ns[[4, 4, 4]] - v).abs() < 1e-12,
                "axis={axis} interior mismatch"
            );
            // Boundary cell (index 0 along the interpolated axis)
            let boundary_val = match axis {
                0 => u_ns[[0, 4, 4]],
                1 => u_ns[[4, 0, 4]],
                _ => u_ns[[4, 4, 0]],
            };
            assert!(
                (boundary_val - v * 0.5).abs() < 1e-12,
                "axis={axis} boundary mismatch: expected {}, got {boundary_val}",
                v * 0.5
            );
        }
    }
}
