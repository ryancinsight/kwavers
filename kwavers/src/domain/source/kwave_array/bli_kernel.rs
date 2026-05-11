//! Bandlimited interpolation (BLI) stencil, disc-basis, and nearest-index helpers.
//!
//! # Mathematical Foundation — BLI Stencil
//!
//! k-Wave's source rasterization maps continuous surface samples onto grid
//! points via a bandlimited interpolation kernel. For a 1-D row the weight at
//! grid index `i` for a sample at continuous coordinate `x` is:
//!
//! ```text
//! w_i = sinc(π(x_i − x) / Δx)
//! ```
//!
//! where `sinc(u) = sin(u)/u` (unnormalized). The 3-D weight is the tensor
//! product `w_x · w_y · w_z`. The stencil half-width in grid cells is
//! `N_sub = ceil(1 / (π · bli_tolerance))` so that weights outside the window
//! are below `bli_tolerance`. With the default `bli_tolerance = 0.05` this
//! gives `N_sub = 7`.
//!
//! Reference: Wise, E.S. et al. (2019). "Representing arbitrary acoustic
//! source and sensor distributions in Fourier collocation methods." J. Acoust.
//! Soc. Am. 146(1):278–288. Algorithm 1.

use super::{
    math::{DISC_AXIS_EPSILON, DISC_BLI_TOLERANCE, DISC_PACKING_NUMBER},
    KWaveArray,
};

impl KWaveArray {
    /// Map a single surface sample point through the BLI stencil, calling
    /// `visit(ix, iy, iz, weight)` for each grid cell that receives nonzero
    /// contribution.
    ///
    /// When `mask_only` is `true`, the weight argument passed to `visit` is
    /// always `1.0` and out-of-window cells are still visited (binary mask
    /// semantics). When `false`, exact BLI weights are computed and zero-weight
    /// cells are skipped.
    pub(super) fn map_surface_sample<F>(
        &self,
        grid: &crate::domain::grid::Grid,
        x_vec: &ndarray::Array1<f64>,
        y_vec: &ndarray::Array1<f64>,
        z_vec: &ndarray::Array1<f64>,
        point: [f64; 3],
        scale: f64,
        mask_only: bool,
        mut visit: F,
    ) where
        F: FnMut(usize, usize, usize, f64),
    {
        let decay_subs = (1.0 / (std::f64::consts::PI * DISC_BLI_TOLERANCE)).ceil() as isize;
        let ongrid_threshold = grid.dx * 1.0e-3;

        let (ix0, x_closest) = Self::nearest_coordinate_index(x_vec, point[0]);
        let (iy0, y_closest) = Self::nearest_coordinate_index(y_vec, point[1]);
        let (iz0, z_closest) = Self::nearest_coordinate_index(z_vec, point[2]);

        let x_on_grid = (x_closest - point[0]).abs() < ongrid_threshold;
        let y_on_grid = (y_closest - point[1]).abs() < ongrid_threshold;
        let z_on_grid = (z_closest - point[2]).abs() < ongrid_threshold;

        if grid.nz == 1 {
            let iz = iz0;
            for di in -decay_subs..=decay_subs {
                for dj in -decay_subs..=decay_subs {
                    if (di * dj).abs() > decay_subs {
                        continue;
                    }
                    if x_on_grid && di != 0 {
                        continue;
                    }
                    if y_on_grid && dj != 0 {
                        continue;
                    }
                    let ix = ix0 as isize + di;
                    let iy = iy0 as isize + dj;
                    if ix < 0 || iy < 0 || ix >= grid.nx as isize || iy >= grid.ny as isize {
                        continue;
                    }
                    let ix = ix as usize;
                    let iy = iy as usize;
                    let wx = Self::sinc(std::f64::consts::PI * (x_vec[ix] - point[0]) / grid.dx);
                    let wy = Self::sinc(std::f64::consts::PI * (y_vec[iy] - point[1]) / grid.dy);
                    let weight = scale * wx * wy;
                    if mask_only || weight != 0.0 {
                        visit(ix, iy, iz, if mask_only { 1.0 } else { weight });
                    }
                }
            }
            return;
        }

        for di in -decay_subs..=decay_subs {
            for dj in -decay_subs..=decay_subs {
                for dk in -decay_subs..=decay_subs {
                    if (di * dj * dk).abs() > decay_subs {
                        continue;
                    }
                    if x_on_grid && di != 0 {
                        continue;
                    }
                    if y_on_grid && dj != 0 {
                        continue;
                    }
                    if z_on_grid && dk != 0 {
                        continue;
                    }
                    let ix = ix0 as isize + di;
                    let iy = iy0 as isize + dj;
                    let iz = iz0 as isize + dk;
                    if ix < 0
                        || iy < 0
                        || iz < 0
                        || ix >= grid.nx as isize
                        || iy >= grid.ny as isize
                        || iz >= grid.nz as isize
                    {
                        continue;
                    }
                    let ix = ix as usize;
                    let iy = iy as usize;
                    let iz = iz as usize;
                    let wx = Self::sinc(std::f64::consts::PI * (x_vec[ix] - point[0]) / grid.dx);
                    let wy = Self::sinc(std::f64::consts::PI * (y_vec[iy] - point[1]) / grid.dy);
                    let wz = Self::sinc(std::f64::consts::PI * (z_vec[iz] - point[2]) / grid.dz);
                    let weight = scale * wx * wy * wz;
                    if mask_only || weight != 0.0 {
                        visit(ix, iy, iz, if mask_only { 1.0 } else { weight });
                    }
                }
            }
        }
    }

    /// Find the index of the coordinate value nearest to `value` in `coords`,
    /// returning `(index, coords[index])`.
    pub(super) fn nearest_coordinate_index(
        coords: &ndarray::Array1<f64>,
        value: f64,
    ) -> (usize, f64) {
        let mut best_index = 0usize;
        let mut best_value = coords[0];
        let mut best_distance = (best_value - value).abs();
        for (index, &coord) in coords.iter().enumerate().skip(1) {
            let distance = (coord - value).abs();
            if distance < best_distance {
                best_index = index;
                best_value = coord;
                best_distance = distance;
            }
        }
        (best_index, best_value)
    }

    /// Unnormalized sinc: `sin(x)/x` with the `x→0` limit returning 1.
    #[inline]
    pub(super) fn sinc(x: f64) -> f64 {
        if x.abs() <= f64::EPSILON {
            1.0
        } else {
            x.sin() / x
        }
    }

    /// Total sample count for a disc rasterized with `num_radial` rings
    /// (including the center point and all azimuthal samples).
    pub(super) fn disc_sample_count(&self, num_radial: usize) -> usize {
        if num_radial == 1 {
            return 1;
        }
        let mut num_points = 1usize;
        for ring_idx in 1..num_radial {
            let num_theta = ((ring_idx as f64) * DISC_PACKING_NUMBER).round().max(1.0) as usize;
            num_points += num_theta;
        }
        num_points
    }

    /// Construct an orthonormal basis `(u, v, n)` for the plane of the disc.
    ///
    /// `n` is the disc normal: `(focus_position − center) / ‖…‖` when a focus
    /// is provided, or `e_z` otherwise. `(u, v)` span the tangent plane.
    /// # Panics
    /// - Panics if assertion fails: `focus position must differ from disc position`.
    ///
    pub(super) fn disc_basis(
        &self,
        center: (f64, f64, f64),
        focus_position: Option<(f64, f64, f64)>,
    ) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let normal = if let Some((fx, fy, fz)) = focus_position {
            let nx = fx - center.0;
            let ny = fy - center.1;
            let nz = fz - center.2;
            let norm = nz.mul_add(nz, nx.mul_add(nx, ny * ny)).sqrt();
            assert!(
                norm > DISC_AXIS_EPSILON,
                "focus position must differ from disc position"
            );
            [nx / norm, ny / norm, nz / norm]
        } else {
            [0.0, 0.0, 1.0]
        };

        let reference: [f64; 3] = if normal[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let mut u = [
            reference[1].mul_add(normal[2], -(reference[2] * normal[1])),
            reference[2].mul_add(normal[0], -(reference[0] * normal[2])),
            reference[0].mul_add(normal[1], -(reference[1] * normal[0])),
        ];
        let u_norm = u[2].mul_add(u[2], u[0].mul_add(u[0], u[1] * u[1])).sqrt();
        if u_norm <= DISC_AXIS_EPSILON {
            u = [1.0, 0.0, 0.0];
        } else {
            u[0] /= u_norm;
            u[1] /= u_norm;
            u[2] /= u_norm;
        }
        let v = [
            normal[1].mul_add(u[2], -(normal[2] * u[1])),
            normal[2].mul_add(u[0], -(normal[0] * u[2])),
            normal[0].mul_add(u[1], -(normal[1] * u[0])),
        ];
        (u, v, normal)
    }
}
