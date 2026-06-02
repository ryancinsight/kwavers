//! Rasterizers for planar elements: rect and disc; plus the per-element dispatcher.
//!
//! # Algorithm — Rectangle (sampled lattice + Euler rotation)
//!
//! The rectangle is treated as a planar surface element, matching the
//! upstream k-wave-python `make_cart_rect` contract. An evenly spaced lattice
//! on the canonical `[−1,1]² × {0}` rectangle is sampled, rotated by the
//! intrinsic X-Y-Z Euler angles, and mapped through the BLI stencil.
//!
//! # Algorithm — Disc (oriented Fibonacci/Archimedean sampling)
//!
//! Let `c` be the disc center and `f` an optional focus point. The unit normal
//! is `n = (f − c) / ‖f − c‖` (or `e_z` when `f` is absent). Disc samples
//! are computed on the orthonormal basis `(u, v)` spanning the plane
//! perpendicular to `n`:
//! ```text
//! p(r, θ) = c + r·cos(θ)·u + r·sin(θ)·v
//! ```
//! using a radial Fibonacci spiral. Each sample contributes one unit of
//! occupancy (binary) or a constant BLI weight (weighted), so that the total
//! discrete mass equals `A / dx²` up to round-off.

use super::math::{apply_matrix, euler_xyz_rotation_matrix, DISC_SAMPLE_UPSAMPLING_RATE};
use super::{ElementShape, KWaveArray};
use kwavers_core::constants::numerical::{TWO_PI};

impl KWaveArray {
    // ─── Rect ──────────────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub(super) fn rasterize_rect(
        &self,
        mask: &mut ndarray::Array3<bool>,
        grid: &crate::grid::Grid,
        center: (f64, f64, f64),
        width: f64,
        height: f64,
        length: f64,
        euler_xyz_deg: (f64, f64, f64),
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_rect_points(
            grid,
            center,
            width,
            height,
            euler_xyz_deg,
            |point, scale| {
                self.map_surface_sample(
                    grid,
                    &x_vec,
                    &y_vec,
                    &z_vec,
                    point,
                    scale,
                    true,
                    |ix, iy, iz, _| {
                        mask[[ix, iy, iz]] = true;
                    },
                );
            },
        );
        let _ = length;
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn rasterize_rect_weighted(
        &self,
        mask: &mut ndarray::Array3<f64>,
        grid: &crate::grid::Grid,
        center: (f64, f64, f64),
        width: f64,
        height: f64,
        length: f64,
        euler_xyz_deg: (f64, f64, f64),
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_rect_points(
            grid,
            center,
            width,
            height,
            euler_xyz_deg,
            |point, scale| {
                self.map_surface_sample(
                    grid,
                    &x_vec,
                    &y_vec,
                    &z_vec,
                    point,
                    scale,
                    false,
                    |ix, iy, iz, weight| {
                        mask[[ix, iy, iz]] += weight;
                    },
                );
            },
        );
        let _ = length;
    }

    /// Emit the canonical rectangle integration lattice used by k-wave-python.
    pub(super) fn rasterize_rect_points<F>(
        &self,
        grid: &crate::grid::Grid,
        center: (f64, f64, f64),
        width: f64,
        height: f64,
        euler_xyz_deg: (f64, f64, f64),
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let area = width * height;
        let m_grid = area / (grid.dx * grid.dx);
        let target_points = (m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let npts_x = ((target_points as f64 * width / height).sqrt().ceil() as usize).max(1);
        let npts_y = ((target_points as f64 / npts_x as f64).ceil() as usize).max(1);
        let num_points = npts_x * npts_y;
        let scale = m_grid / num_points as f64;
        let rot = euler_xyz_rotation_matrix(euler_xyz_deg);

        let dx = if npts_x <= 1 {
            0.0
        } else {
            2.0 / npts_x as f64
        };
        let dy = if npts_y <= 1 {
            0.0
        } else {
            2.0 / npts_y as f64
        };

        for ix in 0..npts_x {
            let ux = -1.0 + dx / 2.0 + dx * ix as f64;
            let lx = ux * width / 2.0;
            for iy in 0..npts_y {
                let uy = -1.0 + dy / 2.0 + dy * iy as f64;
                let ly = uy * height / 2.0;
                let (rx, ry, rz) = apply_matrix(&rot, (lx, ly, 0.0));
                visit([center.0 + rx, center.1 + ry, center.2 + rz], scale);
            }
        }
    }

    // ─── Disc ──────────────────────────────────────────────────────────────

    pub(super) fn rasterize_disc(
        &self,
        mask: &mut ndarray::Array3<bool>,
        grid: &crate::grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_disc_points(grid, center, diameter, focus_position, |point, scale| {
            self.map_surface_sample(
                grid,
                &x_vec,
                &y_vec,
                &z_vec,
                point,
                scale,
                true,
                |ix, iy, iz, _| {
                    mask[[ix, iy, iz]] = true;
                },
            );
        });
    }

    pub(super) fn rasterize_disc_weighted(
        &self,
        mask: &mut ndarray::Array3<f64>,
        grid: &crate::grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_disc_points(grid, center, diameter, focus_position, |point, scale| {
            self.map_surface_sample(
                grid,
                &x_vec,
                &y_vec,
                &z_vec,
                point,
                scale,
                false,
                |ix, iy, iz, weight| {
                    mask[[ix, iy, iz]] += weight;
                },
            );
        });
    }

    /// Emit oriented Fibonacci disc samples.
    pub(super) fn rasterize_disc_points<F>(
        &self,
        grid: &crate::grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let radius = diameter / 2.0;
        let area = std::f64::consts::PI * radius * radius;
        let m_grid = area / (grid.dx * grid.dx);
        let target_points = (m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let num_radial =
            ((target_points as f64 / std::f64::consts::PI).sqrt().ceil() as usize).max(1);
        let num_points = self.disc_sample_count(num_radial);
        let scale = m_grid / num_points as f64;
        let (u, v, _) = self.disc_basis(center, focus_position);

        let mut emit = |x_local: f64, y_local: f64| {
            let point = [
                y_local.mul_add(v[0], x_local.mul_add(u[0], center.0)),
                y_local.mul_add(v[1], x_local.mul_add(u[1], center.1)),
                y_local.mul_add(v[2], x_local.mul_add(u[2], center.2)),
            ];
            visit(point, scale);
        };

        emit(0.0, 0.0);
        if num_radial == 1 {
            return;
        }

        let radial_denom = (num_radial - 1) as f64;
        let radial_step = (radius - radius / (2.0 * radial_denom)) / radial_denom;

        for ring_idx in 1..num_radial {
            let ring_radius = ring_idx as f64 * radial_step;
            let num_theta = ((ring_idx as f64) * super::math::DISC_PACKING_NUMBER)
                .round()
                .max(1.0) as usize;
            for theta_idx in 0..num_theta {
                let theta = TWO_PI * theta_idx as f64 / num_theta as f64;
                emit(ring_radius * theta.cos(), ring_radius * theta.sin());
            }
        }
    }

    // ─── Per-element dispatcher ────────────────────────────────────────────

    /// Rasterize a single element onto a pre-allocated weighted mask.
    ///
    /// Same dispatch as `get_array_weighted_mask` but for one element — used
    /// by `build_per_element_source` to build per-element BLI masks.
    pub(super) fn rasterize_element_weighted(
        &self,
        element: &ElementShape,
        mask: &mut ndarray::Array3<f64>,
        grid: &crate::grid::Grid,
    ) {
        match element {
            ElementShape::Bowl {
                position,
                radius,
                diameter,
            } => {
                self.rasterize_bowl_weighted(mask, grid, *position, *radius, *diameter);
            }
            ElementShape::Disc {
                position,
                diameter,
                focus_position,
            } => {
                self.rasterize_disc_weighted(mask, grid, *position, *diameter, *focus_position);
            }
            ElementShape::Arc {
                position,
                radius,
                diameter,
                start_angle,
                end_angle,
            } => {
                self.rasterize_arc_weighted(
                    mask,
                    grid,
                    *position,
                    *radius,
                    *diameter,
                    *start_angle,
                    *end_angle,
                );
            }
            ElementShape::Rect {
                position,
                width,
                height,
                length,
                euler_xyz_deg,
            } => {
                let (pos_eff, euler_eff) = self.apply_transform_rect(*position, *euler_xyz_deg);
                self.rasterize_rect_weighted(
                    mask, grid, pos_eff, *width, *height, *length, euler_eff,
                );
            }
            ElementShape::Annulus {
                position,
                radius,
                inner_diameter,
                outer_diameter,
            } => {
                self.rasterize_annulus_weighted(
                    mask,
                    grid,
                    *position,
                    *radius,
                    *inner_diameter,
                    *outer_diameter,
                );
            }
        }
    }
}
