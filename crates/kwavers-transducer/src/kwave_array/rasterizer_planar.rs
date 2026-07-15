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
use super::{DiscSourceProfile, ElementShape, KWaveArray, KWaveElement};
use crate::transducers::physics::PlanarApertureGeometry;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array3;

impl KWaveArray {
    // ─── Rect ──────────────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub(super) fn rasterize_rect(
        &self,
        mask: &mut Array3<bool>,
        grid: &kwavers_grid::Grid,
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
        mask: &mut Array3<f64>,
        grid: &kwavers_grid::Grid,
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
        grid: &kwavers_grid::Grid,
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
        mask: &mut Array3<bool>,
        grid: &kwavers_grid::Grid,
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
        mask: &mut Array3<f64>,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) {
        self.rasterize_profiled_disc_weighted(
            mask,
            grid,
            center,
            diameter,
            focus_position,
            DiscSourceProfile::uniform(),
        );
    }

    pub(super) fn rasterize_profiled_disc_weighted(
        &self,
        mask: &mut Array3<f64>,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
        profile: DiscSourceProfile,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_profiled_disc_points(
            grid,
            center,
            diameter,
            focus_position,
            profile,
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
    }

    /// Emit oriented Fibonacci disc samples.
    pub(super) fn rasterize_disc_points<F>(
        &self,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
        visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        self.rasterize_profiled_disc_points(
            grid,
            center,
            diameter,
            focus_position,
            DiscSourceProfile::uniform(),
            visit,
        );
    }

    /// Emit oriented Fibonacci disc samples with a finite-source profile.
    pub(super) fn rasterize_profiled_disc_points<F>(
        &self,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
        profile: DiscSourceProfile,
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
            let radius_fraction = x_local.hypot(y_local) / radius.max(f64::MIN_POSITIVE);
            visit(
                point,
                scale * profile.weight_at_normalized_radius(radius_fraction),
            );
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

    // ─── Validated planar aperture ─────────────────────────────────────────

    pub(super) fn rasterize_planar_aperture(
        &self,
        mask: &mut Array3<bool>,
        grid: &kwavers_grid::Grid,
        geometry: PlanarApertureGeometry,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_planar_aperture_points(grid, geometry, |point, scale| {
            self.map_surface_sample(
                grid,
                &x_vec,
                &y_vec,
                &z_vec,
                point,
                scale,
                true,
                |i, j, k, _| mask[[i, j, k]] = true,
            );
        });
    }

    pub(super) fn rasterize_planar_aperture_weighted(
        &self,
        mask: &mut Array3<f64>,
        grid: &kwavers_grid::Grid,
        geometry: PlanarApertureGeometry,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_planar_aperture_points(grid, geometry, |point, scale| {
            self.map_area_conserving_surface_sample(
                grid,
                &x_vec,
                &y_vec,
                &z_vec,
                point,
                scale,
                |i, j, k, weight| mask[[i, j, k]] += weight,
            );
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn map_area_conserving_surface_sample<F>(
        &self,
        grid: &kwavers_grid::Grid,
        x_vec: &leto::Array1<f64>,
        y_vec: &leto::Array1<f64>,
        z_vec: &leto::Array1<f64>,
        point: [f64; 3],
        scale: f64,
        visit: F,
    ) where
        F: FnMut(usize, usize, usize, f64),
    {
        let mut kernel_sum = 0.0;
        self.map_surface_sample(
            grid,
            x_vec,
            y_vec,
            z_vec,
            point,
            1.0,
            false,
            |_, _, _, weight| kernel_sum += weight,
        );
        if kernel_sum == 0.0 {
            return;
        }
        self.map_surface_sample(
            grid,
            x_vec,
            y_vec,
            z_vec,
            point,
            scale / kernel_sum,
            false,
            visit,
        );
    }

    /// Emit equal-area polar samples whose total BLI mass is `area / dx^2`.
    pub(super) fn rasterize_planar_aperture_points<F>(
        &self,
        grid: &kwavers_grid::Grid,
        geometry: PlanarApertureGeometry,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let (inner, outer, start, span) = geometry.shape().radial_and_angular_bounds();
        let area = geometry.shape().area_m2();
        let grid_area = grid.dx * grid.dy;
        let mass = area / grid_area;
        let target = (mass * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let radial_width = outer - inner;
        let mean_radius =
            2.0 * (outer.powi(3) - inner.powi(3)) / (3.0 * (outer * outer - inner * inner));
        let angular_length = (mean_radius * span).max(f64::MIN_POSITIVE);
        let radial_count = ((target as f64 * radial_width / angular_length)
            .sqrt()
            .ceil() as usize)
            .clamp(1, target);
        let angular_count = target.div_ceil(radial_count);
        let sample_count = radial_count * angular_count;
        let scale = mass / sample_count as f64;
        let squared_span = outer * outer - inner * inner;
        let center = geometry.center_m();
        let first = geometry.first_axis();
        let normal = geometry.normal();
        let second = [
            normal[1].mul_add(first[2], -(normal[2] * first[1])),
            normal[2].mul_add(first[0], -(normal[0] * first[2])),
            normal[0].mul_add(first[1], -(normal[1] * first[0])),
        ];

        for radial_index in 0..radial_count {
            let radial_fraction = (radial_index as f64 + 0.5) / radial_count as f64;
            let radius = (inner * inner + radial_fraction * squared_span).sqrt();
            for angular_index in 0..angular_count {
                let angle = start + span * (angular_index as f64 + 0.5) / angular_count as f64;
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                visit(
                    [
                        y.mul_add(second[0], x.mul_add(first[0], center[0])),
                        y.mul_add(second[1], x.mul_add(first[1], center[1])),
                        y.mul_add(second[2], x.mul_add(first[2], center[2])),
                    ],
                    scale,
                );
            }
        }
    }

    // ─── Per-element dispatcher ────────────────────────────────────────────

    /// Emit BLI-weighted grid-cell contributions for a single element.
    ///
    /// This preserves each element's weighted-mask sampling and BLI stencil
    /// without materializing a full-grid temporary mask for each element.
    pub(super) fn rasterize_element_weighted_cells<F>(
        &self,
        element: &KWaveElement,
        grid: &kwavers_grid::Grid,
        mut visit: F,
    ) where
        F: FnMut(usize, usize, usize, f64),
    {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        if let KWaveElement::PlanarAperture(geometry) = element {
            self.rasterize_planar_aperture_points(grid, *geometry, |point, scale| {
                self.map_area_conserving_surface_sample(
                    grid, &x_vec, &y_vec, &z_vec, point, scale, &mut visit,
                );
            });
            return;
        }
        let KWaveElement::Shape(element) = element else {
            unreachable!("invariant: planar apertures return before shape dispatch")
        };
        let mut visit_point = |point, scale| {
            self.map_surface_sample(
                grid,
                &x_vec,
                &y_vec,
                &z_vec,
                point,
                scale,
                false,
                |ix, iy, iz, weight| {
                    visit(ix, iy, iz, weight);
                },
            );
        };

        match element {
            ElementShape::Bowl {
                position,
                radius,
                diameter,
            } => {
                self.rasterize_bowl_points(grid, *position, *radius, *diameter, &mut visit_point);
            }
            ElementShape::Disc {
                position,
                diameter,
                focus_position,
            } => {
                self.rasterize_disc_points(
                    grid,
                    *position,
                    *diameter,
                    *focus_position,
                    &mut visit_point,
                );
            }
            ElementShape::ProfiledDisc {
                position,
                diameter,
                focus_position,
                profile,
            } => {
                self.rasterize_profiled_disc_points(
                    grid,
                    *position,
                    *diameter,
                    *focus_position,
                    *profile,
                    &mut visit_point,
                );
            }
            ElementShape::Arc {
                position,
                radius,
                diameter: _,
                start_angle,
                end_angle,
            } => {
                self.rasterize_arc_points(
                    grid,
                    *position,
                    *radius,
                    *start_angle,
                    *end_angle,
                    &mut visit_point,
                );
            }
            ElementShape::Rect {
                position,
                width,
                height,
                length: _,
                euler_xyz_deg,
            } => {
                let (pos_eff, euler_eff) = self.apply_transform_rect(*position, *euler_xyz_deg);
                self.rasterize_rect_points(
                    grid,
                    pos_eff,
                    *width,
                    *height,
                    euler_eff,
                    &mut visit_point,
                );
            }
            ElementShape::Annulus {
                position,
                radius,
                inner_diameter,
                outer_diameter,
            } => {
                self.rasterize_annulus_points(
                    grid,
                    *position,
                    *radius,
                    *inner_diameter,
                    *outer_diameter,
                    &mut visit_point,
                );
            }
        }
    }
}
