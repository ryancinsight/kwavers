//! Rasterizers for planar elements: rect, disc and validated planar apertures;
//! plus the per-element dispatcher.
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

use super::math::DISC_SAMPLE_UPSAMPLING_RATE;
use super::{DiscSourceProfile, ElementShape, KWaveArray, KWaveElement};
use crate::transducers::physics::{PlanarApertureGeometry, PlanarApertureShape, SurfaceConvention};
use eunomia::convert::IntegerTarget;
use eunomia::FloatElement;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array3;

impl KWaveArray {
    // ─── Rect ──────────────────────────────────────────────────────────────

    /// Emit the rect-canonical integration lattice used by k-wave-python
    /// (ADR 136): the `[-1,1]^2` half-cell-offset lattice mapped through the
    /// element frame, with the legacy `area/dx^2` sample budget.
    ///
    /// `first`/`second` are the orthonormal in-plane axes (width/height
    /// directions). For a frame built from an Euler rotation this evaluates
    /// the same lattice as the retired Euler-driven path; sample positions
    /// may differ by floating-point reassociation, and the k-Wave reference
    /// mass test pins the values.
    /// Counts convert through the conversion module, never bare `as`.
    pub(super) fn rasterize_rect_lattice_points<F>(
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        width: f64,
        height: f64,
        first: [f64; 3],
        second: [f64; 3],
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let area = width * height;
        let m_grid = area / (grid.dx * grid.dx);
        let target_points =
            usize::from_truncated((m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0));
        let npts_x = usize::from_truncated(
            (f64::from_count(target_points) * width / height)
                .sqrt()
                .ceil(),
        )
        .max(1);
        let npts_y = usize::from_truncated(
            (f64::from_count(target_points) / f64::from_count(npts_x)).ceil(),
        )
        .max(1);
        let num_points = npts_x * npts_y;
        let scale = m_grid / f64::from_count(num_points);

        let dx = if npts_x <= 1 {
            0.0
        } else {
            2.0 / f64::from_count(npts_x)
        };
        let dy = if npts_y <= 1 {
            0.0
        } else {
            2.0 / f64::from_count(npts_y)
        };

        for ix in 0..npts_x {
            let ux = -1.0 + dx / 2.0 + dx * f64::from_count(ix);
            let lx = ux * width / 2.0;
            for iy in 0..npts_y {
                let uy = -1.0 + dy / 2.0 + dy * f64::from_count(iy);
                let ly = uy * height / 2.0;
                visit(
                    [
                        center.0 + lx * first[0] + ly * second[0],
                        center.1 + lx * first[1] + ly * second[1],
                        center.2 + lx * first[2] + ly * second[2],
                    ],
                    scale,
                );
            }
        }
    }

    // ─── Disc ──────────────────────────────────────────────────────────────

    pub(super) fn rasterize_disc(
        mask: &mut Array3<bool>,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        Self::rasterize_disc_points(grid, center, diameter, focus_position, |point, scale| {
            Self::map_surface_sample(
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

    pub(super) fn rasterize_profiled_disc_weighted(
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
        Self::rasterize_profiled_disc_points(
            grid,
            center,
            diameter,
            focus_position,
            profile,
            |point, scale| {
                Self::map_surface_sample(
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
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
        visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        Self::rasterize_profiled_disc_points(
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
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
        profile: DiscSourceProfile,
        visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let (u, v, _) = Self::disc_basis(center, focus_position);
        Self::rasterize_framed_disc_points(grid, center, diameter / 2.0, u, v, profile, visit);
    }

    /// Emit the disc spiral on an explicit orthonormal frame (ADR 136
    /// follow-up): the k-wave-python radial-Fibonacci lattice with its
    /// centre sample, mapped through `(u, v)` about `center`, under the
    /// legacy `area/dx^2` sample budget. Counts convert through the
    /// conversion module, never bare `as`.
    pub(super) fn rasterize_framed_disc_points<F>(
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        u: [f64; 3],
        v: [f64; 3],
        profile: DiscSourceProfile,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let area = std::f64::consts::PI * radius * radius;
        let m_grid = area / (grid.dx * grid.dx);
        let target_points =
            usize::from_truncated((m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0));
        let num_radial = usize::from_truncated(
            (f64::from_count(target_points) / std::f64::consts::PI)
                .sqrt()
                .ceil(),
        )
        .max(1);
        let num_points = Self::disc_sample_count(num_radial);
        let scale = m_grid / f64::from_count(num_points);

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

        let radial_denom = f64::from_count(num_radial - 1);
        let radial_step = (radius - radius / (2.0 * radial_denom)) / radial_denom;

        for ring_idx in 1..num_radial {
            let ring_radius = f64::from_count(ring_idx) * radial_step;
            let num_theta = usize::from_truncated(
                (f64::from_count(ring_idx) * super::math::DISC_PACKING_NUMBER)
                    .round()
                    .max(1.0),
            );
            for theta_idx in 0..num_theta {
                let theta = TWO_PI * f64::from_count(theta_idx) / f64::from_count(num_theta);
                emit(ring_radius * theta.cos(), ring_radius * theta.sin());
            }
        }
    }

    // ─── Validated planar aperture ─────────────────────────────────────────

    /// Geometry after the installed array transform: the centre always
    /// takes the full rigid motion while the orientation follows the
    /// element's surface convention (centre-only under `KWave`, rigid
    /// under `AreaConserving`).
    fn effective_geometry(&self, geometry: PlanarApertureGeometry) -> PlanarApertureGeometry {
        self.apply_transform_planar(geometry, geometry.surface())
    }

    pub(super) fn rasterize_planar_aperture(
        &self,
        mask: &mut Array3<bool>,
        grid: &kwavers_grid::Grid,
        geometry: PlanarApertureGeometry,
    ) {
        let geometry = self.effective_geometry(geometry);
        let [cx, cy, cz] = geometry.center_m();
        let center = (cx, cy, cz);
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        if let (SurfaceConvention::KWave, PlanarApertureShape::Rectangle { width_m, height_m }) =
            (geometry.surface(), geometry.shape())
        {
            let second = Self::second_axis(geometry.normal(), geometry.first_axis());
            Self::rasterize_rect_lattice_points(
                grid,
                center,
                width_m,
                height_m,
                geometry.first_axis(),
                second,
                |point, scale| {
                    Self::map_surface_sample(
                        grid,
                        &x_vec,
                        &y_vec,
                        &z_vec,
                        point,
                        scale,
                        true,
                        |i, j, k, _| mask[[i, j, k]] = true,
                    );
                },
            );
            return;
        }
        if let (SurfaceConvention::KWave, PlanarApertureShape::Disk { radius_m }) =
            (geometry.surface(), geometry.shape())
        {
            let second = Self::second_axis(geometry.normal(), geometry.first_axis());
            Self::rasterize_framed_disc_points(
                grid,
                center,
                radius_m,
                geometry.first_axis(),
                second,
                DiscSourceProfile::uniform(),
                |point, scale| {
                    Self::map_surface_sample(
                        grid,
                        &x_vec,
                        &y_vec,
                        &z_vec,
                        point,
                        scale,
                        true,
                        |i, j, k, _| mask[[i, j, k]] = true,
                    );
                },
            );
            return;
        }
        Self::rasterize_planar_aperture_points(grid, geometry, center, |point, scale| {
            Self::map_surface_sample(
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
        let geometry = self.effective_geometry(geometry);
        let [cx, cy, cz] = geometry.center_m();
        let center = (cx, cy, cz);
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        if let (SurfaceConvention::KWave, PlanarApertureShape::Rectangle { width_m, height_m }) =
            (geometry.surface(), geometry.shape())
        {
            let second = Self::second_axis(geometry.normal(), geometry.first_axis());
            Self::rasterize_rect_lattice_points(
                grid,
                center,
                width_m,
                height_m,
                geometry.first_axis(),
                second,
                |point, scale| {
                    Self::map_surface_sample(
                        grid,
                        &x_vec,
                        &y_vec,
                        &z_vec,
                        point,
                        scale,
                        false,
                        |i, j, k, weight| mask[[i, j, k]] += weight,
                    );
                },
            );
            return;
        }
        if let (SurfaceConvention::KWave, PlanarApertureShape::Disk { radius_m }) =
            (geometry.surface(), geometry.shape())
        {
            let second = Self::second_axis(geometry.normal(), geometry.first_axis());
            Self::rasterize_framed_disc_points(
                grid,
                center,
                radius_m,
                geometry.first_axis(),
                second,
                DiscSourceProfile::uniform(),
                |point, scale| {
                    Self::map_surface_sample(
                        grid,
                        &x_vec,
                        &y_vec,
                        &z_vec,
                        point,
                        scale,
                        false,
                        |i, j, k, weight| mask[[i, j, k]] += weight,
                    );
                },
            );
            return;
        }
        Self::rasterize_planar_aperture_points(grid, geometry, center, |point, scale| {
            Self::map_area_conserving_surface_sample(
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
        Self::map_surface_sample(
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
        Self::map_surface_sample(
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

    /// Second in-plane axis of the stored frame: `normal × first_axis`.
    fn second_axis(normal: [f64; 3], first: [f64; 3]) -> [f64; 3] {
        [
            normal[1].mul_add(first[2], -(normal[2] * first[1])),
            normal[2].mul_add(first[0], -(normal[0] * first[2])),
            normal[0].mul_add(first[1], -(normal[1] * first[0])),
        ]
    }

    /// Emit equal-area surface samples whose total BLI mass is `area / dx^2`:
    /// a polar lattice for disks and annular sectors, a rectangular midpoint
    /// lattice with near-square cells for rectangles.
    ///
    /// `center` is the effective (array-transformed) piston centre; callers
    /// apply the installed array transform before emitting.
    pub(super) fn rasterize_planar_aperture_points<F>(
        grid: &kwavers_grid::Grid,
        geometry: PlanarApertureGeometry,
        center: (f64, f64, f64),
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let shape = geometry.shape();
        let mass = shape.area_m2() / (grid.dx * grid.dy);
        let target = (mass * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let first = geometry.first_axis();
        let normal = geometry.normal();
        let second = Self::second_axis(normal, first);
        let mut emit = |x: f64, y: f64, scale: f64| {
            visit(
                [
                    y.mul_add(second[0], x.mul_add(first[0], center.0)),
                    y.mul_add(second[1], x.mul_add(first[1], center.1)),
                    y.mul_add(second[2], x.mul_add(first[2], center.2)),
                ],
                scale,
            );
        };

        if let PlanarApertureShape::Rectangle { width_m, height_m } = shape {
            let width_count =
                usize::from_truncated((f64::from_count(target) * width_m / height_m).sqrt().ceil())
                    .clamp(1, target);
            let height_count = target.div_ceil(width_count);
            let scale = mass / f64::from_count(width_count * height_count);
            for width_index in 0..width_count {
                let x = width_m
                    * ((f64::from_count(width_index) + 0.5) / f64::from_count(width_count) - 0.5);
                for height_index in 0..height_count {
                    let y = height_m
                        * ((f64::from_count(height_index) + 0.5) / f64::from_count(height_count)
                            - 0.5);
                    emit(x, y, scale);
                }
            }
            return;
        }

        let (inner, outer, start, span) = shape
            .polar_bounds()
            .expect("invariant: every non-rectangular planar shape is polar");
        let radial_width = outer - inner;
        let mean_radius =
            2.0 * (outer.powi(3) - inner.powi(3)) / (3.0 * (outer * outer - inner * inner));
        let angular_length = (mean_radius * span).max(f64::MIN_POSITIVE);
        let radial_count = ((target as f64 * radial_width / angular_length)
            .sqrt()
            .ceil() as usize)
            .clamp(1, target);
        let angular_count = target.div_ceil(radial_count);
        let scale = mass / f64::from_count(radial_count * angular_count);
        let squared_span = outer * outer - inner * inner;
        for radial_index in 0..radial_count {
            let radial_fraction = (radial_index as f64 + 0.5) / radial_count as f64;
            let radius = (inner * inner + radial_fraction * squared_span).sqrt();
            for angular_index in 0..angular_count {
                let angle = start + span * (angular_index as f64 + 0.5) / angular_count as f64;
                emit(radius * angle.cos(), radius * angle.sin(), scale);
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
            // The single-element path honors the stored surface convention,
            // exactly like the full-mask weighted path (ADR 136).
            let geometry = self.effective_geometry(*geometry);
            let [cx, cy, cz] = geometry.center_m();
            let center = (cx, cy, cz);
            if let (
                SurfaceConvention::KWave,
                PlanarApertureShape::Rectangle { width_m, height_m },
            ) = (geometry.surface(), geometry.shape())
            {
                let second = Self::second_axis(geometry.normal(), geometry.first_axis());
                Self::rasterize_rect_lattice_points(
                    grid,
                    center,
                    width_m,
                    height_m,
                    geometry.first_axis(),
                    second,
                    |point, scale| {
                        Self::map_surface_sample(
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
                    },
                );
                return;
            }
            if let (SurfaceConvention::KWave, PlanarApertureShape::Disk { radius_m }) =
                (geometry.surface(), geometry.shape())
            {
                let second = Self::second_axis(geometry.normal(), geometry.first_axis());
                Self::rasterize_framed_disc_points(
                    grid,
                    center,
                    radius_m,
                    geometry.first_axis(),
                    second,
                    DiscSourceProfile::uniform(),
                    |point, scale| {
                        Self::map_surface_sample(
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
                    },
                );
                return;
            }
            Self::rasterize_planar_aperture_points(grid, geometry, center, |point, scale| {
                Self::map_area_conserving_surface_sample(
                    grid, &x_vec, &y_vec, &z_vec, point, scale, &mut visit,
                );
            });
            return;
        }
        let KWaveElement::Shape(element) = element else {
            unreachable!("invariant: planar apertures return before shape dispatch")
        };
        let mut visit_point = |point, scale| {
            Self::map_surface_sample(
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
                Self::rasterize_bowl_points(grid, *position, *radius, *diameter, &mut visit_point);
            }
            ElementShape::ProfiledDisc {
                position,
                diameter,
                focus_position,
                profile,
            } => {
                Self::rasterize_profiled_disc_points(
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
                Self::rasterize_arc_points(
                    grid,
                    *position,
                    *radius,
                    *start_angle,
                    *end_angle,
                    &mut visit_point,
                );
            }
            ElementShape::Annulus {
                position,
                radius,
                inner_diameter,
                outer_diameter,
            } => {
                Self::rasterize_annulus_points(
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
