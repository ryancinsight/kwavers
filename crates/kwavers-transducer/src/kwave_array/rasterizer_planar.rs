//! Rasterizers for planar elements: discs and validated planar apertures
//! (rectangle, disk, annular sector); plus the per-element dispatcher.
//!
//! # Algorithm — Planar aperture (equal-area midpoint lattice)
//!
//! Every rectangle, including the k-Wave-named `add_rect_*` elements, is a
//! planar aperture (ADR 136). A rectangle is sampled on a midpoint lattice of
//! `ceil(10·A/(dx·dy))` points split by aspect ratio, placed on the aperture's
//! first axis and `normal × first_axis`; a single sample along an axis sits at
//! the centre. A sample deposits either the unnormalised BLI stencil
//! ([`SurfaceConvention::KWave`], k-wave-python's contract) or the stencil
//! normalised to unit sum ([`SurfaceConvention::AreaConserving`]).
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
use super::{DiscSourceProfile, ElementShape, KWaveArray, KWaveElement, SurfaceConvention};
use crate::transducers::physics::{PlanarApertureGeometry, PlanarApertureShape};
use eunomia::convert::IntegerTarget;
use eunomia::FloatElement;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array3;

impl KWaveArray {
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

    pub(super) fn rasterize_disc_weighted(
        mask: &mut Array3<f64>,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) {
        Self::rasterize_profiled_disc_weighted(
            mask,
            grid,
            center,
            diameter,
            focus_position,
            DiscSourceProfile::uniform(),
        );
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
        let num_points = Self::disc_sample_count(num_radial);
        let scale = m_grid / num_points as f64;
        let (u, v, _) = Self::disc_basis(center, focus_position);

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
        mask: &mut Array3<bool>,
        grid: &kwavers_grid::Grid,
        geometry: PlanarApertureGeometry,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        Self::rasterize_planar_aperture_points(grid, geometry, |point, scale| {
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
        mask: &mut Array3<f64>,
        grid: &kwavers_grid::Grid,
        geometry: PlanarApertureGeometry,
        convention: SurfaceConvention,
    ) {
        Self::visit_planar_aperture_weighted_cells(
            grid,
            geometry,
            convention,
            |i, j, k, weight| mask[[i, j, k]] += weight,
        );
    }

    /// Visit every weighted grid-cell contribution of one planar aperture,
    /// depositing each sample under its [`SurfaceConvention`].
    pub(super) fn visit_planar_aperture_weighted_cells<F>(
        grid: &kwavers_grid::Grid,
        geometry: PlanarApertureGeometry,
        convention: SurfaceConvention,
        mut visit: F,
    ) where
        F: FnMut(usize, usize, usize, f64),
    {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        match convention {
            SurfaceConvention::KWave => {
                Self::rasterize_planar_aperture_points(grid, geometry, |point, scale| {
                    Self::map_surface_sample(
                        grid, &x_vec, &y_vec, &z_vec, point, scale, false, &mut visit,
                    );
                });
            }
            SurfaceConvention::AreaConserving => {
                Self::rasterize_planar_aperture_points(grid, geometry, |point, scale| {
                    Self::map_area_conserving_surface_sample(
                        grid, &x_vec, &y_vec, &z_vec, point, scale, &mut visit,
                    );
                });
            }
        }
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

    /// Emit equal-area surface samples whose total BLI mass is `area / (dx·dy)`,
    /// the area counted in cells of the xy plane (equal to `area / dx²` on an
    /// isotropic grid): a polar lattice for disks and annular sectors, a
    /// rectangular midpoint lattice with near-square cells for rectangles.
    pub(super) fn rasterize_planar_aperture_points<F>(
        grid: &kwavers_grid::Grid,
        geometry: PlanarApertureGeometry,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let shape = geometry.shape();
        let mass = shape.area_m2() / (grid.dx * grid.dy);
        let target = (mass * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let center = geometry.center_m();
        let first = geometry.first_axis();
        let normal = geometry.normal();
        let second = [
            normal[1].mul_add(first[2], -(normal[2] * first[1])),
            normal[2].mul_add(first[0], -(normal[0] * first[2])),
            normal[0].mul_add(first[1], -(normal[1] * first[0])),
        ];
        let mut emit = |x: f64, y: f64, scale: f64| {
            visit(
                [
                    y.mul_add(second[0], x.mul_add(first[0], center[0])),
                    y.mul_add(second[1], x.mul_add(first[1], center[1])),
                    y.mul_add(second[2], x.mul_add(first[2], center[2])),
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
        if let KWaveElement::PlanarAperture {
            geometry,
            convention,
        } = element
        {
            let geometry = self.apply_transform_planar(*geometry, *convention);
            Self::visit_planar_aperture_weighted_cells(grid, geometry, *convention, visit);
            return;
        }
        let KWaveElement::Shape(element) = element else {
            unreachable!("invariant: planar apertures return before shape dispatch")
        };
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
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
            ElementShape::Disc {
                position,
                diameter,
                focus_position,
            } => {
                Self::rasterize_disc_points(
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
