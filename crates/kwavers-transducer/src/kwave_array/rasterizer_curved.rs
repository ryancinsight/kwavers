//! Rasterizers for curved elements: arc, bowl, and annulus.
//!
//! Each element type provides three variants:
//! - `rasterize_*`: fills a boolean mask (one call per surface sample).
//! - `rasterize_*_weighted`: accumulates BLI weights into a `f64` mask.
//! - `rasterize_*_points`: iterates canonical surface samples, calling a
//!   visitor `F: FnMut([f64; 3], f64)` with `(point, scale)`.
//!
//! # Algorithm — Arc (line-sampled BLI)
//!
//! Arc length `L = R·|Δθ|` is sampled at half-step angular offsets. Each
//! sample is mapped through the BLI stencil in [`bli_kernel`].
//!
//! # Algorithm — Bowl (canonical golden-angle spiral)
//!
//! The spherical cap is sampled with a golden-angle spiral, matching
//! k-wave-python's `make_cart_bowl`. Points lie on the cap surface:
//! `p = [c_x − R·cos(φ), c_y + R·sin(φ)·sin(θ), c_z + R·sin(φ)·cos(θ)]`.
//!
//! # Algorithm — Annulus (ring segment of the bowl spiral)
//!
//! Matches k-wave-python's `make_cart_spherical_segment` parameterization:
//! step size `C` is set as if sampling the full bowl, then `t_start` is the
//! first index where `φ ≥ φ_min` (inner aperture boundary). This keeps the
//! golden-angle phase consistent with the bowl for bowl+annulus composition.

use super::math::{DISC_SAMPLE_UPSAMPLING_RATE, GOLDEN_ANGLE};
use leto::Array3;
use super::KWaveArray;
use kwavers_core::constants::numerical::TWO_PI;

impl KWaveArray {
    // ─── Arc ───────────────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub(super) fn rasterize_arc(
        &self,
        mask: &mut Array3<bool>,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        _diameter: f64,
        start_angle: f64,
        end_angle: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_arc_points(
            grid,
            center,
            radius,
            start_angle,
            end_angle,
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
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn rasterize_arc_weighted(
        &self,
        mask: &mut Array3<f64>,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        _diameter: f64,
        start_angle: f64,
        end_angle: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_arc_points(
            grid,
            center,
            radius,
            start_angle,
            end_angle,
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

    /// Emit arc centerline points at half-step angular offsets.
    pub(super) fn rasterize_arc_points<F>(
        &self,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let arc_length = Self::arc_line_length(radius, start_angle, end_angle);
        let m_grid = arc_length / grid.dx;
        let num_points = (m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let scale = m_grid / num_points as f64;
        let angle_span = end_angle - start_angle;
        for idx in 0..num_points {
            let angle_deg = angle_span.mul_add((idx as f64 + 0.5) / num_points as f64, start_angle);
            let angle_rad = angle_deg.to_radians();
            let point = [
                radius.mul_add(angle_rad.cos(), center.0),
                radius.mul_add(angle_rad.sin(), center.1),
                center.2,
            ];
            visit(point, scale);
        }
    }

    // ─── Bowl ──────────────────────────────────────────────────────────────

    pub(super) fn rasterize_bowl(
        &self,
        mask: &mut Array3<bool>,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_bowl_points(grid, center, radius, diameter, |point, scale| {
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

    pub(super) fn rasterize_bowl_weighted(
        &self,
        mask: &mut Array3<f64>,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        diameter: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_bowl_points(grid, center, radius, diameter, |point, scale| {
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

    /// Emit golden-angle spiral surface samples for a bowl element.
    pub(super) fn rasterize_bowl_points<F>(
        &self,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        diameter: f64,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let area = Self::bowl_surface_area(radius, diameter);
        let m_grid = area / (grid.dx * grid.dx);
        let num_points = (m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let scale = m_grid / num_points as f64;

        if num_points == 1 {
            visit([center.0 - radius, center.1, center.2], scale);
            return;
        }

        let half_aperture = diameter / 2.0;
        let varphi_max = (half_aperture / radius).clamp(-1.0, 1.0).asin();
        let denom = (num_points - 1) as f64;
        let spiral_scale = TWO_PI * (1.0 - varphi_max.cos());

        for t in 0..num_points {
            let theta = GOLDEN_ANGLE * t as f64;
            let varphi = (1.0 - spiral_scale * (t as f64) / (denom * TWO_PI))
                .clamp(-1.0, 1.0)
                .acos();
            let radial = radius * varphi.sin();
            // k-wave canonical: R @ [cos(θ)sin(φ), sin(θ)sin(φ), cos(φ)] →
            // [−cos(φ), sin(θ)sin(φ), cos(θ)sin(φ)] (rotation maps [0,0,−1]
            // → bowl-axis [1,0,0]; see make_cart_bowl/compute_linear_transform).
            let point = [
                radius.mul_add(-varphi.cos(), center.0),
                center.1 + radial * theta.sin(),
                center.2 + radial * theta.cos(),
            ];
            visit(point, scale);
        }
    }

    // ─── Annulus ───────────────────────────────────────────────────────────

    pub(super) fn rasterize_annulus(
        &self,
        mask: &mut Array3<bool>,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_annulus_points(
            grid,
            center,
            radius,
            inner_diameter,
            outer_diameter,
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
    }

    pub(super) fn rasterize_annulus_weighted(
        &self,
        mask: &mut Array3<f64>,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    ) {
        let x_vec = grid.x_coordinates();
        let y_vec = grid.y_coordinates();
        let z_vec = grid.z_coordinates();
        self.rasterize_annulus_points(
            grid,
            center,
            radius,
            inner_diameter,
            outer_diameter,
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

    /// Emit golden-angle spiral surface samples for the spherical ring between
    /// `inner_diameter` and `outer_diameter`. Matches k-wave-python's
    /// `make_cart_spherical_segment` step-size convention so the spiral phase
    /// remains consistent with a co-located bowl element.
    pub(super) fn rasterize_annulus_points<F>(
        &self,
        grid: &kwavers_grid::Grid,
        center: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
        mut visit: F,
    ) where
        F: FnMut([f64; 3], f64),
    {
        let area = Self::annulus_surface_area(radius, inner_diameter, outer_diameter);
        let m_grid = area / (grid.dx * grid.dx);
        let num_points = (m_grid * DISC_SAMPLE_UPSAMPLING_RATE).ceil().max(1.0) as usize;
        let scale = m_grid / num_points as f64;

        let varphi_max = (outer_diameter / 2.0 / radius).clamp(0.0, 1.0).asin();
        let varphi_min = (inner_diameter / 2.0 / radius).clamp(0.0, 1.0).asin();

        if num_points == 1 {
            let varphi = (0.5 * (varphi_min + varphi_max)).clamp(0.0, std::f64::consts::PI);
            let radial = radius * varphi.sin();
            visit(
                [
                    radius.mul_add(-varphi.cos(), center.0),
                    center.1,
                    center.2 + radial,
                ],
                scale,
            );
            return;
        }

        let c_step = (1.0 - varphi_max.cos()) / (num_points as f64 - 1.0);
        let t_start = if varphi_min > 0.0 {
            ((1.0 - varphi_min.cos()) / c_step).ceil()
        } else {
            0.0
        };
        let t_end = (num_points - 1) as f64;
        let span = (t_end - t_start).max(0.0);

        for k in 0..num_points {
            let frac = k as f64 / (num_points as f64 - 1.0);
            let t = t_start + frac * span;
            let cos_phi = (1.0 - c_step * t).clamp(-1.0, 1.0);
            let varphi = cos_phi.acos();
            let theta = GOLDEN_ANGLE * t;
            let radial = radius * varphi.sin();
            // Same k-wave rotation convention as bowl (see rasterize_bowl_points).
            let point = [
                radius.mul_add(-varphi.cos(), center.0),
                center.1 + radial * theta.sin(),
                center.2 + radial * theta.cos(),
            ];
            visit(point, scale);
        }
    }
}
