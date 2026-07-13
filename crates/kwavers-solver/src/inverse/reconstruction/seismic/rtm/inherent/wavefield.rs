//! Single finite-difference wavefield update step for RTM.
//!
//! # Theorem: 4th-order FD Laplacian via Moirai strided views
//!
//! The wave equation second-order-in-time update is:
//! ```text
//! p^{n+1}[i,j,k] = 2·p^n[i,j,k] − p^{n-1}[i,j,k] + c²·dt²·∇²p^n[i,j,k]
//! ```
//!
//! The 4th-order isotropic Laplacian in direction d with grid spacing `h`:
//! ```text
//! ∂²p/∂d² ≈ (α₋₂·p[d-2] + α₋₁·p[d-1] + α₀·p\[d\] + α₁·p[d+1] + α₂·p[d+2]) / h²
//! ```
//! Coefficients `(α₋₂, α₋₁, α₀, α₁, α₂)` come from `fd_coeffs::{FD_COEFF_2, FD_COEFF_1, FD_COEFF_0}`.
//!
//! ## Implementation: strided-view Moirai traversal
//!
//! For interior points `i ∈ [2, nx-2)`, the shifted pressure slices are:
//! ```text
//! pm2 = pressure[..nx-4,  2..ny-2, 2..nz-2]   (i-2)
//! pm1 = pressure[1..nx-3, 2..ny-2, 2..nz-2]   (i-1)
//! p0  = pressure[2..nx-2, 2..ny-2, 2..nz-2]   (i  )
//! pp1 = pressure[3..nx-1, 2..ny-2, 2..nz-2]   (i+1)
//! pp2 = pressure[4..nx,   2..ny-2, 2..nz-2]   (i+2)
//! ```
//! All five slices have shape `(nx-4, ny-4, nz-4)` and are passed to a
//! Moirai-backed strided view pass. The y- and z-directions are
//! handled by two subsequent Moirai passes that accumulate into the
//! same interior laplacian slice.
//!
//! ## Correctness guarantee
//!
//! The three-pass accumulation is equivalent to the original triple-loop:
//! each pass adds one directional contribution starting from the zero-
//! initialised laplacian buffer, so the total equals
//! `(∂²/∂x² + ∂²/∂y² + ∂²/∂z²)p`.  The boundary ghost cells remain zero
//! (PML is applied externally).

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::Array3;

use super::super::super::fd_coeffs::{FD_COEFF_0, FD_COEFF_1, FD_COEFF_2};
use super::super::types::ReverseTimeMigration;
use super::parallel::for_each_view_mut;

impl ReverseTimeMigration {
    /// Advance `pressure` by one finite-difference time step in-place.
    ///
    /// `pressure_previous` holds the field at `t - dt` (staggered two-level
    /// leapfrog storage; caller swaps after this call).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn update_wavefield(
        &self,
        pressure: &mut Array3<f64>,
        pressure_previous: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let dt = self.config.dt;
        let [nx, ny, nz] = pressure.shape();
        let dx2 = grid.dx * grid.dx;
        let dy2 = grid.dy * grid.dy;
        let dz2 = grid.dz * grid.dz;

        // ── 4th-order FD Laplacian, three parallel passes ──────────────────
        // Interior: i ∈ [2,nx-2), j ∈ [2,ny-2), k ∈ [2,nz-2).
        let mut laplacian = Array3::<f64>::zeros((nx, ny, nz));
        let inn = s![2..nx - 2, 2..ny - 2, 2..nz - 2];

        let x_pm2 = pressure
            .slice_with::<3>(&s![..nx - 4, 2..ny - 2, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        let x_pm1 = pressure
            .slice_with::<3>(&s![1..nx - 3, 2..ny - 2, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        let x_p0 = pressure
            .slice_with::<3>(&s![2..nx - 2, 2..ny - 2, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        let x_pp1 = pressure
            .slice_with::<3>(&s![3..nx - 1, 2..ny - 2, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        let x_pp2 = pressure
            .slice_with::<3>(&s![4..nx, 2..ny - 2, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        for_each_view_mut(
            laplacian
                .slice_with_mut::<3>(&inn)
                .expect("invariant: RTM interior slice in range"),
            |idx, lap| {
                let pm2 = x_pm2[idx];
                let pm1 = x_pm1[idx];
                let p0 = x_p0[idx];
                let pp1 = x_pp1[idx];
                let pp2 = x_pp2[idx];
                *lap += FD_COEFF_2.mul_add(
                    pp2,
                    FD_COEFF_1.mul_add(
                        pp1,
                        FD_COEFF_0.mul_add(p0, FD_COEFF_2.mul_add(pm2, FD_COEFF_1 * pm1)),
                    ),
                ) / dx2;
            },
        );

        let y_pm2 = pressure
            .slice_with::<3>(&s![2..nx - 2, ..ny - 4, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        let y_pm1 = pressure
            .slice_with::<3>(&s![2..nx - 2, 1..ny - 3, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        let y_p0 = pressure
            .slice_with::<3>(&s![2..nx - 2, 2..ny - 2, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        let y_pp1 = pressure
            .slice_with::<3>(&s![2..nx - 2, 3..ny - 1, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        let y_pp2 = pressure
            .slice_with::<3>(&s![2..nx - 2, 4..ny, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        for_each_view_mut(
            laplacian
                .slice_with_mut::<3>(&inn)
                .expect("invariant: RTM interior slice in range"),
            |idx, lap| {
                let pm2 = y_pm2[idx];
                let pm1 = y_pm1[idx];
                let p0 = y_p0[idx];
                let pp1 = y_pp1[idx];
                let pp2 = y_pp2[idx];
                *lap += FD_COEFF_2.mul_add(
                    pp2,
                    FD_COEFF_1.mul_add(
                        pp1,
                        FD_COEFF_0.mul_add(p0, FD_COEFF_2.mul_add(pm2, FD_COEFF_1 * pm1)),
                    ),
                ) / dy2;
            },
        );

        let z_pm2 = pressure
            .slice_with::<3>(&s![2..nx - 2, 2..ny - 2, ..nz - 4])
            .expect("invariant: RTM stencil slice in range");
        let z_pm1 = pressure
            .slice_with::<3>(&s![2..nx - 2, 2..ny - 2, 1..nz - 3])
            .expect("invariant: RTM stencil slice in range");
        let z_p0 = pressure
            .slice_with::<3>(&s![2..nx - 2, 2..ny - 2, 2..nz - 2])
            .expect("invariant: RTM stencil slice in range");
        let z_pp1 = pressure
            .slice_with::<3>(&s![2..nx - 2, 2..ny - 2, 3..nz - 1])
            .expect("invariant: RTM stencil slice in range");
        let z_pp2 = pressure
            .slice_with::<3>(&s![2..nx - 2, 2..ny - 2, 4..nz])
            .expect("invariant: RTM stencil slice in range");
        for_each_view_mut(
            laplacian
                .slice_with_mut::<3>(&inn)
                .expect("invariant: RTM interior slice in range"),
            |idx, lap| {
                let pm2 = z_pm2[idx];
                let pm1 = z_pm1[idx];
                let p0 = z_p0[idx];
                let pp1 = z_pp1[idx];
                let pp2 = z_pp2[idx];
                *lap += FD_COEFF_2.mul_add(
                    pp2,
                    FD_COEFF_1.mul_add(
                        pp1,
                        FD_COEFF_0.mul_add(p0, FD_COEFF_2.mul_add(pm2, FD_COEFF_1 * pm1)),
                    ),
                ) / dz2;
            },
        );

        // ── Leapfrog pressure update: p^{n+1} = 2p^n − p^{n-1} + c²dt²∇²p ─
        let previous = pressure_previous.view();
        let laplacian = laplacian.view();
        let velocity = self.velocity_model.view();
        for_each_view_mut(pressure.view_mut(), |idx, p| {
            let p_prev = previous[idx];
            let lap = laplacian[idx];
            let vel = velocity[idx];
            *p = (vel * vel * dt * dt).mul_add(lap, 2.0f64.mul_add(*p, -p_prev));
        });

        Ok(())
    }
}
