//! Eckart acoustic streaming model.
//!
//! ## Mathematical foundation
//!
//! Eckart (1948) showed that a sound wave attenuated by viscous and
//! thermal losses transfers momentum to the fluid at a rate
//!
//! ```text
//! F_streaming = (2 α / c) · I   [N/m³]
//! ```
//!
//! where `α` is the amplitude absorption coefficient (Np/m), `c` is the
//! local sound speed, and `I = ⟨p²⟩ / (ρ c)` is the time-averaged
//! acoustic intensity for a progressive plane wave (Hamilton & Blackstock
//! 1998, eq. 2.27).
//!
//! Substituting:
//!
//! ```text
//! F_streaming = (2 α / c) · p² / (2 ρ c) = α · p² / (ρ c²)   [N/m³]
//! ```
//!
//! In the strongly-damped, low-Reynolds-number Eckart regime the steady
//! streaming velocity satisfies a Stokes balance
//!
//! ```text
//! μ ∇² v_s = −F_streaming
//! ```
//!
//! whose magnitude scales as
//!
//! ```text
//! |v_s| ~ F_streaming · L² / μ
//! ```
//!
//! for a characteristic flow length `L` (e.g. the focal-beam radius or
//! the smallest available grid spacing). For a time-stepping update we
//! therefore advance
//!
//! ```text
//! v_s(t+Δt) = v_s(t) + Δt · F_streaming · L² / μ
//! ```
//!
//! ## References
//!
//! - Eckart C. (1948). *Phys. Rev.* 73(1), 68–76.
//! - Lighthill MJ (1978). *Waves in Fluids*. Cambridge UP, §4.7.
//! - Hamilton MF & Blackstock DT (1998). *Nonlinear Acoustics*, §2.4.
//! - Nyborg WL (1953). *J. Acoust. Soc. Am.* 25, 68–75.

use kwavers_core::constants::acoustic_parameters::REFERENCE_FREQUENCY_HZ;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use ndarray::{Array3, Zip};

/// Eckart steady-streaming velocity field driven by acoustic absorption.
#[derive(Debug)]
pub struct StreamingModel {
    pub velocity: Array3<f64>,
}

impl StreamingModel {
    #[must_use]
    pub fn new(grid: &Grid) -> Self {
        Self {
            velocity: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }

    /// Advance the streaming velocity field by `dt`.
    ///
    /// At each voxel the time-averaged Eckart body force
    /// `F = α · ⟨p²⟩ / (ρ c²)` is balanced against viscous drag through
    /// the Stokes-flow scaling `v_s ~ F · L² / μ`, where `L` is the
    /// finest local grid spacing — the smallest physically meaningful
    /// length scale represented on the mesh.
    ///
    /// Absorption is evaluated at the SSOT
    /// [`REFERENCE_FREQUENCY_HZ`] (1 MHz). Callers that drive the
    /// streaming model at a different operating frequency should compose
    /// a wrapping integrator that re-evaluates `α(f)`.
    pub fn update_velocity(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        let length_scale_sq = grid.dx.min(grid.dy).min(grid.dz).powi(2);
        Zip::indexed(&mut self.velocity)
            .and(pressure)
            .for_each(|(i, j, k), v, &p| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let rho = kwavers_medium::density_at(medium, x, y, z, grid);
                let c = kwavers_medium::sound_speed_at(medium, x, y, z, grid);
                let alpha = medium.absorption_coefficient(x, y, z, grid, REFERENCE_FREQUENCY_HZ);
                let mu = medium.viscosity(x, y, z, grid);
                if rho <= 0.0 || c <= 0.0 || mu <= 0.0 {
                    return;
                }
                // Time-averaged Eckart body force density (N/m^3) for a
                // progressive plane wave: F = α · ⟨p²⟩ / (ρ c²). The factor
                // of 1/2 turning |p|² into ⟨p²⟩ for sinusoidal forcing is
                // already embedded in the simplification 2α·I/c → α·p²/(ρc²).
                let force_density = alpha * p * p / (rho * c * c);
                // Stokes-flow scaling: μ ∇²v ≈ −F  ⇒  v ~ F · L² / μ.
                let dvdt = force_density * length_scale_sq / mu;
                let next = *v + dt * dvdt;
                if next.is_finite() {
                    *v = next;
                } else {
                    *v = 0.0;
                }
            });
    }

    #[must_use]
    pub fn velocity(&self) -> &Array3<f64> {
        &self.velocity
    }
}

use crate::traits::StreamingModelTrait;

impl StreamingModelTrait for StreamingModel {
    fn update_velocity(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        self.update_velocity(pressure, grid, medium, dt);
    }

    fn velocity(&self) -> &Array3<f64> {
        self.velocity()
    }
}
