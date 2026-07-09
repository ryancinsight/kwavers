//! `ConservationDiagnostics` trait impl for the Westervelt FDTD solver.
//!
//! ## Acoustic energy
//!
//! Total acoustic potential energy density (Hamilton & Blackstock 1998 §1.3):
//!
//! ```text
//! e(x) = p²(x) / (2·ρ₀·c₀²)    [J m⁻³]
//! E = ∫∫∫ e(x) dV              [J]
//! ```
//!
//! Kinetic energy is not tracked here because the Westervelt FDTD stores only
//! the scalar pressure field. The potential-energy proxy is sufficient for
//! detecting non-physical growth (instability) or excessive decay (over-damping).
//!
//! ## Acoustic momentum
//!
//! Linear-acoustic momentum density `g = ρ₀·u` where `u ≈ ∇Ψ` and `Ψ` is the
//! velocity potential. From the linearized momentum equation `ρ₀·∂u/∂t = −∇p`,
//! the time-integrated gradient gives `ρ₀·u ≈ −∫∇p dt`. For a single time step,
//! the central-difference estimate of `u` at the current instant is
//! `u ≈ ∇Ψ ≈ (1/ρ₀)·∇p / (iω)` (frequency domain). Here we track the
//! instantaneous pressure-gradient components as a proxy for detecting asymmetric
//! wave distortion.
//!
//! ## Acoustic mass perturbation
//!
//! From the linear equation of state `p = c₀²·ρ′` where `ρ′ = ρ − ρ₀`:
//! ```text
//! ρ′(x) = p(x) / c₀²
//! M′ = ∫∫∫ ρ′(x) dV = (1/c₀²) ∫∫∫ p(x) dV
//! ```
//!
//! For a closed domain `M′` should vanish at all times (mass conservation).
//!
//! ## Parallelism
//!
//! All three integrals use Moirai indexed reductions to amortize summation cost
//! on large 3D grids while keeping the solver on the Atlas execution provider.

use super::WesterveltFdtd;
use crate::forward::nonlinear::conservation::ConservationDiagnostics;
use moirai_parallel::{reduce_index_with, Adaptive};

impl ConservationDiagnostics for WesterveltFdtd {
    /// Total acoustic potential energy `E = ∫ p²/(2ρ₀c₀²) dV`.
    fn calculate_total_energy(&self) -> f64 {
        let rho0 = self.medium_properties.rho0;
        let c0 = self.medium_properties.c0;
        let factor = 1.0 / (2.0 * rho0 * c0 * c0);
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        let pressure = self
            .pressure
            .as_slice()
            .expect("invariant: Westervelt pressure is standard-layout");

        reduce_index_with::<Adaptive, _, _, _>(
            (pressure.shape()[0] * pressure.shape()[1] * pressure.shape()[2]),
            0.0,
            |idx| pressure[idx] * pressure[idx] * factor * dv,
            |a, b| a + b,
        )
    }

    /// Pressure-gradient momentum proxy `(∫ ρ₀·∂p/∂x dV, ∫ ρ₀·∂p/∂y dV, ∫ ρ₀·∂p/∂z dV)`.
    ///
    /// Uses central differences on the interior. Boundary contributions are zero.
    fn calculate_total_momentum(&self) -> (f64, f64, f64) {
        let rho0 = self.medium_properties.rho0;
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        let inv2dx = 1.0 / (2.0 * self.grid.dx);
        let inv2dy = 1.0 / (2.0 * self.grid.dy);
        let inv2dz = 1.0 / (2.0 * self.grid.dz);
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;

        let (px, py, pz) = reduce_index_with::<Adaptive, _, _, _>(
            nx.saturating_sub(2),
            (0.0, 0.0, 0.0),
            |offset| {
                let i = offset + 1;
                let mut sx = 0.0f64;
                let mut sy = 0.0f64;
                let mut sz = 0.0f64;
                for j in 1..ny - 1 {
                    for k in 1..nz - 1 {
                        let dp_dx =
                            (self.pressure[[i + 1, j, k]] - self.pressure[[i - 1, j, k]]) * inv2dx;
                        let dp_dy =
                            (self.pressure[[i, j + 1, k]] - self.pressure[[i, j - 1, k]]) * inv2dy;
                        let dp_dz =
                            (self.pressure[[i, j, k + 1]] - self.pressure[[i, j, k - 1]]) * inv2dz;
                        let scale = rho0 * dv;
                        sx += dp_dx * scale;
                        sy += dp_dy * scale;
                        sz += dp_dz * scale;
                    }
                }
                (sx, sy, sz)
            },
            |(ax, ay, az), (bx, by, bz)| (ax + bx, ay + by, az + bz),
        );

        (px, py, pz)
    }

    /// Acoustic mass perturbation `M′ = (1/c₀²) ∫ p dV`.
    ///
    /// Theorem (mass conservation): in a lossless closed domain `M′(t) = const`.
    /// Non-zero drift indicates either boundary leakage or numerical dissipation.
    fn calculate_total_mass(&self) -> f64 {
        let c0 = self.medium_properties.c0;
        let dv = self.grid.dx * self.grid.dy * self.grid.dz;
        let rho0 = self.medium_properties.rho0;
        let c0_sq = c0 * c0;
        let pressure = self
            .pressure
            .as_slice()
            .expect("invariant: Westervelt pressure is standard-layout");

        reduce_index_with::<Adaptive, _, _, _>(
            (pressure.shape()[0] * pressure.shape()[1] * pressure.shape()[2]),
            0.0,
            |idx| rho0 * (1.0 + pressure[idx] / (rho0 * c0_sq)) * dv,
            |a, b| a + b,
        )
    }
}
