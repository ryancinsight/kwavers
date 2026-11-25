//! Westervelt nonlinear-term kernel: `∂²(p²)/∂t²` via the product rule.
//!
//! ## Theorem (B/A vs β interface contract)
//!
//! The `Medium` trait returns the **parameter of nonlinearity** B/A
//! (dimensionless ratio of the Taylor-expansion coefficients of the equation of
//! state, Beyer 1960).  The **coefficient of nonlinearity** β used in the
//! Westervelt equation is defined as:
//!
//! ```text
//! β = 1 + B/(2A)
//! ```
//!
//! The Westervelt nonlinear term is:
//!
//! ```text
//! (β / ρ₀c₀⁴) · ∂²(p²)/∂t²
//! ```
//!
//! **Derivation.** The equation of state expanded to second order in density
//! perturbation ρ' = ρ − ρ₀ is:
//!
//! ```text
//! p = c₀²ρ' + (B/A)·c₀²ρ'²/(2ρ₀) + O(ρ'³)
//! ```
//!
//! Inserting the nonlinear continuity equation `∂ρ'/∂t = −ρ₀·div(u)` and
//! eliminating velocity via the linearized momentum equation yields the
//! Westervelt equation with coefficient β = 1 + B/(2A).  (Hamilton &
//! Blackstock 1998, §2.3.2, eq. 2.3.10.)
//!
//! **Interface contract.** Callers of `medium.nonlinearity(i, j, k)` receive
//! the raw B/A value.  The pressure update computes β = 1 + B/A / 2 internally.
//! Do not pass β directly as `nonlinearity`; the conversion is the solver's
//! responsibility, not the medium's.
//!
//! Typical values: water ≈ 5.0 (B/A), soft tissue ≈ 6.0–7.5.
//!
//! The curvature has units Pa²/s². The pressure update multiplies it by
//! β·Δt²/(ρ·c²) to obtain its pressure increment in Pa.
//!
//! With three pressure histories `p^n`, `p^{n-1}`, `p^{n-2}` available,
//! `∂²(p²)/∂t² ≈ 2p · ∂²p/∂t² + 2(∂p/∂t)²`. On the very first step only
//! two histories exist, so the kernel falls back to `2(∂p/∂t)²`
//! (forward-difference initialization, LeVeque 2007 §2.14).
//!
//! ## References
//!
//! - Westervelt PJ (1963). J. Acoust. Soc. Am. 35(4), 535–537.
//!   DOI: 10.1121/1.1918525
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press.
//!   §2.3.2, eq. (2.3.10).
//! - Beyer RT (1960). J. Acoust. Soc. Am. 32(6), 719–721.
//!   DOI: 10.1121/1.1908195 (original B/A measurement in water)
//! - LeVeque RJ (2007). Finite Difference Methods for ODEs and PDEs.
//!   SIAM. §2.14.

use kwavers_grid::Grid;
use kwavers_medium::{properties::coefficient_of_nonlinearity, Medium};
use moirai_parallel::{enumerate_mut_with, Adaptive};

use super::WesterveltFdtd;

#[inline]
fn curvature<const HAS_HISTORY: bool>(p: f64, previous: f64, oldest: f64, dt: f64) -> f64 {
    let dp_dt = (p - previous) / dt;
    let slope_term = 2.0 * dp_dt * dp_dt;
    if HAS_HISTORY {
        let d2p_dt2 = (2.0f64.mul_add(-previous, p) + oldest) / (dt * dt);
        (2.0 * p).mul_add(d2p_dt2, slope_term)
    } else {
        slope_term
    }
}

impl WesterveltFdtd {
    pub(super) fn advance_pressure<const HAS_HISTORY: bool>(
        &mut self,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) {
        let pressure = self
            .pressure
            .as_slice()
            .expect("invariant: Westervelt pressure is standard-layout");
        let pressure_prev = self
            .pressure_prev
            .as_slice()
            .expect("invariant: Westervelt previous pressure is standard-layout");
        let laplacian = self
            .laplacian
            .as_slice()
            .expect("invariant: Westervelt laplacian is standard-layout");
        let pressure_next = self
            .pressure_next
            .as_slice_mut()
            .expect("invariant: Westervelt next pressure is standard-layout");
        let enable_absorption = self.config.enable_absorption;
        let artificial_viscosity = self.config.artificial_viscosity;
        let ny = grid.ny;
        let nz = grid.nz;
        let slab_len = ny * nz;

        enumerate_mut_with::<Adaptive, _, _>(pressure_next, |idx, p_next| {
            let i = idx / slab_len;
            let rem = idx % slab_len;
            let j = rem / nz;
            let k = rem % nz;
            let p = pressure[idx];
            let p_prev = pressure_prev[idx];
            let lap = laplacian[idx];
            let nl = curvature::<HAS_HISTORY>(p, p_prev, *p_next, dt);
            let c = medium.sound_speed(i, j, k).into_base();
            let rho = medium.density(i, j, k).into_base();
            let beta = coefficient_of_nonlinearity(medium.nonlinearity(i, j, k).into_base());

            // Linear wave + artificial viscosity share the precomputed Laplacian.
            // artificial_viscosity·Δt·∇²p uses the same stencil order as the
            // main wave operator — no redundant stencil recomputation.
            let linear_and_visc = (c * dt).mul_add(c * dt, artificial_viscosity * dt) * lap;

            // Nonlinear coefficient β·Δt²/(ρ·c²)
            let nl_coeff = beta * dt * dt / (rho * c * c);

            // Leapfrog propagation step (Hamilton & Blackstock 1998 §3.5)
            let p_propagated = 2.0f64.mul_add(p, -p_prev) + linear_and_visc + nl_coeff * nl;

            // Multiplicative absorption (Stokes-Kirchhoff, O(Δt) operator splitting):
            //   p *= exp(−α·c·Δt)
            // Stable for all α ≥ 0.  See module-level doc for derivation.
            *p_next = if enable_absorption {
                let alpha = medium.absorption(i, j, k).into_base();
                p_propagated * (-alpha * c * dt).exp()
            } else {
                p_propagated
            };
        });
    }
}
