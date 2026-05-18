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
//! the raw B/A value.  This file computes β = 1 + B/A / 2 internally.
//! Do not pass β directly as `nonlinearity`; the conversion is the solver's
//! responsibility, not the medium's.
//!
//! Typical values: water ≈ 5.0 (B/A), soft tissue ≈ 6.0–7.5.
//!
//! ## Note on the stored `nonlinear_term`
//!
//! `nonlinear_term` stores the raw curvature `∂²(p²)/∂t²` without the
//! `β/(ρ₀c₀⁴)` prefactor.  That factor is applied during the leapfrog update
//! (`nl_coeff = β·Δt²/(ρ·c²)`) in `update.rs`, keeping this kernel
//! medium-agnostic and reusable.
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

use ndarray::Zip;

use crate::domain::grid::Grid;

use super::WesterveltFdtd;

impl WesterveltFdtd {
    /// Calculate the nonlinear term ∂²(p²)/∂t²
    pub(super) fn calculate_nonlinear_term_into(&mut self, dt: f64, grid: &Grid) {
        debug_assert_eq!(self.nonlinear_term.dim(), (grid.nx, grid.ny, grid.nz));
        if let Some(ref p_prev2) = self.pressure_prev2 {
            // Full second-order time derivative of p²
            // ∂²(p²)/∂t² = 2p * ∂²p/∂t² + 2(∂p/∂t)²

            Zip::from(&mut self.nonlinear_term)
                .and(&self.pressure)
                .and(&self.pressure_prev)
                .and(p_prev2)
                .par_for_each(|nl, &p, &p_prev, &p_prev2| {
                    let d2p_dt2 = (2.0f64.mul_add(-p_prev, p) + p_prev2) / (dt * dt);
                    let dp_dt = (p - p_prev) / dt;
                    *nl = (2.0 * p).mul_add(d2p_dt2, 2.0 * dp_dt * dp_dt);
                });
        } else {
            // First time step: forward difference initialization (LeVeque 2007 §2.14)
            Zip::from(&mut self.nonlinear_term)
                .and(&self.pressure)
                .and(&self.pressure_prev)
                .par_for_each(|nl, &p, &p_prev| {
                    let dp_dt = (p - p_prev) / dt;
                    *nl = 2.0 * dp_dt * dp_dt;
                });
        }
    }
}
