//! Westervelt nonlinear-term kernel: `∂²(p²)/∂t²` via the product rule.
//!
//! With three pressure histories `p^n`, `p^{n-1}`, `p^{n-2}` available,
//! `∂²(p²)/∂t² ≈ 2p · ∂²p/∂t² + 2(∂p/∂t)²`. On the very first step only
//! two histories exist, so the kernel falls back to `2(∂p/∂t)²`
//! (forward-difference initialization, LeVeque 2007 §2.14).

use ndarray::{Array3, Zip};

use crate::domain::grid::Grid;

use super::WesterveltFdtd;

impl WesterveltFdtd {
    /// Calculate the nonlinear term ∂²(p²)/∂t²
    pub(super) fn calculate_nonlinear_term(&self, dt: f64, grid: &Grid) -> Array3<f64> {
        let mut nonlinear = Array3::zeros((grid.nx, grid.ny, grid.nz));

        if let Some(ref p_prev2) = self.pressure_prev2 {
            // Full second-order time derivative of p²
            // ∂²(p²)/∂t² = 2p * ∂²p/∂t² + 2(∂p/∂t)²

            Zip::from(&mut nonlinear)
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
            Zip::from(&mut nonlinear)
                .and(&self.pressure)
                .and(&self.pressure_prev)
                .par_for_each(|nl, &p, &p_prev| {
                    let dp_dt = (p - p_prev) / dt;
                    *nl = 2.0 * dp_dt * dp_dt;
                });
        }

        nonlinear
    }
}
