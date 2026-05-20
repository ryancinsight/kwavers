//! Westervelt nonlinear pressure correction and history rotation.
//!
//! ## References
//!
//! - Westervelt, P. J. (1963). J. Acoust. Soc. Am. 35(4), 535–537.
//! - Hamilton, M. F. & Blackstock, D. T. (1998). Nonlinear Acoustics, Ch. 3.
//! - Aanonsen, S. I. et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768.

use ndarray::{Array3, Zip};

use super::super::solver::FdtdSolver;

impl FdtdSolver {
    /// Apply Westervelt nonlinear correction to the current pressure field.
    ///
    /// Discretization (Hamilton & Blackstock 1998, Eq. 3.43a):
    /// ```text
    /// S_nl^n = (β/(ρ₀c₀²)) · [2pⁿ(pⁿ−2pⁿ⁻¹+pⁿ⁻²)/Δt² + 2((pⁿ−pⁿ⁻¹)/Δt)²]  [Pa/s²]
    /// Δpⁿ    = Δt² · S_nl^n                                                        [Pa]
    /// ```
    /// Note: nl_coeff = β/(ρ₀c₀²), so Δp = Δt² · nl_coeff · d²(p²)/dt².
    pub(crate) fn apply_westervelt_nonlinear_correction(&mut self, dt: f64) {
        let (Some(nl_coeff), Some(ref mut nl_scratch)) =
            (self.nl_coeff.as_ref(), self.nl_scratch.as_mut())
        else {
            return;
        };

        let dt2_inv = 1.0 / (dt * dt);
        let dt_inv = 1.0 / dt;

        match (&self.p_prev, &self.p_prev2) {
            (Some(p_prev), Some(p_prev2)) => {
                for (((&p, &pp), &pp2), (&nlc, nl)) in self
                    .fields
                    .p
                    .iter()
                    .zip(p_prev.iter())
                    .zip(p_prev2.iter())
                    .zip(nl_coeff.iter().zip(nl_scratch.iter_mut()))
                {
                    let d2p_dt2 = (2.0f64.mul_add(-pp, p) + pp2) * dt2_inv;
                    let dp_dt = (p - pp) * dt_inv;
                    let d2p2_dt2 = (2.0 * p).mul_add(d2p_dt2, 2.0 * dp_dt * dp_dt);
                    *nl = dt * dt * nlc * d2p2_dt2;
                }
            }
            (Some(p_prev), None) => {
                for ((&p, &pp), (&nlc, nl)) in self
                    .fields
                    .p
                    .iter()
                    .zip(p_prev.iter())
                    .zip(nl_coeff.iter().zip(nl_scratch.iter_mut()))
                {
                    let dp_dt = (p - pp) * dt_inv;
                    let d2p2_dt2 = 2.0 * dp_dt * dp_dt;
                    *nl = dt * dt * nlc * d2p2_dt2;
                }
            }
            _ => return,
        }

        Zip::from(self.fields.p.view_mut())
            .and(nl_scratch.view())
            .par_for_each(|p, &nl| *p += nl);
    }

    /// Rotate pressure history: p^{n-2} ← p^{n-1} ← p^n (swap to avoid allocation).
    pub(crate) fn rotate_pressure_history(&mut self) {
        if let Some(ref mut p_prev2) = self.p_prev2 {
            if let Some(ref mut p_prev) = self.p_prev {
                std::mem::swap(p_prev2, p_prev);
            }
        } else if self.p_prev.is_some() {
            self.p_prev2 = self.p_prev.take();
            self.p_prev = Some(Array3::zeros(self.fields.p.dim()));
        }

        if let Some(ref mut p_prev) = self.p_prev {
            p_prev.assign(&self.fields.p);
        }
    }
}
