//! Forward (`apply`) and adjoint (`apply_transpose`) time-stepping kernels
//! for the fractional-Laplacian absorption operator. See `absorption` module
//! docs for the discretisation derivation and self-adjointness proof.

use moirai_parallel::{enumerate_mut_with, map_collect_index_with, Adaptive};

use super::spectrum::spectral_filter;
use super::FractionalLaplacianAbsorption;

impl FractionalLaplacianAbsorption {
    /// Apply the fractional-Laplacian absorption correction to `next`,
    /// where `next` is the post-wave-equation, post-sponge pressure field
    /// at time level `n+1` and `current`, `previous` hold the levels `n`
    /// and `n−1` (rotated-buffer convention from `forward::update_cells`).
    ///
    /// On exit `next[i] += Δp_abs[n+1, i]`.
    pub(crate) fn apply(&mut self, current: &[f64], previous: &[f64], next: &mut [f64]) {
        let cells = self.n * self.n * self.n;
        debug_assert_eq!(current.len(), cells);
        debug_assert_eq!(previous.len(), cells);
        debug_assert_eq!(next.len(), cells);

        // ── L_y(p[n−1]):  use the cached value from the previous step if
        //    present (zero cache → first step → compute from `previous`).
        let l_y_prev = if let Some(cached) = self.prev_l_y.take() {
            cached
        } else {
            spectral_filter(self.n, previous, &self.k_pow_y)
        };

        // ── L_y(p[n]): always recompute (the new "previous" for next step).
        let l_y_curr = spectral_filter(self.n, current, &self.k_pow_y);

        // ── Apply correction: next += -dt·τ·(L_y(p[n]) - L_y(p[n-1]))
        enumerate_mut_with::<Adaptive, _, _>(next, |i, dst| {
            *dst += -self.dt_tau[i] * (l_y_curr[i] - l_y_prev[i]);
        });

        // Cache `L_y(p[n])` for the next step (becomes `L_y(p[n−1])`).
        self.prev_l_y = Some(l_y_curr);
    }

    /// Adjoint (transpose) of [`apply`] used by `adjoint::gradient` to
    /// backpropagate the discrete adjoint variables across one absorption
    /// step.  Given the forward Jacobian
    ///
    /// ```text
    ///   J_curr = −dt·τ·L_y      (∂Δp_abs / ∂p[n])
    ///   J_prev =  dt·τ·L_y      (∂Δp_abs / ∂p[n−1])
    /// ```
    ///
    /// and the self-adjointness of `L_y` plus the diagonality of the
    /// per-voxel `dt·τ` factor, the transposes accumulate into the
    /// adjoint variables as
    ///
    /// ```text
    ///   adj_curr += −L_y( dt·τ ⊙ adj_next )
    ///   adj_prev +=  L_y( dt·τ ⊙ adj_next )
    /// ```
    ///
    /// where `⊙` is the per-voxel Hadamard product.
    pub(crate) fn apply_transpose(
        &self,
        adj_next: &[f64],
        adj_curr: &mut [f64],
        adj_prev: &mut [f64],
    ) {
        let cells = self.n * self.n * self.n;
        debug_assert_eq!(adj_next.len(), cells);
        debug_assert_eq!(adj_curr.len(), cells);
        debug_assert_eq!(adj_prev.len(), cells);

        let scaled_tau: Vec<f64> =
            map_collect_index_with::<Adaptive, _, _>(cells, |i| adj_next[i] * self.dt_tau[i]);
        let l_y_tau = spectral_filter(self.n, &scaled_tau, &self.k_pow_y);

        enumerate_mut_with::<Adaptive, _, _>(adj_curr, |i, dst| {
            *dst -= l_y_tau[i];
        });
        enumerate_mut_with::<Adaptive, _, _>(adj_prev, |i, dst| {
            *dst += l_y_tau[i];
        });
    }

    /// Reset cached state so the next call to `apply` recomputes
    /// `L_y(p[n−1])` from scratch. Call between independent simulation
    /// runs that share an operator instance.
    #[allow(dead_code)]
    pub(crate) fn reset(&mut self) {
        self.prev_l_y = None;
    }
}
