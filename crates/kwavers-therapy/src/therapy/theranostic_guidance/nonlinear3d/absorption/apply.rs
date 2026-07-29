//! Forward (`apply`) and adjoint (`apply_transpose`) time-stepping kernels
//! for the fractional-Laplacian absorption operator. See `absorption` module
//! docs for the discretisation derivation and self-adjointness proof.
//!
//! `apply` is the hot path (called every forward timestep).  It achieves
//! zero per-step allocation by reusing three persistent buffers on the
//! operator struct: `spatial_buf` for the FFT input copy, `spectrum_buf`
//! for the complex spectrum scratch, and `l_y_curr_buf` for the current
//! spectral-filter result.  The previous step's result lives in `prev_l_y`
//! and the buffers are swapped each call so the old storage is recycled
//! rather than dropped-and-reallocated.
//!
//! `apply_transpose` is the cold path (called once per adjoint gradient
//! computation) and allocates its scratch buffers locally.

use leto::Array3;
use moirai_parallel::{enumerate_mut_with, map_collect_index_with, Adaptive};

use super::spectrum::spectral_filter_into;
use super::FractionalLaplacianAbsorption;

impl FractionalLaplacianAbsorption {
    /// Apply the fractional-Laplacian absorption correction to `next`,
    /// where `next` is the post-wave-equation, post-sponge pressure field
    /// at time level `n+1` and `current`, `previous` hold the levels `n`
    /// and `n−1` (rotated-buffer convention from `forward::update_cells`).
    ///
    /// # Zero-alloc invariant
    ///
    /// After the first timestep (which primes `prev_l_y`) every subsequent
    /// call reuses the four persistent fields — `prev_l_y`, `l_y_curr_buf`,
    /// `spatial_buf`, `spectrum_buf` — without any heap allocation.
    pub(crate) fn apply(&mut self, current: &[f64], previous: &[f64], next: &mut [f64]) {
        let n = self.n;
        let cells = n * n * n;
        debug_assert_eq!(current.len(), cells);
        debug_assert_eq!(previous.len(), cells);
        debug_assert_eq!(next.len(), cells);

        // L_y(p[n−1]): from cache (first call → compute into fresh buffer).
        let l_y_prev: Array3<f64> = match self.prev_l_y.take() {
            Some(cached) => cached,
            None => {
                let mut buf = Array3::zeros([n, n, n]);
                spectral_filter_into(
                    n,
                    previous,
                    &self.k_pow_y,
                    &mut buf,
                    &mut self.spatial_buf,
                    &mut self.spectrum_buf,
                );
                buf
            }
        };

        // L_y(p`N`): recompute into the persistent `l_y_curr_buf`.
        spectral_filter_into(
            n,
            current,
            &self.k_pow_y,
            &mut self.l_y_curr_buf,
            &mut self.spatial_buf,
            &mut self.spectrum_buf,
        );

        // Apply correction: next += −dt·τ·(L_y(p`N`) − L_y(p[n−1]))
        let l_y_curr_slice = self.l_y_curr_buf.as_slice().unwrap();
        let l_y_prev_slice = l_y_prev.as_slice().unwrap();
        enumerate_mut_with::<Adaptive, _, _>(next, |i, dst| {
            *dst += -self.dt_tau[i] * (l_y_curr_slice[i] - l_y_prev_slice[i]);
        });

        // Cache L_y(p`N`) for next step.  Swap storage so the old cache
        // buffer is recycled as next step's l_y_curr_buf — zero per-step
        // allocation after the first call primes the cache.
        // After swap: prev_l_y = old l_y_curr_buf (holds L_y(p`N`)),
        //             l_y_curr_buf = old prev_l_y (will be overwritten next call).
        self.prev_l_y = Some(std::mem::replace(&mut self.l_y_curr_buf, l_y_prev));
    }

    /// Adjoint (transpose) of [`apply`] used by `adjoint::gradient` to
    /// backpropagate the discrete adjoint variables across one absorption
    /// step.  Given the forward Jacobian
    ///
    /// ```text
    ///   J_curr = −dt·τ·L_y      (∂Δp_abs / ∂p`N`)
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
    ///
    /// # Allocation note
    ///
    /// This is the cold path (called once per adjoint gradient).  Scratch
    /// buffers are allocated locally; the hot forward path (`apply`) is
    /// the one that optimises for zero per-step allocation.
    pub(crate) fn apply_transpose(
        &self,
        adj_next: &[f64],
        adj_curr: &mut [f64],
        adj_prev: &mut [f64],
    ) {
        let n = self.n;
        let cells = n * n * n;
        debug_assert_eq!(adj_next.len(), cells);
        debug_assert_eq!(adj_curr.len(), cells);
        debug_assert_eq!(adj_prev.len(), cells);

        let scaled_tau: Vec<f64> =
            map_collect_index_with::<Adaptive, _, _>(cells, |i| adj_next[i] * self.dt_tau[i]);

        let mut local_spatial = Array3::zeros([n, n, n]);
        let mut local_spectrum =
            Array3::from_elem([n, n, n], kwavers_math::fft::Complex64::default());
        let mut l_y_tau = Array3::zeros([n, n, n]);
        spectral_filter_into(
            n,
            &scaled_tau,
            &self.k_pow_y,
            &mut l_y_tau,
            &mut local_spatial,
            &mut local_spectrum,
        );

        let l_y_tau_slice = l_y_tau.as_slice().unwrap();
        enumerate_mut_with::<Adaptive, _, _>(adj_curr, |i, dst| {
            *dst -= l_y_tau_slice[i];
        });
        enumerate_mut_with::<Adaptive, _, _>(adj_prev, |i, dst| {
            *dst += l_y_tau_slice[i];
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
