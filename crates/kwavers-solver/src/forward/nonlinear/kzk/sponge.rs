//! Absorbing sponge layer for KZK transverse boundary conditions.
//!
//! Applies a raised-cosine taper to the transverse field after each
//! diffraction half-step, attenuating outgoing waves at the grid edges
//! and preventing FFT wrap-around (periodic-BC) artifacts.
//!
//! ## Window function
//!
//! For a grid of size N along one axis, with sponge fraction f (0 < f ≤ 0.5),
//! the sponge occupies the outer M = round(f·N) cells on each side:
//! ```text
//! W[i] = 1                              for M ≤ i < N − M   (interior)
//! W[i] = (1 − cos(π(i+1)/M)) / 2       for 0 ≤ i < M       (left edge)
//! W[i] = W[N−1−i]                       for N−M ≤ i < N     (right edge, symmetric)
//! ```
//! The product W(x)·W(y) forms the 2D taper applied to each complex slice.
//!
//! ## References
//!
//! - Huijssen J, Verweij MD (2010). "An iterative method for the computation
//!   of nonlinear, wide-angle, pulsed acoustic fields of medical diagnostic
//!   transducers." J. Acoust. Soc. Am. 127(1), 33–44. DOI: 10.1121/1.3268599
//! - Berenger JP (1994). "A perfectly matched layer for the absorption of
//!   electromagnetic waves." J. Comput. Phys. 114(2), 185–200.

use std::f64::consts::PI;

use kwavers_math::fft::Complex64;
use leto::{Array2, ArrayViewMut2};
use moirai_parallel::{enumerate_mut_with, Adaptive};

/// Raised-cosine sponge window applied at each transverse boundary.
#[derive(Debug)]
pub struct SpongeLayer {
    /// Precomputed 2D taper W(x, y) = wx[i] * wy[j], shape [nx, ny].
    window: Array2<f64>,
}

impl SpongeLayer {
    /// Construct a sponge layer for the given transverse grid.
    #[must_use]
    pub fn new(nx: usize, ny: usize, fraction: f64) -> Self {
        debug_assert!(fraction > 0.0 && fraction <= 0.5);

        let wx = Self::build_taper(nx, fraction);
        let wy = Self::build_taper(ny, fraction);
        let mut window = Array2::<f64>::zeros((nx, ny));

        for i in 0..nx {
            for j in 0..ny {
                window[[i, j]] = wx[i] * wy[j];
            }
        }

        Self { window }
    }

    fn build_taper(n: usize, fraction: f64) -> Vec<f64> {
        let mut taper = vec![1.0_f64; n];
        let m = ((fraction * n as f64).round() as usize).clamp(1, n / 2);

        for i in 0..m {
            let weight = 0.5 * (1.0 - (PI * (i + 1) as f64 / m as f64).cos());
            taper[i] = weight;
            taper[n - 1 - i] = weight;
        }

        taper
    }

    /// Apply the sponge layer in place with no hot-path allocation.
    pub fn apply(&self, field: &mut ArrayViewMut2<Complex64>) {
        let window = self
            .window
            .as_slice()
            .expect("invariant: sponge window is standard-layout");
        if let Some(field_values) = field.as_mut_slice() {
            enumerate_mut_with::<Adaptive, _, _>(field_values, |idx, value| {
                *value *= window[idx];
            });
        } else {
            for (([_, _], value), &weight) in field
                .reborrow()
                .indexed_iter_mut()
                .expect("invariant: 2-D field view yields indexed iterator")
                .zip(window.iter())
            {
                *value *= weight;
            }
        }
    }
}
