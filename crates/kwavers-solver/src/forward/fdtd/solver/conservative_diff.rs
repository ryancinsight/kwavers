//! Skew-symmetric collocated first derivative for the FDTD leapfrog.
//!
//! # Why the general central difference is not usable here
//!
//! [`kwavers_math::numerics::operators::CentralDifference2`] and its higher-order
//! siblings close the boundary with **one-sided** differences, which is the
//! right choice for differentiating an arbitrary field: it stays consistent at
//! the edge instead of pretending the field vanishes outside.
//!
//! It is the wrong choice inside a leapfrog. A one-sided row puts a non-zero
//! entry on the operator's diagonal, so the matrix is no longer skew-symmetric,
//! so the discrete gradient and divergence stop being negative adjoints, so the
//! scheme has no conserved energy and the boundary rows pump it in. Measured on
//! a lossless standing wave over two thousand steps: energy grew by a factor of
//! 1.3·10⁴.
//!
//! # The closure
//!
//! Out-of-range taps are treated as **zero** rather than replaced by a
//! one-sided formula:
//!
//! ```text
//!   ∂f/∂x |_i ≈ (1/Δx) Σ_{n=1..N} cₙ · ( f_{i+n} − f_{i−n} ),   f_j = 0 for j ∉ [0, n)
//! ```
//!
//! The central coefficients `cₙ` are antisymmetric, so with this closure the
//! matrix is exactly skew-symmetric (`Gᵀ = −G`) for every order and every grid
//! size.
//!
//! # The wall this gives, which is not a rigid one
//!
//! A field vanishing outside the domain is a **pressure-release** wall, not a
//! rigid one. Conservative, but not inert: a transversely uniform field has a
//! non-zero transverse gradient at the wall, so a thin `N × 4 × 4` slab behaves
//! as a soft-walled waveguide rather than a 1-D line (KW-SOL-085).
//!
//! The staggered path fixed that by reflecting taps instead of zeroing them.
//! **That fix does not transfer here**, and the reason is structural rather than
//! incidental: reflection folds `f[−1] = f[0]` back onto row 0, putting a
//! non-zero entry on the diagonal — and a skew-symmetric matrix has a zero
//! diagonal by definition. So on a collocated grid, reflection and conservation
//! are in direct conflict, exactly as one-sided closures are. Recovering a rigid
//! wall here needs summation-by-parts operators, which is a much larger piece of
//! work; it is filed as KW-SOL-086 rather than approximated. Until then the
//! staggered path is the one to use for quasi-1-D work, and it is the default.
//!
//! The coefficients are derived, not tabulated, by
//! [`central_first_derivative_coefficients`], so an order is a parameter rather
//! than another hand-entered table.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::numerics::operators::central_first_derivative_coefficients;
use leto::{Array3, ArrayView3};

/// Collocated central first-derivative operator with the zero-extension
/// closure, per axis.
#[derive(Debug, Clone)]
pub(crate) struct ConservativeCentralDifference {
    coefficients: Vec<f64>,
    dx: f64,
    dy: f64,
    dz: f64,
}

impl ConservativeCentralDifference {
    /// Build for an even accuracy `order` (2, 4, or 6) and grid spacings.
    ///
    /// # Errors
    /// Rejects an odd or zero order and non-positive spacings.
    pub(crate) fn new(order: usize, dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if order == 0 || !order.is_multiple_of(2) {
            return Err(KwaversError::InvalidInput(format!(
                "conservative central difference needs an even order, got {order}"
            )));
        }
        if !(dx > 0.0 && dy > 0.0 && dz > 0.0) {
            return Err(KwaversError::InvalidInput(
                "conservative central difference needs positive grid spacings".to_owned(),
            ));
        }
        Ok(Self {
            coefficients: central_first_derivative_coefficients(order / 2)?,
            dx,
            dy,
            dz,
        })
    }

    /// `∂f/∂x` into `dst`, which must be grid-shaped.
    pub(crate) fn apply_x_into(&self, field: ArrayView3<'_, f64>, dst: &mut Array3<f64>) {
        let [nx, ny, nz] = field.shape();
        let scale = 1.0 / self.dx;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut sum = 0.0;
                    for (offset, &c) in self.coefficients.iter().enumerate() {
                        let n = offset + 1;
                        let high = if i + n < nx {
                            field[[i + n, j, k]]
                        } else {
                            0.0
                        };
                        let low = if i >= n { field[[i - n, j, k]] } else { 0.0 };
                        sum += c * (high - low);
                    }
                    dst[[i, j, k]] = sum * scale;
                }
            }
        }
    }

    /// `∂f/∂y` into `dst`.
    pub(crate) fn apply_y_into(&self, field: ArrayView3<'_, f64>, dst: &mut Array3<f64>) {
        let [nx, ny, nz] = field.shape();
        let scale = 1.0 / self.dy;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut sum = 0.0;
                    for (offset, &c) in self.coefficients.iter().enumerate() {
                        let n = offset + 1;
                        let high = if j + n < ny {
                            field[[i, j + n, k]]
                        } else {
                            0.0
                        };
                        let low = if j >= n { field[[i, j - n, k]] } else { 0.0 };
                        sum += c * (high - low);
                    }
                    dst[[i, j, k]] = sum * scale;
                }
            }
        }
    }

    /// `∂f/∂z` into `dst`.
    pub(crate) fn apply_z_into(&self, field: ArrayView3<'_, f64>, dst: &mut Array3<f64>) {
        let [nx, ny, nz] = field.shape();
        let scale = 1.0 / self.dz;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut sum = 0.0;
                    for (offset, &c) in self.coefficients.iter().enumerate() {
                        let n = offset + 1;
                        let high = if k + n < nz {
                            field[[i, j, k + n]]
                        } else {
                            0.0
                        };
                        let low = if k >= n { field[[i, j, k - n]] } else { 0.0 };
                        sum += c * (high - low);
                    }
                    dst[[i, j, k]] = sum * scale;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
