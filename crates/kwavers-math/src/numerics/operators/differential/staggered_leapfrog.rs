//! Arbitrary-even-order staggered gradient/divergence pair for a Yee leapfrog.
//!
//! # What this is for
//!
//! A velocity–pressure leapfrog needs two operators that are **negative
//! adjoints**, `D = −Gᵀ`, or it has no conserved energy (see
//! [`super::staggered_grid::divergence`]). [`StaggeredGridOperator`] supplies
//! that pair at second order. This supplies it at any even order, which is what
//! a high-accuracy ultrasound FDTD wants: Fullwave 2.5 runs eighth order in
//! space, and the point of the higher order is fewer points per wavelength for
//! the same phase error, not a smaller residual on a fixed grid.
//!
//! # The pair
//!
//! With `N = order/2` tap pairs and the staggered coefficients `cₙ` from
//! [`staggered_first_derivative_coefficients`], the gradient maps cell-centred
//! `p` to face-centred `u` (face `i+½` stored at index `i`), and the divergence
//! maps back:
//!
//! ```text
//!   G:  u[i] = (1/Δx) Σₙ cₙ ( p[i+n]   − p[i−n+1] )
//!   D:  d[j] = (1/Δx) Σₙ cₙ ( u[j+n−1] − u[j−n]   )
//! ```
//!
//! Note the one-index shift between them: that is the half-cell stagger, and it
//! is what makes the transpose work out. Both operators treat taps outside the
//! grid as **zero**.
//!
//! # Why that is exactly adjoint
//!
//! Expand `⟨Gp, u⟩` and re-index each term:
//!
//! ```text
//!   ⟨Gp, u⟩ = (1/Δx) Σₙ cₙ [ Σᵢ u[i]p[i+n] − Σᵢ u[i]p[i−n+1] ]
//!           = (1/Δx) Σₙ cₙ  Σⱼ p[j] ( u[j−n] − u[j+n−1] )        (j = i+n, j = i−n+1)
//!           = −⟨p, Du⟩
//! ```
//!
//! The re-indexing is only valid if the sums run over all integers, which is
//! exactly what zero-extension provides — a tap that falls outside contributes
//! nothing on either side. So the identity is exact for **every** order and grid
//! size, not asymptotically. `N = 1` reduces to the familiar
//! `(p[i+1] − p[i])/Δx` and `(u[j] − u[j−1])/Δx`.
//!
//! # Boundary condition
//!
//! Zero-extension means both fields vanish outside the grid: a closed domain.
//! It differs from [`StaggeredGridOperator`]'s pairing, which instead forces the
//! far velocity face to zero — both are conservative, but they are different
//! walls, so the two are not interchangeable mid-simulation. A PML operates
//! inside the domain and leaves either intact.
//!
//! # References
//!
//! - Virieux, J. (1986). "P-SV wave propagation in heterogeneous media."
//!   *Geophysics* 51(4), 889–901.
//! - Levander, A.R. (1988). "Fourth-order finite-difference P-SV seismograms."
//!   *Geophysics* 53(11), 1425–1436.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array3, ArrayView3};

use super::staggered_grid::staggered_first_derivative_coefficients;

/// Which of the three axes an operator acts along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// First (slowest-varying) axis.
    X,
    /// Second axis.
    Y,
    /// Third (fastest-varying) axis.
    Z,
}

/// Staggered gradient/divergence pair of a given even order.
#[derive(Debug, Clone)]
pub struct StaggeredLeapfrogOperator {
    coefficients: Vec<f64>,
    spacing: [f64; 3],
}

impl StaggeredLeapfrogOperator {
    /// Build for an even accuracy `order` and per-axis grid spacings.
    ///
    /// # Errors
    /// Rejects an odd or zero order, an order beyond the coefficient
    /// derivation's verified range, and non-positive spacings.
    pub fn new(order: usize, dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if order == 0 || !order.is_multiple_of(2) {
            return Err(KwaversError::InvalidInput(format!(
                "staggered leapfrog operator needs an even order, got {order}"
            )));
        }
        if !(dx > 0.0 && dy > 0.0 && dz > 0.0) {
            return Err(KwaversError::InvalidInput(
                "staggered leapfrog operator needs positive grid spacings".to_owned(),
            ));
        }
        Ok(Self {
            coefficients: staggered_first_derivative_coefficients(order / 2)?,
            spacing: [dx, dy, dz],
        })
    }

    /// Accuracy order `2N`.
    #[must_use]
    pub fn order(&self) -> usize {
        2 * self.coefficients.len()
    }

    /// Half-width of the stencil in cells — the halo a domain decomposition
    /// must exchange for this order.
    #[must_use]
    pub fn halo_width(&self) -> usize {
        self.coefficients.len()
    }

    /// Courant limit as a multiple of `Δx/c`, for `dimensions` spatial axes.
    ///
    /// # Derivation
    ///
    /// The staggered symbol along one axis is
    /// `S(θ) = 2 Σₙ cₙ sin((n−½)θ)`, whose magnitude is bounded by
    /// `S_max = 2 Σₙ |cₙ|`. Leapfrog stability needs
    /// `(c·Δt/2)·|k_eff| ≤ 1` with `|k_eff| = S_max·√D/Δx` in `D` dimensions,
    /// so
    ///
    /// ```text
    ///   Δt ≤ 2Δx / (c · S_max · √D) = Δx / (c · √D · Σₙ|cₙ|)
    /// ```
    ///
    /// At order 2 the sum is 1 and this recovers the familiar `1/√3` in 3-D.
    ///
    /// # Why this is not `AcousticSpatialOrder::cfl_limit`
    ///
    /// That table (`1/√3`, `1/√15`, `1/√27`) is the **collocated**
    /// central-difference limit. The two agree at order 2, which is why the
    /// distinction went unnoticed, but diverge immediately after: at order 4 the
    /// staggered limit is 0.495 against the collocated 0.258. Using the
    /// collocated number for a staggered run costs roughly half the achievable
    /// step for no accuracy gain.
    #[must_use]
    pub fn cfl_limit(&self, dimensions: usize) -> f64 {
        let sum: f64 = self.coefficients.iter().map(|c| c.abs()).sum();
        1.0 / ((dimensions as f64).sqrt() * sum)
    }

    /// Gradient along `axis`: cell-centred `field` to face-centred `dst`, with
    /// face `i+½` stored at index `i`. Both are grid-shaped.
    pub fn gradient_into(&self, axis: Axis, field: ArrayView3<'_, f64>, dst: &mut Array3<f64>) {
        // Taps at +n and −n+1 relative to the output index.
        self.apply(axis, field, dst, |n| (n as isize, 1 - n as isize));
    }

    /// Divergence along `axis`: face-centred `field` back to cell-centred `dst`.
    pub fn divergence_into(&self, axis: Axis, field: ArrayView3<'_, f64>, dst: &mut Array3<f64>) {
        // Taps at +n−1 and −n: the gradient's shifted by one, which is the
        // half-cell stagger and what makes `D = −Gᵀ`.
        self.apply(axis, field, dst, |n| (n as isize - 1, -(n as isize)));
    }

    /// Shared kernel: `dst[i] = (1/Δ) Σₙ cₙ (field[i+hi(n)] − field[i+lo(n)])`,
    /// with out-of-range taps contributing zero.
    fn apply(
        &self,
        axis: Axis,
        field: ArrayView3<'_, f64>,
        dst: &mut Array3<f64>,
        taps: impl Fn(usize) -> (isize, isize),
    ) {
        let shape = field.shape();
        debug_assert_eq!(
            dst.shape(),
            shape,
            "gradient/divergence output is grid-shaped"
        );
        let index = match axis {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        };
        let extent = shape[index] as isize;
        let scale = 1.0 / self.spacing[index];

        // Reading `field` at a shifted position along one axis.
        let at = |base: [usize; 3], shifted: isize| -> f64 {
            if shifted < 0 || shifted >= extent {
                return 0.0;
            }
            let mut probe = base;
            probe[index] = shifted as usize;
            field[probe]
        };

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let base = [i, j, k];
                    let here = base[index] as isize;
                    let mut sum = 0.0;
                    for (offset, &c) in self.coefficients.iter().enumerate() {
                        let (hi, lo) = taps(offset + 1);
                        sum += c * (at(base, here + hi) - at(base, here + lo));
                    }
                    dst[base] = sum * scale;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
