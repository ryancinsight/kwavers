//! Arbitrary-even-order staggered gradient/divergence pair for a Yee leapfrog.
//!
//! # What this is for
//!
//! A velocity–pressure leapfrog needs two operators that are **negative
//! adjoints**, `D = −Gᵀ`, or it has no conserved energy (see
//! the second-order pair's divergence). `StaggeredGridOperator` supplies
//! that pair at second order. This supplies it at any even order, which is what
//! a high-accuracy ultrasound FDTD wants: Fullwave 2.5 runs eighth order in
//! space, and the point of the higher order is fewer points per wavelength for
//! the same phase error, not a smaller residual on a fixed grid.
//!
//! # The pair
//!
//! With `N = order/2` tap pairs and the staggered coefficients `cₙ` from
//! `staggered_first_derivative_coefficients`, the gradient maps cell-centred
//! `p` to face-centred `u` (face `i+½` stored at index `i`), and the divergence
//! maps back:
//!
//! ```text
//!   G:  u[i] = (1/Δx) Σₙ cₙ ( p[i+n] − p[i−n+1] )       taps reflected at the walls
//!   D:  −Gᵀ                                             by construction
//! ```
//!
//! Only the gradient has a stencil. The divergence is *defined* as its negative
//! transpose rather than written down separately, so `D = −Gᵀ` holds identically
//! — at every order, every grid size, and every boundary — instead of holding
//! because a closure argument works out. In the interior the transpose is the
//! familiar half-cell-shifted stencil `d[j] = (1/Δx) Σₙ cₙ (u[j+n−1] − u[j−n])`;
//! near a wall it is whatever the transpose of the reflected gradient is, which
//! is exactly the closure that conserves energy.
//!
//! # Boundary condition: a rigid wall, by even reflection
//!
//! A tap falling outside the grid is **reflected**, `p[−1] = p[0]` and
//! `p[nx] = p[nx−1]`, mirroring about the wall rather than vanishing at it. That
//! is `∂p/∂n = 0`: a rigid wall, the conventional acoustic default.
//!
//! The property that matters in practice is that **a uniform field has exactly
//! zero gradient**, at every order and every extent. Zero-extension does not
//! have it — it is a *pressure-release* wall, so `p` steps to zero half a cell
//! outside and the stencil sees that step. The consequence is not subtle: an
//! `N × 4 × 4` slab stops being a 1-D line and becomes a 4-cell-wide soft
//! waveguide. A purely axial packet launched into one had more energy in the
//! transverse velocity than the axial within 150 steps, ran at about half speed,
//! and never coherently arrived (KW-SOL-085). Quasi-1-D grids are the dominant
//! modelling idiom here, so the wall that leaves them inert is the right default.
//!
//! Reflection also makes the far velocity face vanish on its own: at `i = nx−1`
//! every tap pair becomes `p[nx+n−1] − p[nx−n]`, which reflection maps to
//! `p[nx−n] − p[nx−n] = 0`. Where `StaggeredGridOperator` forced that face to
//! zero as a separate step, here it is a consequence. A PML operates inside the
//! domain and leaves the wall intact.
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

    /// Strides and the per-axis geometry for linear indexing.
    ///
    /// The arrays are row-major with the last axis contiguous (verified against
    /// `as_slice`), so an offset of one step along `axis` is a fixed stride.
    /// Resolving that once per point instead of recomputing a three-index
    /// address per *tap* is the whole optimization: the address arithmetic, not
    /// the arithmetic on the values, dominated these kernels (KW-SOL-089).
    fn linear_geometry(axis: Axis, shape: [usize; 3]) -> ([usize; 3], usize, [usize; 2]) {
        let strides = [shape[1] * shape[2], shape[2], 1];
        let index = match axis {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        };
        let others = match index {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        (strides, index, others)
    }

    /// Gradient along `axis`: cell-centred `field` to face-centred `dst`, with
    /// face `i+½` stored at index `i`. Both are grid-shaped.
    ///
    /// Taps outside the grid are reflected about the wall, giving `∂p/∂n = 0`.
    pub fn gradient_into(&self, axis: Axis, field: ArrayView3<'_, f64>, dst: &mut Array3<f64>) {
        let (index, extent, scale) = self.axis_geometry(axis, field.shape(), dst.shape());
        let halo = self.coefficients.len() as isize;
        let shape = field.shape();
        let (strides, _, others) = Self::linear_geometry(axis, shape);
        let stride = strides[index] as isize;

        let (Some(source), Some(target)) = (field.as_slice(), dst.as_slice_mut()) else {
            self.gradient_into_indexed(axis, field, dst);
            return;
        };

        for a in 0..shape[others[0]] {
            for b in 0..shape[others[1]] {
                let line = a * strides[others[0]] + b * strides[others[1]];
                for along in 0..extent as usize {
                    let here = along as isize;
                    let linear = line + along * strides[index];
                    let mut sum = 0.0;
                    if here >= halo - 1 && here + halo < extent {
                        // No tap can leave the grid, so the mirror never fires
                        // and every address is `linear ± n·stride`.
                        let base = linear as isize;
                        for (offset, &c) in self.coefficients.iter().enumerate() {
                            let n = offset as isize + 1;
                            let hi = (base + n * stride) as usize;
                            let lo = (base + (1 - n) * stride) as usize;
                            sum += c * (source[hi] - source[lo]);
                        }
                    } else {
                        for (offset, &c) in self.coefficients.iter().enumerate() {
                            let n = offset as isize + 1;
                            let hi = line + reflect(here + n, extent) * strides[index];
                            let lo = line + reflect(here - n + 1, extent) * strides[index];
                            sum += c * (source[hi] - source[lo]);
                        }
                    }
                    target[linear] = sum * scale;
                }
            }
        }
    }

    /// Gradient via three-index addressing, for arrays that are not contiguous.
    fn gradient_into_indexed(&self, axis: Axis, field: ArrayView3<'_, f64>, dst: &mut Array3<f64>) {
        let (index, extent, scale) = self.axis_geometry(axis, field.shape(), dst.shape());
        for i in 0..field.shape()[0] {
            for j in 0..field.shape()[1] {
                for k in 0..field.shape()[2] {
                    let base = [i, j, k];
                    let here = base[index] as isize;
                    let mut sum = 0.0;
                    for (offset, &c) in self.coefficients.iter().enumerate() {
                        let n = offset as isize + 1;
                        let mut hi = base;
                        hi[index] = reflect(here + n, extent);
                        let mut lo = base;
                        lo[index] = reflect(here - n + 1, extent);
                        sum += c * (field[hi] - field[lo]);
                    }
                    dst[base] = sum * scale;
                }
            }
        }
    }

    /// Divergence along `axis`: face-centred `field` back to cell-centred `dst`.
    ///
    /// This is `−Gᵀ` applied directly, which is why it scatters where the
    /// gradient gathers: each face sends `∓cₙ` of its value to the two cells the
    /// gradient drew from, reflected indices included. Writing the transpose out
    /// as its own stencil would mean re-deriving the wall closure and hoping it
    /// matches; scattering makes `D = −Gᵀ` true by construction, so energy
    /// conservation does not depend on getting a boundary case right.
    pub fn divergence_into(&self, axis: Axis, field: ArrayView3<'_, f64>, dst: &mut Array3<f64>) {
        let (index, extent, scale) = self.axis_geometry(axis, field.shape(), dst.shape());
        let halo = self.coefficients.len() as isize;
        let shape = field.shape();
        let (strides, _, others) = Self::linear_geometry(axis, shape);
        let stride = strides[index] as isize;
        dst.fill(0.0);

        let (Some(source), Some(target)) = (field.as_slice(), dst.as_slice_mut()) else {
            self.divergence_into_indexed(axis, field, dst);
            return;
        };

        for a in 0..shape[others[0]] {
            for b in 0..shape[others[1]] {
                let line = a * strides[others[0]] + b * strides[others[1]];
                for along in 0..extent as usize {
                    let here = along as isize;
                    let linear = line + along * strides[index];
                    let value = source[linear] * scale;
                    if here >= halo - 1 && here + halo < extent {
                        let base = linear as isize;
                        for (offset, &c) in self.coefficients.iter().enumerate() {
                            let n = offset as isize + 1;
                            target[(base + n * stride) as usize] -= c * value;
                            target[(base + (1 - n) * stride) as usize] += c * value;
                        }
                    } else {
                        for (offset, &c) in self.coefficients.iter().enumerate() {
                            let n = offset as isize + 1;
                            target[line + reflect(here + n, extent) * strides[index]] -= c * value;
                            target[line + reflect(here - n + 1, extent) * strides[index]] +=
                                c * value;
                        }
                    }
                }
            }
        }
    }

    /// Divergence via three-index addressing, for arrays that are not contiguous.
    fn divergence_into_indexed(
        &self,
        axis: Axis,
        field: ArrayView3<'_, f64>,
        dst: &mut Array3<f64>,
    ) {
        let (index, extent, scale) = self.axis_geometry(axis, field.shape(), dst.shape());
        for i in 0..field.shape()[0] {
            for j in 0..field.shape()[1] {
                for k in 0..field.shape()[2] {
                    let base = [i, j, k];
                    let here = base[index] as isize;
                    let value = field[base] * scale;
                    for (offset, &c) in self.coefficients.iter().enumerate() {
                        let n = offset as isize + 1;
                        let mut hi = base;
                        hi[index] = reflect(here + n, extent);
                        let mut lo = base;
                        lo[index] = reflect(here - n + 1, extent);
                        dst[hi] -= c * value;
                        dst[lo] += c * value;
                    }
                }
            }
        }
    }

    /// Axis index, extent along it, and the reciprocal spacing.
    fn axis_geometry(&self, axis: Axis, shape: [usize; 3], dst: [usize; 3]) -> (usize, isize, f64) {
        debug_assert_eq!(dst, shape, "gradient/divergence output is grid-shaped");
        let index = match axis {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        };
        (index, shape[index] as isize, 1.0 / self.spacing[index])
    }
}

/// Mirror an index about the nearest wall until it lands inside `[0, extent)`.
///
/// Cell centres sit at `(i+½)Δ`, so the walls fall *between* cells and the
/// mirror is `−1−m` at the low end and `2·extent−1−m` at the high end — no cell
/// is its own reflection. The loop repeats for stencils deeper than the grid,
/// which only arises for extents below the halo width; it terminates for any
/// `extent ≥ 1`.
fn reflect(mut m: isize, extent: isize) -> usize {
    debug_assert!(extent >= 1, "reflection needs a non-empty axis");
    loop {
        if m < 0 {
            m = -1 - m;
        } else if m >= extent {
            m = 2 * extent - 1 - m;
        } else {
            return m as usize;
        }
    }
}

#[cfg(test)]
mod tests;
