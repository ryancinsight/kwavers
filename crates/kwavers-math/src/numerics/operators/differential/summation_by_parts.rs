//! Summation-by-parts first derivative: a collocated operator that is both
//! conservative and compatible with a rigid wall.
//!
//! # The problem this exists to solve
//!
//! A collocated leapfrog uses one derivative operator for both the gradient and
//! the divergence, so its energy behaviour is governed entirely by that
//! operator's symmetry. Closing the stencil by treating out-of-range taps as
//! zero makes it exactly skew-symmetric, `Dᵀ = −D`, which conserves energy — but
//! a field vanishing outside the domain is a *pressure-release* wall. Under it a
//! transversely uniform field has a non-zero gradient at the boundary, so a thin
//! slab behaves as a soft-walled waveguide instead of the 1-D line it is meant
//! to model (KW-SOL-085).
//!
//! The staggered path fixed that by reflecting taps at the wall. **That does not
//! transfer to a collocated grid**: reflection folds `f[−1] = f[0]` back onto row
//! zero, putting a non-zero entry on the diagonal, and a skew-symmetric matrix
//! has a zero diagonal by definition. Reflection and conservation are in direct
//! conflict here, exactly as one-sided closures are.
//!
//! # The resolution: conserve in a weighted norm
//!
//! Summation by parts drops the demand that `D` be skew-symmetric in the plain
//! inner product and asks instead for a positive diagonal norm `H` with
//!
//! ```text
//!   D = H⁻¹Q     and     Q + Qᵀ = B = diag(−1, 0, …, 0, +1)
//! ```
//!
//! `B` is the discrete analogue of the boundary term in integration by parts,
//! which is where the name comes from. For the acoustic pair `p_t = −K·D·u`,
//! `u_t = −ρ⁻¹·D·p`, the energy in the `H`-weighted norm obeys
//!
//! ```text
//!   d/dt ‖E‖_H = −pᵀ(Q + Qᵀ)u = −pᵀB u = −( p_{N−1}u_{N−1} − p₀u₀ )
//! ```
//!
//! so imposing `u = 0` at the two end points — the rigid wall — conserves it
//! exactly. Nothing is given up: `H` is the trapezoidal quadrature weight, half
//! at the end points and one inside, so the conserved quantity is a *better*
//! discretisation of the energy integral than the unweighted sum it replaces.
//!
//! Critically, every SBP operator satisfies `D·1 = 0` by construction — its rows
//! sum to zero because it is at least first-order accurate. A uniform field has
//! zero gradient, which is the property the whole exercise is for. The
//! coefficients are solved numerically rather than carried as exact rationals,
//! so that zero is to round-off (`~5e-12` relative) rather than bitwise — five
//! orders below anything a structural defect would produce.
//!
//! # Diagonal-norm accuracy
//!
//! A diagonal norm caps boundary accuracy at half the interior order: interior
//! `2m` pairs with boundary `m`, giving global order `m + 1`. That is the known
//! and accepted cost of keeping `H` diagonal, which in turn is what keeps the
//! energy a simple weighted sum rather than requiring a norm solve every step.
//!
//! # Derived, not transcribed
//!
//! The boundary blocks are solved from the accuracy and symmetry conditions at
//! construction rather than carried as tables copied from the literature. A
//! mistyped table entry is silently wrong and unverifiable at the call site; a
//! derived block can be — and is — checked against the conditions it was built
//! to satisfy, and the construction fails loudly if they do not hold. See
//! [`SummationByPartsOperator::new`].
//!
//! # References
//!
//! - Kreiss, H.-O., & Scherer, G. (1974). "Finite element and finite difference
//!   methods for hyperbolic partial differential equations." In *Mathematical
//!   Aspects of Finite Elements in Partial Differential Equations*, 195–212.
//! - Strand, B. (1994). "Summation by parts for finite difference approximations
//!   for d/dx." *Journal of Computational Physics*, 110(1), 47–67.
//! - Svärd, M., & Nordström, J. (2014). "Review of summation-by-parts schemes for
//!   initial-boundary-value problems." *Journal of Computational Physics*, 268,
//!   17–38.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array3, ArrayView3};

use super::central_first_derivative_coefficients;
use super::staggered_leapfrog::Axis;
use crate::numerics::dense_solve::{residual_norm, solve_least_squares};

/// Ridge used when solving the boundary conditions, and the bound the residual
/// must meet afterwards.
const DERIVATION_RIDGE: f64 = 1.0e-12;
/// The conditions are exactly satisfiable, so the residual is round-off. A
/// looser bound here would let a near-miss block through as if it were an SBP
/// operator, which is the one failure this derivation must not have.
const DERIVATION_TOLERANCE: f64 = 1.0e-9;

/// One axis's derived operator: interior stencil plus its boundary closure.
#[derive(Debug, Clone)]
struct AxisOperator {
    /// Interior central coefficients `c₁..c_m`, or empty when the axis is too
    /// short to carry a derivative at all.
    interior: Vec<f64>,
    /// Diagonal norm weights for the `rows` boundary points. Interior weight
    /// is 1.
    norm: Vec<f64>,
    /// The `rows × cols` boundary block of `Q`, row-major.
    block: Vec<f64>,
    rows: usize,
    cols: usize,
    /// Reciprocal grid spacing along this axis.
    inverse_spacing: f64,
}

impl AxisOperator {
    /// Zero operator, for an axis with no room for any stencil.
    ///
    /// A one-cell axis carries no resolvable wave, and its correct derivative is
    /// zero — which is also inert, so a thin transverse extent stays a 1-D line
    /// rather than becoming a boundary-dominated direction.
    fn inert(inverse_spacing: f64) -> Self {
        Self {
            interior: Vec::new(),
            norm: Vec::new(),
            block: Vec::new(),
            rows: 0,
            cols: 0,
            inverse_spacing,
        }
    }

    /// Norm weight at `index` along an axis of `extent` points.
    fn weight(&self, index: usize, extent: usize) -> f64 {
        if index < self.rows {
            self.norm[index]
        } else if index + self.rows >= extent {
            self.norm[extent - 1 - index]
        } else {
            1.0
        }
    }

    /// `(D f)[index]` along one axis, given a reader for the axis samples.
    fn derivative_at(&self, index: usize, extent: usize, sample: impl Fn(usize) -> f64) -> f64 {
        if self.interior.is_empty() {
            return 0.0;
        }
        if index < self.rows {
            let row = &self.block[index * self.cols..(index + 1) * self.cols];
            let sum: f64 = row.iter().enumerate().map(|(j, &q)| q * sample(j)).sum();
            return sum / self.norm[index] * self.inverse_spacing;
        }
        if index + self.rows >= extent {
            // SBP operators satisfy `D = −P D P` for the reversal `P`, so the
            // far block is the near block read backwards and negated. Deriving
            // it rather than storing it is what keeps the two ends from
            // drifting apart under an edit.
            let mirrored = extent - 1 - index;
            let row = &self.block[mirrored * self.cols..(mirrored + 1) * self.cols];
            let sum: f64 = row
                .iter()
                .enumerate()
                .map(|(j, &q)| q * sample(extent - 1 - j))
                .sum();
            return -sum / self.norm[mirrored] * self.inverse_spacing;
        }
        let sum: f64 = self
            .interior
            .iter()
            .enumerate()
            .map(|(offset, &c)| {
                let n = offset + 1;
                c * (sample(index + n) - sample(index - n))
            })
            .sum();
        sum * self.inverse_spacing
    }
}

/// Collocated summation-by-parts first derivative, per axis.
///
/// Each axis carries its own derivation because the boundary block needs room:
/// an axis shorter than twice the block cannot host one, and falls back to the
/// highest order that fits rather than failing the whole grid.
#[derive(Debug, Clone)]
pub struct SummationByPartsOperator {
    axes: [AxisOperator; 3],
    shape: [usize; 3],
    order: usize,
}

impl SummationByPartsOperator {
    /// Derive for an even interior `order` on a grid of `shape` with `spacing`.
    ///
    /// The grid shape is a parameter because an SBP operator is only defined
    /// where its boundary blocks fit without overlapping. An axis with too few
    /// points falls back to the highest even order that does fit, down to an
    /// inert zero operator for a single point — an accuracy reduction confined
    /// to an axis that cannot resolve a wave anyway, rather than a hard failure
    /// on a legitimate quasi-1-D grid.
    ///
    /// # Errors
    /// Rejects an odd or zero order and non-positive spacings, and — the case
    /// that matters — fails if a derived block does not satisfy the conditions
    /// it was solved from, rather than returning an operator that is not
    /// summation-by-parts.
    pub fn new(order: usize, shape: [usize; 3], spacing: [f64; 3]) -> KwaversResult<Self> {
        if order == 0 || !order.is_multiple_of(2) {
            return Err(KwaversError::InvalidInput(format!(
                "summation-by-parts operator needs an even order, got {order}"
            )));
        }
        if !spacing.iter().all(|d| *d > 0.0) {
            return Err(KwaversError::InvalidInput(
                "summation-by-parts operator needs positive grid spacings".to_owned(),
            ));
        }

        let mut axes = [
            AxisOperator::inert(1.0),
            AxisOperator::inert(1.0),
            AxisOperator::inert(1.0),
        ];
        for axis in 0..3 {
            let inverse_spacing = 1.0 / spacing[axis];
            let mut derived = AxisOperator::inert(inverse_spacing);
            // Highest interior order whose two boundary blocks fit disjointly.
            let mut candidate = order;
            while candidate >= 2 {
                let rows = boundary_rows(candidate / 2);
                let cols = boundary_columns(candidate / 2);
                if shape[axis] >= 2 * rows && shape[axis] >= cols {
                    derived = derive_axis(candidate, inverse_spacing)?;
                    break;
                }
                candidate -= 2;
            }
            axes[axis] = derived;
        }

        Ok(Self { axes, shape, order })
    }

    /// Interior accuracy order requested at construction.
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Interior order actually derived along `axis`, which is lower where the
    /// grid was too short to host the requested block.
    #[must_use]
    pub fn realized_order(&self, axis: Axis) -> usize {
        2 * self.axes[axis_index(axis)].interior.len()
    }

    /// Diagonal norm weight at `index` along `axis`.
    ///
    /// The energy quadrature must use these weights, not a plain sum: the
    /// conserved quantity is `Σ hᵢ Eᵢ`, and an unweighted sum is not conserved.
    /// End points carry half weight, which is the trapezoidal rule.
    #[must_use]
    pub fn norm_weight(&self, axis: Axis, index: usize) -> f64 {
        let index_axis = axis_index(axis);
        self.axes[index_axis].weight(index, self.shape[index_axis])
    }

    /// `∂f/∂axis` into `dst`, which must be grid-shaped.
    pub fn apply_into(&self, axis: Axis, field: ArrayView3<'_, f64>, dst: &mut Array3<f64>) {
        let shape = field.shape();
        debug_assert_eq!(dst.shape(), shape, "derivative output is grid-shaped");
        let index_axis = axis_index(axis);
        let operator = &self.axes[index_axis];
        let extent = shape[index_axis];

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let base = [i, j, k];
                    dst[base] = operator.derivative_at(base[index_axis], extent, |along| {
                        let mut probe = base;
                        probe[index_axis] = along;
                        field[probe]
                    });
                }
            }
        }
    }
}

fn axis_index(axis: Axis) -> usize {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

/// Number of boundary rows the closure modifies, for interior half-order `m`.
///
/// Order 2 modifies only the first row; above that the block is `2m` rows wide,
/// which is the standard diagonal-norm shape and the smallest that admits a
/// solution.
fn boundary_rows(half_order: usize) -> usize {
    if half_order == 1 {
        1
    } else {
        2 * half_order
    }
}

/// Number of columns each boundary row spans, for interior half-order `m`.
///
/// The block must reach `m` columns past its last row, because those columns are
/// fixed by antisymmetry against the first interior rows.
fn boundary_columns(half_order: usize) -> usize {
    boundary_rows(half_order) + half_order
}

/// Solve one axis's boundary closure from the SBP and accuracy conditions.
///
/// # The system
///
/// With `Q`'s boundary block written as `qᵢⱼ`, the unknowns are the norm weights
/// `h₀..h_{r−1}` and the strictly-upper triangle of the block's leading `r × r`
/// part. Everything else is fixed:
///
/// - `Q + Qᵀ = B` forces the leading block to be antisymmetric apart from
///   `q₀₀ = −½`, which supplies the `−1` in `B`.
/// - The same condition fixes columns `r..r+m` against the first interior rows:
///   interior row `j` holds `−c_{j−i}` in column `i`, so `qᵢⱼ = c_{j−i}`.
/// - Accuracy asks `D xᵏ = k xᵏ⁻¹` at each boundary row for `k = 0..m`, which in
///   terms of `Q` reads `Σⱼ qᵢⱼ jᵏ = hᵢ · k · iᵏ⁻¹`. The `k = 0` row is the
///   statement that `D` annihilates constants — the inertness property.
///
/// That is `r(m+1)` equations in `r + r(r−1)/2` unknowns: over-determined, and
/// consistent because SBP operators of these orders exist. The residual check
/// afterwards is what distinguishes "solved" from "least-squares fitted".
fn derive_axis(order: usize, inverse_spacing: f64) -> KwaversResult<AxisOperator> {
    let half = order / 2;
    let interior = central_first_derivative_coefficients(half)?;
    let rows = boundary_rows(half);
    let cols = boundary_columns(half);

    let free_pairs = rows * (rows - 1) / 2;
    let unknowns = rows + free_pairs;
    let equations = rows * (half + 1);

    // Index of the unknown holding `q[i][j]` for `i < j < rows`.
    let pair_index = |i: usize, j: usize| -> usize {
        let (i, j) = (i.min(j), i.max(j));
        // Offset of row `i`'s first strictly-upper entry, plus the column step.
        let preceding = i * rows - i * (i + 1) / 2;
        rows + preceding + (j - i - 1)
    };

    let mut matrix = vec![0.0_f64; equations * unknowns];
    let mut rhs = vec![0.0_f64; equations];

    for i in 0..rows {
        for k in 0..=half {
            let equation = i * (half + 1) + k;
            let row = &mut matrix[equation * unknowns..(equation + 1) * unknowns];

            // Columns inside the leading block: antisymmetric unknowns, plus the
            // fixed q00 which moves to the right-hand side.
            for j in 0..rows {
                let weight = monomial(j, k);
                if i == j {
                    if i == 0 {
                        rhs[equation] += 0.5 * weight;
                    }
                    continue;
                }
                let sign = if i < j { 1.0 } else { -1.0 };
                row[pair_index(i, j)] += sign * weight;
            }

            // Columns fixed by antisymmetry against the interior rows.
            for j in rows..cols {
                let distance = j - i;
                if distance >= 1 && distance <= half {
                    rhs[equation] -= interior[distance - 1] * monomial(j, k);
                }
            }

            // The `hᵢ · k · iᵏ⁻¹` term, moved to the left.
            if k >= 1 {
                row[i] -= (k as f64) * monomial(i, k - 1);
            }
        }
    }

    let solution = solve_least_squares(
        &matrix,
        &rhs,
        equations,
        unknowns,
        DERIVATION_RIDGE,
        "summation-by-parts boundary closure",
    )?;
    let residual = residual_norm(&matrix, &rhs, &solution, equations, unknowns);
    if residual > DERIVATION_TOLERANCE || residual.is_nan() {
        return Err(KwaversError::InvalidInput(format!(
            "summation-by-parts closure for order {order} does not satisfy its own \
             conditions: residual {residual:.3e}"
        )));
    }

    let norm: Vec<f64> = solution[..rows].to_vec();
    if !norm.iter().all(|h| *h > 0.0) {
        return Err(KwaversError::InvalidInput(format!(
            "summation-by-parts norm for order {order} is not positive definite: {norm:?}"
        )));
    }

    let mut block = vec![0.0_f64; rows * cols];
    for i in 0..rows {
        for j in 0..rows {
            block[i * cols + j] = if i == j {
                if i == 0 {
                    -0.5
                } else {
                    0.0
                }
            } else if i < j {
                solution[pair_index(i, j)]
            } else {
                -solution[pair_index(i, j)]
            };
        }
        for j in rows..cols {
            let distance = j - i;
            if distance >= 1 && distance <= half {
                block[i * cols + j] = interior[distance - 1];
            }
        }
    }

    Ok(AxisOperator {
        interior,
        norm,
        block,
        rows,
        cols,
        inverse_spacing,
    })
}

/// `baseᵏ` with the `0⁰ = 1` convention the accuracy conditions need.
fn monomial(base: usize, power: usize) -> f64 {
    if power == 0 {
        1.0
    } else {
        (base as f64).powi(power as i32)
    }
}

#[cfg(test)]
mod tests;
