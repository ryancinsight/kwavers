//! Staggered (half-grid) first-derivative stencil coefficients to arbitrary
//! even order, **derived** rather than tabulated.
//!
//! # The stencil
//!
//! On a Yee-staggered grid the derivative is evaluated halfway between samples.
//! With `N = half_order` pairs of taps,
//!
//! ```text
//!   ∂f/∂x |_{i+1/2} ≈ (1/Δx) Σ_{n=1..N} cₙ · ( f_{i+n} − f_{i−n+1} )
//! ```
//!
//! which is accurate to order `2N`. `N = 1` is the familiar
//! `(f_{i+1} − f_i)/Δx`; `N = 4` is the 8th-order stencil used by
//! high-accuracy ultrasound FDTD codes.
//!
//! # Why derived
//!
//! Expanding about the half-point, the taps sit at `±aₙΔx` with
//! `aₙ = n − ½`, and `f(+a) − f(−a)` is odd, so only odd derivatives survive:
//!
//! ```text
//!   (1/Δx) Σₙ cₙ [f(+aₙΔx) − f(−aₙΔx)]
//!     = 2 Σₙ cₙ [ aₙ f′ + aₙ³Δx² f‴/6 + aₙ⁵Δx⁴ f⁽⁵⁾/120 + … ]
//! ```
//!
//! Matching `f′` and annihilating the next `N−1` odd derivatives gives the
//! square linear system
//!
//! ```text
//!   Σₙ cₙ aₙ^{2m+1} = ½·δ_{m,0},     m = 0 … N−1
//! ```
//!
//! solved here by Gaussian elimination with partial pivoting. Deriving the
//! coefficients rather than tabulating them means a new order is a parameter,
//! not a new hand-entered constant table to get wrong — and the derivation is
//! checked against the published values for orders 2–8 plus a measured
//! order-of-accuracy test.
//!
//! # References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on
//!   arbitrarily spaced grids." *Mathematics of Computation* 51(184), 699–706.
//! - Levander, A.R. (1988). "Fourth-order finite-difference P-SV seismograms."
//!   *Geophysics* 53(11), 1425–1436. (The staggered 4th-order `9/8, −1/24`.)

use kwavers_core::error::{KwaversError, KwaversResult};

/// Largest supported half-order.
///
/// The Vandermonde-like system is increasingly ill-conditioned in `N`. At
/// `N = 4` (8th order — the highest a wave solver here profitably uses, and the
/// order Fullwave 2.5 runs) the coefficients match the published rationals to
/// `1e-13` relative. By `N = 8` the high Taylor moments cancel terms of order
/// `10^12` against each other and the residual is only `~1e-11` of the summed
/// magnitude — still far tighter than any discretization error, but no longer
/// exact. The cap keeps the derivation inside its verified range rather than
/// silently returning noise.
pub const MAX_HALF_ORDER: usize = 8;

/// Derive the staggered first-derivative coefficients `cₙ`, `n = 1…N`, for a
/// stencil of accuracy order `2·half_order`.
///
/// The returned coefficients are in units of `1/Δx`: multiply the tap
/// differences by `cₙ` and divide the sum by `Δx`.
///
/// # Errors
/// Rejects `half_order == 0` and `half_order > MAX_HALF_ORDER`, and reports a
/// singular system (which cannot occur for the well-posed `aₙ` used here, but
/// is surfaced rather than silently producing garbage).
///
/// # Examples
/// ```
/// use kwavers_math::numerics::operators::staggered_first_derivative_coefficients;
///
/// // Second order: the plain half-grid difference.
/// let c = staggered_first_derivative_coefficients(1).unwrap();
/// assert!((c[0] - 1.0).abs() < 1e-14);
///
/// // Fourth order: Levander's 9/8, -1/24.
/// let c = staggered_first_derivative_coefficients(2).unwrap();
/// assert!((c[0] - 9.0 / 8.0).abs() < 1e-14);
/// assert!((c[1] + 1.0 / 24.0).abs() < 1e-14);
/// ```
pub fn staggered_first_derivative_coefficients(half_order: usize) -> KwaversResult<Vec<f64>> {
    // Taps sit at the half-points a_j = j + 1/2.
    coefficients_for_offsets(half_order, |j| j as f64 + 0.5, "staggered")
}

/// Derive the **collocated** central first-derivative coefficients `c_n`,
/// `n = 1…N`, for a stencil of accuracy order `2·half_order`:
///
/// ```text
///   ∂f/∂x |_i ≈ (1/Δx) Σ_{n=1..N} cₙ · ( f_{i+n} − f_{i−n} )
/// ```
///
/// Same derivation as the staggered case with the taps at whole points
/// `aₙ = n` instead of half points, so `N = 1` gives the familiar `1/2`
/// (`(f_{i+1} − f_{i−1})/2Δx`) and `N = 2` gives `2/3, −1/12`.
///
/// The coefficients are **antisymmetric** by construction, which is what makes
/// the operator skew-symmetric — and therefore energy-conserving in a leapfrog —
/// once out-of-range taps are treated as zero rather than replaced by a
/// one-sided formula.
///
/// # Errors
/// Rejects `half_order == 0` and `half_order > MAX_HALF_ORDER`.
///
/// # Examples
/// ```
/// use kwavers_math::numerics::operators::central_first_derivative_coefficients;
///
/// let c = central_first_derivative_coefficients(1).unwrap();
/// assert!((c[0] - 0.5).abs() < 1e-14);
///
/// let c = central_first_derivative_coefficients(2).unwrap();
/// assert!((c[0] - 2.0 / 3.0).abs() < 1e-14);
/// assert!((c[1] + 1.0 / 12.0).abs() < 1e-14);
/// ```
pub fn central_first_derivative_coefficients(half_order: usize) -> KwaversResult<Vec<f64>> {
    // Taps sit at the whole points a_j = j + 1.
    coefficients_for_offsets(half_order, |j| j as f64 + 1.0, "central")
}

/// Shared derivation: solve `Σₙ cₙ aₙ^{2m+1} = ½·δ_{m,0}` for the given tap
/// offsets. Staggered and collocated stencils differ only in where the taps
/// sit, so the linear system and its solve are the same.
fn coefficients_for_offsets(
    half_order: usize,
    offset: impl Fn(usize) -> f64,
    kind: &str,
) -> KwaversResult<Vec<f64>> {
    if half_order == 0 || half_order > MAX_HALF_ORDER {
        return Err(KwaversError::InvalidInput(format!(
            "{kind} half-order must be 1..={MAX_HALF_ORDER}, got {half_order}"
        )));
    }
    let n = half_order;

    let offsets: Vec<f64> = (0..n).map(&offset).collect();
    let mut matrix = vec![0.0_f64; n * n];
    for m in 0..n {
        let power = (2 * m + 1) as i32;
        for (j, &a) in offsets.iter().enumerate() {
            matrix[m * n + j] = a.powi(power);
        }
    }
    // Only the f′ condition has a right-hand side.
    let mut rhs = vec![0.0_f64; n];
    rhs[0] = 0.5;

    solve_in_place(&mut matrix, &mut rhs, n)?;
    Ok(rhs)
}

/// Gaussian elimination with partial pivoting; `rhs` receives the solution.
fn solve_in_place(matrix: &mut [f64], rhs: &mut [f64], n: usize) -> KwaversResult<()> {
    for col in 0..n {
        // Pivot on the largest magnitude in the column.
        let mut pivot = col;
        for row in (col + 1)..n {
            if matrix[row * n + col].abs() > matrix[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if matrix[pivot * n + col] == 0.0 {
            return Err(KwaversError::InvalidInput(
                "staggered coefficient system is singular".to_owned(),
            ));
        }
        if pivot != col {
            for k in 0..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }

        let diagonal = matrix[col * n + col];
        for row in (col + 1)..n {
            let factor = matrix[row * n + col] / diagonal;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution.
    for col in (0..n).rev() {
        let mut acc = rhs[col];
        for k in (col + 1)..n {
            acc -= matrix[col * n + k] * rhs[k];
        }
        rhs[col] = acc / matrix[col * n + col];
    }
    Ok(())
}

#[cfg(test)]
mod tests;
