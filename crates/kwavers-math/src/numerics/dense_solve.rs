//! Small dense linear solves for deriving finite-difference coefficients.
//!
//! Both users here derive their stencils rather than carrying transcribed
//! tables, which is the point: a table copied from a paper is unverifiable at
//! the call site and silently wrong if mistyped, whereas a derived coefficient
//! set can be checked against the conditions it was built to satisfy.
//!
//! The systems are tiny — at most a few dozen unknowns — so a dense direct
//! solve is the right tool and the cost is irrelevant beside the clarity.

use kwavers_core::error::{KwaversError, KwaversResult};

/// Gaussian elimination with partial pivoting; `rhs` receives the solution.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] if the matrix is singular. `context`
/// names the system in that message, since the caller knows what it was solving
/// and this function does not.
pub(crate) fn solve_in_place(
    matrix: &mut [f64],
    rhs: &mut [f64],
    n: usize,
    context: &str,
) -> KwaversResult<()> {
    for col in 0..n {
        // Pivot on the largest magnitude in the column.
        let mut pivot = col;
        for row in (col + 1)..n {
            if matrix[row * n + col].abs() > matrix[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if matrix[pivot * n + col] == 0.0 {
            return Err(KwaversError::InvalidInput(format!("{context} is singular")));
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

/// Minimum-norm least-squares solution of `A x = b`, via ridge-regularised
/// normal equations `(AᵀA + λI) x = Aᵀb`.
///
/// One routine covers both shapes because the summation-by-parts systems are
/// over-determined but *consistent* — more accuracy conditions than free
/// coefficients, with an exact solution among them — while some orders leave
/// free parameters. Ridge picks the minimum-norm member either way, and the
/// caller is expected to check the residual rather than trust the solve: a
/// system with no exact solution would otherwise return a plausible-looking
/// vector that satisfies none of the conditions.
///
/// `ridge` must be small enough not to perturb the answer and large enough to
/// regularise the rank deficiency; `1e-12` relative to the column norms is the
/// scale that suits these systems.
///
/// # Errors
/// Returns [`KwaversError::InvalidInput`] if the regularised system is singular
/// or the shapes do not agree.
pub(crate) fn solve_least_squares(
    a: &[f64],
    b: &[f64],
    rows: usize,
    cols: usize,
    ridge: f64,
    context: &str,
) -> KwaversResult<Vec<f64>> {
    if a.len() != rows * cols || b.len() != rows {
        return Err(KwaversError::InvalidInput(format!(
            "{context}: system shape {rows}x{cols} does not match the data supplied"
        )));
    }

    let mut normal = vec![0.0_f64; cols * cols];
    let mut projected = vec![0.0_f64; cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut acc = 0.0;
            for row in 0..rows {
                acc += a[row * cols + i] * a[row * cols + j];
            }
            normal[i * cols + j] = acc;
        }
        normal[i * cols + i] += ridge;
        let mut acc = 0.0;
        for row in 0..rows {
            acc += a[row * cols + i] * b[row];
        }
        projected[i] = acc;
    }

    solve_in_place(&mut normal, &mut projected, cols, context)?;
    Ok(projected)
}

/// Largest absolute residual of `A x − b`.
///
/// The acceptance test for a derived coefficient set: the conditions either
/// hold or the derivation is rejected.
pub(crate) fn residual_norm(a: &[f64], b: &[f64], x: &[f64], rows: usize, cols: usize) -> f64 {
    (0..rows).fold(0.0_f64, |worst, row| {
        let predicted: f64 = (0..cols).map(|c| a[row * cols + c] * x[c]).sum();
        worst.max((predicted - b[row]).abs())
    })
}

#[cfg(test)]
mod tests;
