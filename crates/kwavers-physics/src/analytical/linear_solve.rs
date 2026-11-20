//! Dense `A·x = b` solve shared by the analytical kernels.
//!
//! The Tikhonov/Born helpers in [`super::inverse`] and the spectroscopic
//! unmixing in [`super::photoacoustics`] each carried the same
//! partial-pivoting elimination: the same pivot rule, the same `1e-300`
//! singularity floor and the same back substitution, one over a flat row-major
//! block and one rebuilding a `Vec<f64>` per row. One flat implementation now
//! serves both, so the mechanics exist once and a solve costs one allocation
//! instead of one per row.

/// Solve `A·x = b` for a flat row-major `n × n` matrix and a length-`n` `b`.
///
/// Returns a zero vector when a pivot magnitude falls at or below the
/// singularity floor `1e-300` — the behaviour both former copies had.
///
/// # Panics
///
/// Panics when `a` is shorter than `n * n` or `b` shorter than `n`.
#[allow(clippy::needless_range_loop)]
pub(super) fn solve_flat(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // One augmented block of `n` rows with stride `n + 1`: the pivot search,
    // the row swap and every elimination pass walk contiguous memory.
    let stride = n + 1;
    let mut aug = vec![0.0_f64; n * stride];
    for row in 0..n {
        aug[row * stride..row * stride + n].copy_from_slice(&a[row * n..(row + 1) * n]);
        aug[row * stride + n] = b[row];
    }

    for col in 0..n {
        // Partial pivot
        let pivot = (col..n)
            .max_by(|&i, &j| {
                aug[i * stride + col]
                    .abs()
                    .total_cmp(&aug[j * stride + col].abs())
            })
            .expect("invariant: a pivot search runs over a non-empty row range");
        if pivot != col {
            for offset in 0..stride {
                aug.swap(col * stride + offset, pivot * stride + offset);
            }
        }
        let diag = aug[col * stride + col];
        if diag.abs() < 1e-300 {
            return vec![0.0; n];
        }
        for row in (col + 1)..n {
            let factor = aug[row * stride + col] / diag;
            for k in col..=n {
                let val = aug[col * stride + k] * factor;
                aug[row * stride + k] -= val;
            }
        }
    }
    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        x[i] = aug[i * stride + n];
        for j in (i + 1)..n {
            x[i] -= aug[i * stride + j] * x[j];
        }
        x[i] /= aug[i * stride + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::solve_flat;

    #[test]
    fn a_two_by_two_system_solves_exactly() {
        // [[2, 1], [1, 3]]·[1, 3] = [5, 10]
        let x = solve_flat(&[2.0, 1.0, 1.0, 3.0], &[5.0, 10.0], 2);
        assert_eq!(x, vec![1.0, 3.0]);
    }

    #[test]
    fn an_identity_block_returns_the_right_hand_side() {
        let x = solve_flat(&[1.0, 0.0, 0.0, 1.0], &[2.5, -4.0], 2);
        assert_eq!(x, vec![2.5, -4.0]);
    }

    #[test]
    fn a_singular_system_returns_zeros() {
        // Second row is a multiple of the first, so elimination finds a zero pivot.
        let x = solve_flat(&[1.0, 2.0, 2.0, 4.0], &[1.0, 2.0], 2);
        assert_eq!(x, vec![0.0, 0.0]);
    }

    #[test]
    fn a_zero_dimension_system_is_empty() {
        assert!(solve_flat(&[], &[], 0).is_empty());
    }
}
