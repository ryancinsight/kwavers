//! Deterministic row/source encoding for same-aperture linear operators.
//!
//! # Theorem: encoded normal equations
//!
//! Let `A in R^(m x n)` be a finite-frequency same-aperture operator and
//! `C in R^(k x m)` be a deterministic encoding matrix whose nonzero entries
//! are normalized signs. The encoded operator is `B = C A`. For any model
//! vector `x`, `B x = C(A x)`. For any encoded residual `y`, `B^T y =
//! A^T C^T y`. Therefore solving the encoded normal equation uses
//! `B^T B = A^T C^T C A` exactly for the selected encoding; it is a compressed
//! linear inverse, not nonlinear full-waveform inversion.

use rayon::prelude::*;

use super::linear_operator::LinearOperator;
use super::row_matrix::RowMatrix;

#[derive(Clone, Debug)]
pub struct EncodedOperator<O> {
    inner: O,
    spec: EncodingSpec,
}

#[derive(Clone, Debug)]
pub struct EncodingSpec {
    original_rows: usize,
    encoded_rows: usize,
    rows_per_code: usize,
    entries: Vec<EncodedEntry>,
}

#[derive(Clone, Copy, Debug)]
struct EncodedEntry {
    source_row: usize,
    sign: f32,
}

impl EncodingSpec {
    #[must_use]
    pub fn deterministic_signs(original_rows: usize, rows_per_code: usize) -> Self {
        let rows_per_code = rows_per_code.max(1);
        let encoded_rows = original_rows.div_ceil(rows_per_code);
        let mut entries = Vec::with_capacity(original_rows);
        for encoded_row in 0..encoded_rows {
            let start = encoded_row * rows_per_code;
            let end = (start + rows_per_code).min(original_rows);
            let scale = ((end - start) as f32).sqrt().recip();
            for row in start..end {
                let phase = row.wrapping_mul(0x9E37_79B1) ^ encoded_row;
                let sign = if phase.count_ones() & 1 == 0 {
                    1.0
                } else {
                    -1.0
                };
                entries.push(EncodedEntry {
                    source_row: row,
                    sign: sign * scale,
                });
            }
        }
        Self {
            original_rows,
            encoded_rows,
            rows_per_code,
            entries,
        }
    }

    #[must_use]
    pub fn original_rows(&self) -> usize {
        self.original_rows
    }

    #[must_use]
    pub fn encoded_rows(&self) -> usize {
        self.encoded_rows
    }

    #[must_use]
    pub fn rows_per_code(&self) -> usize {
        self.rows_per_code
    }

    fn group_range(&self, encoded_row: usize) -> std::ops::Range<usize> {
        let start = encoded_row * self.rows_per_code;
        let end = (start + self.rows_per_code).min(self.original_rows);
        start..end
    }
}

impl<O: LinearOperator + Sync> EncodedOperator<O> {
    #[must_use]
    pub fn deterministic_signs(inner: O, rows_per_code: usize) -> Self {
        let spec = EncodingSpec::deterministic_signs(inner.rows(), rows_per_code);
        Self { inner, spec }
    }

    #[must_use]
    pub fn encoding_spec(&self) -> &EncodingSpec {
        &self.spec
    }

    #[must_use]
    pub fn into_inner(self) -> O {
        self.inner
    }

    #[must_use]
    pub fn materialize(&self) -> RowMatrix {
        let mut matrix = RowMatrix::zeros(self.rows(), self.cols());
        matrix
            .data
            .par_chunks_mut(self.cols())
            .enumerate()
            .for_each(|(row, values)| self.row_values(row, values));
        matrix
    }
}

impl<O: LinearOperator + Sync> LinearOperator for EncodedOperator<O> {
    fn rows(&self) -> usize {
        self.spec.encoded_rows
    }

    fn cols(&self) -> usize {
        self.inner.cols()
    }

    fn matvec(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols());
        debug_assert_eq!(out.len(), self.rows());
        let mut inner_out = vec![0.0_f32; self.inner.rows()];
        self.inner.matvec(x, &mut inner_out);
        out.par_iter_mut().enumerate().for_each(|(encoded, dst)| {
            let mut sum = 0.0_f32;
            for idx in self.spec.group_range(encoded) {
                let entry = self.spec.entries[idx];
                sum += entry.sign * inner_out[entry.source_row];
            }
            *dst = sum;
        });
    }

    fn t_matvec(&self, y: &[f32], out: &mut [f32]) {
        debug_assert_eq!(y.len(), self.rows());
        debug_assert_eq!(out.len(), self.cols());
        let mut lifted = vec![0.0_f32; self.inner.rows()];
        for (encoded, value) in y.iter().copied().enumerate() {
            for idx in self.spec.group_range(encoded) {
                let entry = self.spec.entries[idx];
                lifted[entry.source_row] += entry.sign * value;
            }
        }
        self.inner.t_matvec(&lifted, out);
    }

    fn row_values(&self, row: usize, out: &mut [f32]) {
        debug_assert!(row < self.rows());
        debug_assert_eq!(out.len(), self.cols());
        let mut scratch = vec![0.0_f32; self.cols()];
        self.row_values_with_scratch(row, out, &mut scratch);
    }

    /// Compute the diagonal of `B^T B` where `B` is the encoded operator.
    ///
    /// # Algorithm
    ///
    /// `(B^T B)_{jj} = sum_i B_{ij}^2`. Each encoded row `i` contributes
    /// its squared column values to the accumulator. Rows are independent,
    /// so the sum parallelizes with Rayon's fold-reduce pattern.
    /// Each Rayon thread maintains a partial `cols`-length accumulator,
    /// avoiding contention. The final reduce sums partial accumulators.
    ///
    /// Total memory: `O(threads * cols)` — e.g. 8 threads × 4096 cols × 4 B ≈ 128 KB.
    fn normal_diag(&self) -> Vec<f32> {
        use rayon::prelude::*;
        let cols = self.cols();
        (0..self.rows())
            .into_par_iter()
            .fold(
                || vec![0.0_f32; cols],
                |mut acc, encoded| {
                    let mut row = vec![0.0_f32; cols];
                    let mut scratch = vec![0.0_f32; cols];
                    self.row_values_with_scratch(encoded, &mut row, &mut scratch);
                    for (dst, v) in acc.iter_mut().zip(row.iter()) {
                        *dst += v * v;
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0_f32; cols],
                |mut a, b| {
                    for (av, bv) in a.iter_mut().zip(b.iter()) {
                        *av += bv;
                    }
                    a
                },
            )
    }

    fn storage_values(&self) -> usize {
        self.inner.storage_values() + self.spec.entries.len() * 2 + 3
    }
}

impl<O: LinearOperator + Sync> EncodedOperator<O> {
    fn row_values_with_scratch(&self, row: usize, out: &mut [f32], scratch: &mut [f32]) {
        debug_assert!(row < self.rows());
        debug_assert_eq!(out.len(), self.cols());
        debug_assert_eq!(scratch.len(), self.cols());
        out.fill(0.0);
        for idx in self.spec.group_range(row) {
            let entry = self.spec.entries[idx];
            self.inner.row_values(entry.source_row, scratch);
            for (dst, value) in out.iter_mut().zip(scratch.iter()) {
                *dst += entry.sign * *value;
            }
        }
    }
}

#[must_use]
pub fn encode_measurements<O: LinearOperator + Sync>(
    operator: &EncodedOperator<O>,
    data: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(data.len(), operator.encoding_spec().original_rows());
    let mut encoded = vec![0.0_f32; operator.rows()];
    for (row, dst) in encoded.iter_mut().enumerate() {
        *dst = operator
            .encoding_spec()
            .group_range(row)
            .map(|idx| {
                let entry = operator.encoding_spec().entries[idx];
                entry.sign * data[entry.source_row]
            })
            .sum();
    }
    encoded
}
