//! Compressed Sparse Row (CSR) matrix format
//!
//! This module provides [`CompressedSparseRowMatrix<T>`] for acoustic simulations
//! requiring `Complex64` support and mutable in-place assembly (BEM/FEM).
//!
//! # Migration path to `leto::infrastructure::sparse`
//!
//! New code that only needs `f64` entries and read-only matvec should prefer
//! [`leto::sparse::CsrArray<T>`](leto::infrastructure::sparse::CsrArray) which is
//! the Atlas SSOT sparse substrate and participates in the coeus autodiff graph.
//!
//! Conversion bridges below (`From` impls) allow gradual per-callsite migration:
//!
//! ```ignore
//! use leto::sparse::CsrArray;
//! use kwavers_math::linear_algebra::sparse::CompressedSparseRowMatrix;
//!
//! let kw: CompressedSparseRowMatrix<f64> = build_matrix();
//! let leto: CsrArray<f64> = CsrArray::from(&kw);  // zero-copy metadata
//! let roundtrip: CompressedSparseRowMatrix<f64> = CompressedSparseRowMatrix::from(&leto);
//! ```

use kwavers_core::error::KwaversResult;
use leto::{Array1, ArrayView1, ArrayView2};
use std::ops::{AddAssign, Mul};

/// Compressed Sparse Row matrix format
#[derive(Debug, Clone)]
pub struct CompressedSparseRowMatrix<T = f64> {
    /// Number of rows
    pub rows: usize,
    /// Number of columns  
    pub cols: usize,
    /// Non-zero values
    pub values: Vec<T>,
    /// Column indices for each value
    pub col_indices: Vec<usize>,
    /// Row pointers (start index for each row)
    pub row_pointers: Vec<usize>,
    /// Number of non-zero elements
    pub nnz: usize,
}

pub trait CsrScalar: Copy + Default + AddAssign + Mul<Output = Self> {
    fn magnitude(self) -> f64;
    fn zero() -> Self;
}

impl CompressedSparseRowMatrix<f64> {
    /// Create CSR matrix from dense matrix with sparsity threshold
    #[must_use]
    pub fn from_dense(dense: ArrayView2<f64>, threshold: f64) -> Self {
        let [rows, cols] = dense.shape();
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_pointers = vec![0; rows + 1];

        for i in 0..rows {
            row_pointers[i] = values.len();
            for j in 0..cols {
                if dense[[i, j]].abs() > threshold {
                    values.push(dense[[i, j]]);
                    col_indices.push(j);
                }
            }
        }
        row_pointers[rows] = values.len();

        let nnz = values.len();
        Self {
            rows,
            cols,
            values,
            col_indices,
            row_pointers,
            nnz,
        }
    }
}

impl CsrScalar for f64 {
    fn magnitude(self) -> f64 {
        self.abs()
    }

    fn zero() -> Self {
        0.0
    }
}

impl CsrScalar for eunomia::Complex64 {
    fn magnitude(self) -> f64 {
        self.norm()
    }

    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl<T> CompressedSparseRowMatrix<T> {
    /// Create CSR matrix with specified dimensions
    #[must_use]
    pub fn create(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_pointers: vec![0; rows + 1],
            nnz: 0,
        }
    }

    /// Create CSR matrix with pre-allocated capacity
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Self {
        Self {
            rows,
            cols,
            values: Vec::with_capacity(capacity),
            col_indices: Vec::with_capacity(capacity),
            row_pointers: vec![0; rows + 1],
            nnz: 0,
        }
    }

    /// Matrix-vector multiplication: y = A * x
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn multiply_vector(&self, x: ArrayView1<T>) -> KwaversResult<Array1<T>>
    where
        T: CsrScalar,
    {
        if x.shape()[0] != self.cols {
            return Err(kwavers_core::error::KwaversError::Numerical(
                kwavers_core::error::NumericalError::Instability {
                    operation: "csr_matvec".to_owned(),
                    condition: x.shape()[0] as f64,
                },
            ));
        }

        let mut y = Array1::from_elem([self.rows], T::zero());

        for i in 0..self.rows {
            let mut sum = T::zero();
            for j in self.row_pointers[i]..self.row_pointers[i + 1] {
                sum += self.values[j] * x[self.col_indices[j]];
            }
            y[i] = sum;
        }

        Ok(y)
    }

    /// Get row as slice
    #[must_use]
    pub fn get_row(&self, row: usize) -> (&[T], &[usize]) {
        let start = self.row_pointers[row];
        let end = self.row_pointers[row + 1];
        (&self.values[start..end], &self.col_indices[start..end])
    }

    /// Compute Frobenius norm
    #[must_use]
    pub fn frobenius_norm(&self) -> f64
    where
        T: CsrScalar,
    {
        self.values
            .iter()
            .map(|&v| v.magnitude().powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Get sparsity ratio
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        1.0 - (self.nnz as f64) / ((self.rows * self.cols) as f64)
    }

    /// Convert to dense matrix
    #[must_use]
    pub fn to_dense(&self) -> leto::Array2<T>
    where
        T: Copy + /*Zero*/ Default,
    {
        let mut dense = leto::Array2::from_elem([self.rows, self.cols], T::default());

        for i in 0..self.rows {
            for j in self.row_pointers[i]..self.row_pointers[i + 1] {
                dense[[i, self.col_indices[j]]] = self.values[j];
            }
        }

        dense
    }

    /// Add value to matrix at (row, col) position
    pub fn add_value(&mut self, row: usize, col: usize, value: T)
    where
        T: Copy + AddAssign,
    {
        // Find position in row
        let row_start = self.row_pointers[row];
        let row_end = self.row_pointers[row + 1];

        // Check if entry already exists
        for i in row_start..row_end {
            if self.col_indices[i] == col {
                self.values[i] += value;
                return;
            }
        }

        // Insert new entry (simple insertion - not optimal for performance)
        self.values.insert(row_end, value);
        self.col_indices.insert(row_end, col);

        // Update row pointers for subsequent rows
        for i in (row + 1)..=self.rows {
            self.row_pointers[i] += 1;
        }

        self.nnz += 1;
    }

    /// Set diagonal entry
    pub fn set_diagonal(&mut self, row: usize, value: T)
    where
        T: Copy + AddAssign,
    {
        self.add_value(row, row, value);
    }

    /// Get diagonal entry
    #[must_use]
    pub fn get_diagonal(&self, row: usize) -> T
    where
        T: Copy + /*Zero*/ Default,
    {
        let row_start = self.row_pointers[row];
        let row_end = self.row_pointers[row + 1];

        for i in row_start..row_end {
            if self.col_indices[i] == row {
                return self.values[i];
            }
        }

        T::default()
    }

    /// Zero out row (except diagonal)
    pub fn zero_row_off_diagonals(&mut self, row: usize) {
        let row_start = self.row_pointers[row];

        let mut i = row_start;
        while i < self.row_pointers[row + 1] {
            if self.col_indices[i] != row {
                // Remove off-diagonal entry
                self.values.remove(i);
                self.col_indices.remove(i);

                // Update row pointers for subsequent rows
                for r in (row + 1)..=self.rows {
                    self.row_pointers[r] -= 1;
                }

                self.nnz -= 1;
                // Don't increment i since we removed an element
            } else {
                i += 1;
            }
        }
    }

    /// Zero out entire row
    pub fn zero_row(&mut self, row: usize) {
        let row_start = self.row_pointers[row];
        let row_end = self.row_pointers[row + 1];
        let num_to_remove = row_end - row_start;

        // Remove all entries in row
        self.values.drain(row_start..row_end);
        self.col_indices.drain(row_start..row_end);

        // Update row pointers for subsequent rows
        for r in (row + 1)..=self.rows {
            self.row_pointers[r] -= num_to_remove;
        }

        self.nnz -= num_to_remove;
    }

    /// Compress matrix by removing near-zero entries
    pub fn compress(&mut self, tolerance: f64)
    where
        T: CsrScalar,
    {
        let mut i = 0;
        while i < self.values.len() {
            if self.values[i].magnitude() < tolerance {
                // Remove entry
                self.values.remove(i);
                self.col_indices.remove(i);

                // Update row pointers - need to find which row this was in
                for r in 0..self.rows {
                    if i >= self.row_pointers[r] && i < self.row_pointers[r + 1] {
                        // Update subsequent row pointers
                        for s in (r + 1)..=self.rows {
                            self.row_pointers[s] -= 1;
                        }
                        break;
                    }
                }

                self.nnz -= 1;
                // Don't increment i since we removed an element
            } else {
                i += 1;
            }
        }
    }
}

// ── Leto sparse interop ────────────────────────────────────────────────────────

/// Convert a `CompressedSparseRowMatrix<f64>` into a leto `CsrArray<f64>`.
///
/// Triplets are reconstructed from the CSR storage, sorted by row and column,
/// and fed into `CsrArray::from_coo` — a single allocation. This path is
/// intentionally explicit (not zero-cost) to signal that it is a migration aid
/// rather than a production hot-path: callers that need repeated matvec should
/// hold the `CsrArray` directly.
impl From<&CompressedSparseRowMatrix<f64>> for leto::CsrArray<f64> {
    fn from(kw: &CompressedSparseRowMatrix<f64>) -> Self {
        use leto::SparseStorageMut;
        let mut coo = leto::CooArray::with_capacity(kw.rows, kw.cols, kw.nnz);
        for row in 0..kw.rows {
            for idx in kw.row_pointers[row]..kw.row_pointers[row + 1] {
                coo.add(row, kw.col_indices[idx], kw.values[idx]);
            }
        }
        leto::CsrArray::from_coo(coo)
    }
}

/// Convert a leto `CsrArray<f64>` back into a `CompressedSparseRowMatrix<f64>`.
///
/// Useful when migrated code needs to hand a result back to an unconverted caller.
impl From<&leto::CsrArray<f64>> for CompressedSparseRowMatrix<f64> {
    fn from(csr: &leto::CsrArray<f64>) -> Self {
        use leto::SparseStorage;
        let rows = csr.nrows();
        let cols = csr.ncols();
        let mut out = CompressedSparseRowMatrix::with_capacity(rows, cols, csr.nnz());
        for row in 0..rows {
            out.row_pointers[row] = out.values.len();
            for (col, &value) in csr.row_entries(row) {
                out.values.push(value);
                out.col_indices.push(col);
                out.nnz += 1;
            }
        }
        out.row_pointers[rows] = out.values.len();
        out
    }
}

#[cfg(test)]
mod tests {
    //! CsrScalar-magnitude + cross-type default-equivalence regression pins.
    //!
    //! Drift surfaces as a numerical failure (not a silent compile regression)
    //! when a future change touches:
    //! - the per-impl `CsrScalar::magnitude` body;
    //! - the `eunomia::ComplexField` blanket-impl wiring through
    //!   `Complex<T>::norm()` (csr.rs → eunomia::impls::field.rs:148-149);
    //! - the `eunomia::Complex64` value-domain invariants defined in
    //!   `eunomia::impls::field`.
    //!
    //! Comparison strategy: `assert_eq!` for the integer-square-root-of-
    //! perfect-square cases (bit-exact under IEEE 754). No transcendental
    //! `magnitude` cases in the fixture set — magnitude always evaluates
    //! `sqrt(re² + im²)` over integer operands, and all chosen operands
    //! produce bit-exact outputs (3²+4²=25, 5²+12²=169, 6²+8²=100).

    use super::CsrScalar;

    // ───── `CsrScalar::magnitude` over `eunomia::Complex64` (native surface) ─────

    /// USER-VERBATIM: `|3+4i| = 5`.
    #[test]
    fn csr_scalar_magnitude_eunomia_complex_3_4_5() {
        // 3² + 4² = 25; sqrt(25) = 5 — bit-exact under IEEE 754.
        let z = eunomia::Complex64::new(3.0, 4.0);
        assert_eq!(CsrScalar::magnitude(z), 5.0);
    }

    /// 5/12/13 Pythagorean triple.
    #[test]
    fn csr_scalar_magnitude_eunomia_complex_5_12_13() {
        let z = eunomia::Complex64::new(5.0, 12.0);
        assert_eq!(CsrScalar::magnitude(z), 13.0);
    }

    /// Sign-invariance (all four quadrants return the same magnitude).
    #[test]
    fn csr_scalar_magnitude_eunomia_complex_sign_invariant_all_quadrants() {
        let q1 = eunomia::Complex64::new(6.0, 8.0);
        let q2 = eunomia::Complex64::new(-6.0, 8.0);
        let q3 = eunomia::Complex64::new(6.0, -8.0);
        let q4 = eunomia::Complex64::new(-6.0, -8.0);
        assert_eq!(CsrScalar::magnitude(q1), 10.0);
        assert_eq!(CsrScalar::magnitude(q2), 10.0);
        assert_eq!(CsrScalar::magnitude(q3), 10.0);
        assert_eq!(CsrScalar::magnitude(q4), 10.0);
    }

    /// |0+0i| = 0 — zero-complex baseline.
    #[test]
    fn csr_scalar_magnitude_eunomia_complex_zero_complex() {
        let z = eunomia::Complex64::new(0.0, 0.0);
        assert_eq!(CsrScalar::magnitude(z), 0.0);
    }

    /// |1+0i| = 1 — also pin `default()` form (different construction route).
    #[test]
    fn csr_scalar_magnitude_eunomia_complex_one_real_imag_zero() {
        let z = eunomia::Complex64::new(1.0, 0.0);
        assert_eq!(CsrScalar::magnitude(z), 1.0);
        assert_eq!(CsrScalar::magnitude(eunomia::Complex64::default()), 0.0);
    }

    /// |0+7i| = 7 — pure-imaginary branch (real-component zero path).
    #[test]
    fn csr_scalar_magnitude_eunomia_complex_pure_imaginary() {
        let z = eunomia::Complex64::new(0.0, 7.0);
        assert_eq!(CsrScalar::magnitude(z), 7.0);
    }

    // ───── `CsrScalar::magnitude` over `f64` (real-degenerate case) ─────

    #[test]
    fn csr_scalar_magnitude_f64_positive() {
        assert_eq!(CsrScalar::magnitude(3.5_f64), 3.5);
    }

    #[test]
    fn csr_scalar_magnitude_f64_negative() {
        // |−3.5| = 3.5 — pins f64::abs() body.
        assert_eq!(CsrScalar::magnitude(-3.5_f64), 3.5);
    }

    #[test]
    fn csr_scalar_magnitude_f64_zero() {
        assert_eq!(CsrScalar::magnitude(0.0_f64), 0.0);
    }

    #[test]
    fn csr_scalar_magnitude_f64_one() {
        assert_eq!(CsrScalar::magnitude(1.0_f64), 1.0);
    }

    // ───── Cross-type default-equivalence (eunomia default) ─────

    /// USER-VERBATIM: `eunomia::Complex64::default()` pin. `re`/`im` are both
    /// zero and remain aligned with the expected physical convention.
    #[test]
    fn eunomia_default_complex64_field_by_field_equals_eunomia_default() {
        let e = eunomia::Complex64::default();
        assert_eq!(e.re, 0.0_f64);
        assert_eq!(e.im, 0.0_f64);
        // Canonical comparison stays field-level because `Complex64` uses a
        // public representation contract (`re`, `im`).
    }

    /// Pin `<eunomia::Complex<f64> as ComplexField>::modulus()` directly.
    /// This is the SSOT-source of `CsrScalar::magnitude` post-ADR §2.
    /// Compiles today: `eunomia::ComplexField::modulus` is implemented
    /// for `Complex<f64>` via the blanket
    /// `impl<T: RealField> ComplexField for Complex<T>` at
    /// `crates/eunomia/src/impls/field.rs:148-149` (calls `self.norm()`).
    #[test]
    fn complex_field_modulus_eunomia_complex_3_4_5() {
        let z = eunomia::Complex64::new(3.0, 4.0);
        assert_eq!(
            <eunomia::Complex64 as eunomia::ComplexField>::modulus(z),
            5.0
        );
        assert_eq!(
            <eunomia::Complex64 as eunomia::ComplexField>::modulus_squared(z),
            25.0,
        );
    }

    // ── Leto interop bridge roundtrip ──────────────────────────────────────────

    fn three_by_three_sample() -> super::CompressedSparseRowMatrix<f64> {
        use super::CompressedSparseRowMatrix;
        let mut m = CompressedSparseRowMatrix::create(3, 3);
        m.add_value(0, 0, 1.0);
        m.add_value(0, 2, 2.0);
        m.add_value(1, 1, 3.0);
        m.add_value(2, 0, 4.0);
        m.add_value(2, 2, 5.0);
        m
    }

    #[test]
    fn kwavers_csr_to_leto_csr_preserves_entries() {
        use leto::SparseStorage;
        let kw = three_by_three_sample();
        let leto_csr = leto::CsrArray::from(&kw);

        assert_eq!(leto_csr.nrows(), 3);
        assert_eq!(leto_csr.ncols(), 3);
        assert_eq!(leto_csr.nnz(), 5);
        assert_eq!(leto_csr.get(0, 0), Some(1.0));
        assert_eq!(leto_csr.get(0, 2), Some(2.0));
        assert_eq!(leto_csr.get(1, 1), Some(3.0));
        assert_eq!(leto_csr.get(2, 0), Some(4.0));
        assert_eq!(leto_csr.get(2, 2), Some(5.0));
        assert_eq!(leto_csr.get(0, 1), None);
    }

    #[test]
    fn leto_csr_to_kwavers_csr_roundtrip() {
        use super::CompressedSparseRowMatrix;
        let kw = three_by_three_sample();
        let leto_csr = leto::CsrArray::from(&kw);
        let back = CompressedSparseRowMatrix::from(&leto_csr);

        assert_eq!(back.rows, kw.rows);
        assert_eq!(back.cols, kw.cols);
        assert_eq!(back.nnz, kw.nnz);
        // Entries are sorted by column within each row after the roundtrip.
        let (vals0, cols0) = back.get_row(0);
        assert_eq!(vals0, &[1.0, 2.0]);
        assert_eq!(cols0, &[0_usize, 2_usize]);
        let (vals1, cols1) = back.get_row(1);
        assert_eq!(vals1, &[3.0]);
        assert_eq!(cols1, &[1_usize]);
    }
}
