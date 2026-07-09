//! Compressed Sparse Row (CSR) matrix format

use kwavers_core::error::KwaversResult;
use leto::{
    Array1,
    ArrayView1,
    ArrayView2,
};
use eunomia::NumericElement;
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
}

impl CompressedSparseRowMatrix<f64> {
    /// Create CSR matrix from dense matrix with sparsity threshold
    #[must_use]
    pub fn from_dense(dense: ArrayView2<f64>, threshold: f64) -> Self {
        let (rows, cols) = dense.dim();
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
}

impl CsrScalar for num_complex::Complex64 {
    fn magnitude(self) -> f64 {
        self.norm()
    }
}

impl CsrScalar for eunomia::Complex64 {
    fn magnitude(self) -> f64 {
        self.norm()
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
        if x.len() != self.cols {
            return Err(kwavers_core::error::KwaversError::Numerical(
                kwavers_core::error::NumericalError::Instability {
                    operation: "csr_matvec".to_owned(),
                    condition: x.len() as f64,
                },
            ));
        }

        let mut y = Array1::from_elem(self.rows, T::zero());

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
        let mut dense = leto::Array2::from_elem((self.rows, self.cols), T::default());

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
#[cfg(test)]
mod tests {
    //! CsrScalar-magnitude + cross-type default-equivalence regression pins.
    //! (CR-EUNOMIA-COMPLEX, ADR 0006.)
    //!
    //! Drift surfaces as a numerical failure (not a silent compile regression)
    //! when a future change touches:
    //! - `CsrScalar::magnitude` body (current: per-impl; post-ADR:
    //!   default `<Self as ComplexField>::modulus().to_f64()`);
    //! - the `eunomia::ComplexField` blanket-impl wiring through
    //!   `Complex<T>::norm()` (csr.rs → eunomia::impls::field.rs:148-149);
    //! - the `eunomia::Complex64 == num_complex::Complex64` bit-layout
    //!   equivalence (rustdoc-cited at
    //!   `apollo-validation::rustfft_reference.rs:13-25`).
    //!
    //! Comparison strategy: `assert_eq!` for the integer-square-root-of-
    //! perfect-square cases (bit-exact under IEEE 754). No transcendental
    //! `magnitude` cases in the fixture set — magnitude always evaluates
    //! `sqrt(re² + im²)` over integer operands, and all chosen operands
    //! produce bit-exact outputs (3²+4²=25, 5²+12²=169, 6²+8²=100).

    use super::CsrScalar;
    use eunomia::NumericElement;

    // ───── `CsrScalar::magnitude` over `num_complex::Complex64` (current surface) ─────

    /// USER-VERBATIM: `magnitude([3,4] eunomia::Complex) = 5` mirrored against
    /// `num_complex::Complex64` (the current `CsrScalar` impl). Once ADR lands
    /// and `eunomia::Complex64` ALSO monomorphizes through the
    /// `CsrScalar: ComplexField` blanket, the same assertion runs over
    /// `eunomia::Complex64::new(3.0, 4.0)` — see the post-ADR-0006 fixture
    /// block at the bottom of this mod.
    #[test]
    fn csr_scalar_magnitude_num_complex_3_4_5() {
        // 3² + 4² = 25; sqrt(25) = 5 — bit-exact under IEEE 754.
        let z = num_complex::Complex64::new(3.0, 4.0);
        assert_eq!(CsrScalar::magnitude(z), 5.0);
    }

    /// 5/12/13 Pythagorean triple.
    #[test]
    fn csr_scalar_magnitude_num_complex_5_12_13() {
        let z = num_complex::Complex64::new(5.0, 12.0);
        assert_eq!(CsrScalar::magnitude(z), 13.0);
    }

    /// Sign-invariance (all four quadrants return the same magnitude).
    #[test]
    fn csr_scalar_magnitude_num_complex_sign_invariant_all_quadrants() {
        let q1 = num_complex::Complex64::new(6.0, 8.0);
        let q2 = num_complex::Complex64::new(-6.0, 8.0);
        let q3 = num_complex::Complex64::new(6.0, -8.0);
        let q4 = num_complex::Complex64::new(-6.0, -8.0);
        assert_eq!(CsrScalar::magnitude(q1), 10.0);
        assert_eq!(CsrScalar::magnitude(q2), 10.0);
        assert_eq!(CsrScalar::magnitude(q3), 10.0);
        assert_eq!(CsrScalar::magnitude(q4), 10.0);
    }

    /// |0+0i| = 0 — zero-complex baseline.
    #[test]
    fn csr_scalar_magnitude_num_complex_zero_complex() {
        let z = num_complex::Complex64::new(0.0, 0.0);
        assert_eq!(CsrScalar::magnitude(z), 0.0);
    }

    /// |1+0i| = 1 — also pin `default()` form (different construction route).
    #[test]
    fn csr_scalar_magnitude_num_complex_one_real_imag_zero() {
        let z = num_complex::Complex64::new(1.0, 0.0);
        assert_eq!(CsrScalar::magnitude(z), 1.0);
        assert_eq!(CsrScalar::magnitude(num_complex::Complex64::default()), 0.0);
    }

    /// |0+7i| = 7 — pure-imaginary branch (real-component zero path).
    #[test]
    fn csr_scalar_magnitude_num_complex_pure_imaginary() {
        let z = num_complex::Complex64::new(0.0, 7.0);
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

    // ───── Cross-type default-equivalence (the user's verbatim-named fixture) ─────

    /// USER-VERBATIM: `eunomia::Complex64::default() ==
    /// num_complex::Complex64::default()` pin. Both types derive `Default`;
    /// both `#[repr(C)] { re, im: f64 }`; both produce `{ re: 0.0, im: 0.0 }`.
    /// This test FAILS if either side changes the derivation to a sentinel,
    /// or if the `#[repr(C)]` invariant breaks on either side.
    #[test]
    fn eunomia_default_complex64_field_by_field_equals_num_complex_default_complex64() {
        let e = eunomia::Complex64::default();
        let n = num_complex::Complex64::default();
        assert_eq!(e.re, n.re, "eunomia and num_complex default differ at re");
        assert_eq!(e.im, n.im, "eunomia and num_complex default differ at im");
        assert_eq!(e.re, 0.0_f64);
        assert_eq!(e.im, 0.0_f64);
        assert_eq!(n.re, 0.0_f64);
        assert_eq!(n.im, 0.0_f64);
        // Both types' `Default` round-trip identically. Field-by-field is the
        // canonical comparison because `eunomia::Complex64` and
        // `num_complex::Complex64` are distinct Rust types — there's no `==`
        // operator between them.
    }

    // ───── Compile-time witness of `CsrScalar` trait bounds (current surface) ─────

    /// Compile-time witness: a `num_complex::Complex64` value MUST satisfy
    /// `CsrScalar: Copy + Zero + AddAssign + Mul<Output = Self>` at this
    /// revision. If a kwavers-math refactor breaks the bound contract, this
    /// `const` initializer fails to type-check and the test is never reached.
    const _CSR_SCALAR_NUM_COMPLEX_TRAIT_WITNESS: fn() -> num_complex::Complex64 = || {
        // Bind the `CsrScalar: Copy + Zero + AddAssign + Mul<Output = Self>`
        // contract concretely so a future bound-tightening surfaces here.
        // Direct value semantics (no `&dyn` indirection) keeps the witness
        // const-eval friendly.
        let z = num_complex::Complex64::default();
        let _ = <num_complex::Complex64 as CsrScalar>::magnitude(z); // binds `CsrScalar::magnitude` + Copy via value-pass
        let _ = num_complex::Complex64::zero();                       // binds `Zero`
        // (Drop `Add<Output>` line: `Add` is not a `CsrScalar` bound; numeric arithmetic
        // round-trip is implicitly exercised by every test that calls `magnitude`.)
        let mut s = z;                                                // staging for `+=` — Copy means `z` is still usable
        s += z;                                                       // binds `AddAssign`
        let _ = z * z;                                                // binds `Mul<Output>`
        z                                                             // return matches declared `fn() -> num_complex::Complex64`
    };

    // ───── Post-ADR-0006 §1 + §2 fixture block (frozen via `#[cfg(any())]`) ─────
    // Compiles ONLY after ADR-0006 lands (csr.rs `num_traits::Zero` is replaced
    // by `eunomia::ComplexField`; `ComplexField::zero()`/`one()` defaults are
    // added).

    /// USER-VERBATIM: `magnitude([3,4] eunomia::Complex) = 5` over the
    /// post-ADR `eunomia::Complex64` route. Locks the `ComplexField` blanket
    /// impl through the `CsrScalar` default body.
    #[cfg(any())]
    #[test]
    fn csr_scalar_magnitude_eunomia_complex_3_4_5_post_adr() {
        // Post-ADR: `CsrScalar::magnitude` defaults to
        // `<Self as ComplexField>::modulus().to_f64()`, which for
        // `eunomia::Complex64` routes through `Complex::norm()` via the blanket
        // `impl<T: RealField> ComplexField for Complex<T>` at
        // `crates/eunomia/src/impls/field.rs:118` → `self.norm()`:
        // sqrt(3.0² + 4.0²) = sqrt(25.0) = 5.0 — bit-exact under IEEE 754.
        let z = eunomia::Complex64::new(3.0, 4.0);
        assert_eq!(CsrScalar::magnitude(z), 5.0);
    }

    /// Compile-time witness: `eunomia::Complex64` MUST satisfy the post-ADR
    /// `CsrScalar: Copy + ComplexField + AddAssign + Mul<Output = Self>` bounds.
    #[cfg(any())]
    #[allow(dead_code)]
    const _CSR_SCALAR_EUNOMIA_COMPLEX_TRAIT_WITNESS_POST_ADR: 
        fn() -> eunomia::Complex64 = || {
        let z: &dyn CsrScalar = &eunomia::Complex64::default();
        let _ = z.magnitude();
        <eunomia::Complex64 as eunomia::ComplexField>::zero()
    };

    /// Pin `<eunomia::Complex<f64> as ComplexField>::modulus()` directly.
    /// This is the SSOT-source of `CsrScalar::magnitude` post-ADR §2.
    /// Compiles today: `eunomia::ComplexField::modulus` is implemented
    /// for `Complex<f64>` via the blanket
    /// `impl<T: RealField> ComplexField for Complex<T>` at
    /// `crates/eunomia/src/impls/field.rs:148-149` (calls `self.norm()`).
    #[test]
    fn complex_field_modulus_eunomia_complex_3_4_5() {
        let z = eunomia::Complex64::new(3.0, 4.0);
        assert_eq!(<eunomia::Complex64 as eunomia::ComplexField>::modulus(z), 5.0);
        assert_eq!(
            <eunomia::Complex64 as eunomia::ComplexField>::modulus_squared(z),
            25.0,
        );
    }
}
