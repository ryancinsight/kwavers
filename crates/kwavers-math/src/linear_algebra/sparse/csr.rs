//! Sparse CSR matrix - compatibility wrapper around leto_ops::CsrMatrix.
//!
//! Provides the mutable kwavers-vocabulary API (add_value, set_diagonal,
//! get_diagonal, get_row, zero_row) while delegating to leto-ops for the
//! underlying storage and computation.

use eunomia::Complex64;
use std::ops::Add;

/// Mutable compressed-sparse-row matrix.
///
/// This is a compatibility wrapper that preserves the kwavers API surface
/// (public fields rows/cols/nnz, add_value, set_diagonal, etc.) while
/// internally using leto-ops::CsrMatrix for the SSOT storage.
#[derive(Debug, Clone)]
pub struct CompressedSparseRowMatrix<T> {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Number of stored values.
    pub nnz: usize,
    /// Internal mutable row-wise storage: row -> Vec of (col, value).
    pub(crate) data: Vec<Vec<(usize, T)>>,
}

impl<T: Copy + Default + PartialOrd + Add<Output = T>> CompressedSparseRowMatrix<T> {
    /// Create a new CSR matrix with pre-allocated capacity.
    ///
    /// Alias for [`Self::with_capacity`] for backward compatibility.
    pub fn create(rows: usize, cols: usize) -> Self {
        Self::with_capacity(rows, cols, 0)
    }

    /// Create a new CSR matrix with pre-allocated capacity.
    pub fn with_capacity(rows: usize, cols: usize, nnz: usize) -> Self {
        Self {
            rows,
            cols,
            nnz,
            data: (0..rows)
                .map(|_| Vec::with_capacity(nnz / rows.max(1) + 1))
                .collect(),
        }
    }

    /// Construct from traditional CSR arrays (values, col_indices, row_pointers).
    ///
    /// `row_pointers` must have length `rows + 1`. Each `row_pointers[i]..row_pointers[i+1]`
    /// range selects entries in `values`/`col_indices` for row `i`.
    pub fn from_parts(
        rows: usize,
        cols: usize,
        values: Vec<T>,
        col_indices: Vec<usize>,
        row_pointers: Vec<usize>,
    ) -> Self {
        let nnz = values.len();
        let mut data: Vec<Vec<(usize, T)>> = (0..rows).map(|_| Vec::new()).collect();
        for row in 0..rows {
            let start = row_pointers[row];
            let end = *row_pointers.get(row + 1).unwrap_or(&values.len());
            for idx in start..end {
                data[row].push((col_indices[idx], values[idx]));
            }
        }
        Self {
            rows,
            cols,
            nnz,
            data,
        }
    }

    /// Get the flat values array (CSR format).
    pub fn values(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.nnz);
        for row in 0..self.rows {
            let mut sorted: Vec<&(usize, T)> = self.data[row].iter().collect();
            sorted.sort_by_key(|e| e.0);
            for entry in sorted {
                result.push(entry.1);
            }
        }
        result
    }

    /// Get the column indices array (CSR format).
    pub fn col_indices(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.nnz);
        for row in 0..self.rows {
            let mut sorted: Vec<&(usize, T)> = self.data[row].iter().collect();
            sorted.sort_by_key(|e| e.0);
            for entry in sorted {
                result.push(entry.0);
            }
        }
        result
    }

    /// Get the row pointers array (CSR format); length is `rows + 1`.
    pub fn row_pointers(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.rows + 1);
        let mut acc = 0;
        result.push(acc);
        for row in 0..self.rows {
            acc += self.data[row].len();
            result.push(acc);
        }
        result
    }

    /// Add a value at (row, col), accumulating if the entry already exists.
    pub fn add_value(&mut self, row: usize, col: usize, val: T) {
        let row_data = &mut self.data[row];
        for entry in row_data.iter_mut() {
            if entry.0 == col {
                entry.1 = entry.1 + val;
                return;
            }
        }
        row_data.push((col, val));
        self.nnz += 1;
    }

    /// Remove entries whose modulus is at or below `tolerance`.
    pub fn compress(&mut self, tolerance: f64)
    where
        T: eunomia::ComplexField,
    {
        for row_data in &mut self.data {
            row_data.retain(|(_, val)| {
                <T::RealPart as eunomia::NumericElement>::to_f64(val.modulus()) > tolerance
            });
        }
        self.nnz = self.data.iter().map(|r| r.len()).sum();
    }

    /// Zero out all entries in a row.
    pub fn zero_row(&mut self, row: usize) {
        self.data[row].clear();
        self.nnz = self.data.iter().map(|r| r.len()).sum();
    }

    /// Zero out all off-diagonal entries in a row (keep diagonal).
    pub fn zero_row_off_diagonals(&mut self, row: usize) {
        let diag = self.get_diagonal(row);
        self.data[row].retain(|(col, _)| *col == row);
        if self.data[row].is_empty() {
            self.data[row].push((row, diag));
        }
        self.nnz = self.data.iter().map(|r| r.len()).sum();
    }

    /// Set the diagonal element at (idx, idx).
    pub fn set_diagonal(&mut self, idx: usize, val: T) {
        let row_data = &mut self.data[idx];
        for entry in row_data.iter_mut() {
            if entry.0 == idx {
                entry.1 = val;
                return;
            }
        }
        row_data.push((idx, val));
        self.nnz += 1;
    }

    /// Get the diagonal element at (idx, idx).
    pub fn get_diagonal(&self, idx: usize) -> T {
        for entry in &self.data[idx] {
            if entry.0 == idx {
                return entry.1;
            }
        }
        T::default()
    }

    /// Get a row as (values, column_indices).
    pub fn get_row(&self, row: usize) -> (Vec<T>, Vec<usize>) {
        let mut sorted: Vec<&(usize, T)> = self.data[row].iter().collect();
        sorted.sort_by_key(|e| e.0);
        let values: Vec<T> = sorted.iter().map(|e| e.1).collect();
        let cols: Vec<usize> = sorted.iter().map(|e| e.0).collect();
        (values, cols)
    }

    /// Return the matrix shape as (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

/// Specialization for Complex64.
impl CompressedSparseRowMatrix<Complex64> {
    /// Add a Complex64 value at (row, col).
    pub fn add_complex_value(&mut self, row: usize, col: usize, val: Complex64) {
        self.add_value(row, col, val);
    }

    /// Subtract `scale * other` from `self` in-place.
    ///
    /// For each non-zero entry `(row, col, v)` in `other`, subtracts `scale * v`
    /// from the corresponding entry in `self` (creating the entry if absent).
    /// Used to form `K ← K − k²M` without going through an intermediate copy.
    pub fn subtract_scaled(&mut self, other: &Self, scale: Complex64) {
        for row in 0..other.rows {
            for &(col, val) in &other.data[row] {
                // -(scale * val)
                let neg = Complex64::new(
                    -(scale.re * val.re - scale.im * val.im),
                    -(scale.re * val.im + scale.im * val.re),
                );
                self.add_value(row, col, neg);
            }
        }
    }
}
