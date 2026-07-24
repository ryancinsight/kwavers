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
    data: Vec<Vec<(usize, T)>>,
}

impl<T: Copy + Default + PartialOrd + Add<Output = T>> CompressedSparseRowMatrix<T> {
    /// Create a new CSR matrix with pre-allocated capacity.
    pub fn with_capacity(rows: usize, cols: usize, nnz: usize) -> Self {
        Self {
            rows,
            cols,
            nnz,
            data: (0..rows).map(|_| Vec::with_capacity(nnz / rows.max(1) + 1)).collect(),
        }
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

    /// Zero out all entries in a row.
    pub fn zero_row(&mut self, row: usize) {
        self.data[row].clear();
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
}
