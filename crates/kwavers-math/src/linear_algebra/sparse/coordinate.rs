//! Coordinate (COO) sparse matrix format.
//!
//! Provides a simple mutable COO matrix that can be converted to CSR.

use super::csr::CompressedSparseRowMatrix;
use std::ops::Add;

/// Coordinate format sparse matrix.
///
/// Stores (row, col, value) triplets. Can be converted to CSR format
/// via .
#[derive(Debug, Clone)]
pub struct CoordinateMatrix<T> {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Stored triplets: (row, col, value).
    pub entries: Vec<(usize, usize, T)>,
}

impl<T: Copy + Default + PartialOrd + Add<Output = T>> CoordinateMatrix<T> {
    /// Create a new empty COO matrix.
    pub fn create(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: Vec::new(),
        }
    }

    /// Add a value at (row, col), accumulating if the entry already exists.
    ///
    /// Alias for  for backward compatibility.
    pub fn add_triplet(&mut self, row: usize, col: usize, val: T) {
        self.add_value(row, col, val);
    }

    /// Add a value at (row, col), accumulating if the entry already exists.
    pub fn add_value(&mut self, row: usize, col: usize, val: T) {
        for entry in self.entries.iter_mut() {
            if entry.0 == row && entry.1 == col {
                entry.2 = entry.2 + val;
                return;
            }
        }
        self.entries.push((row, col, val));
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> CompressedSparseRowMatrix<T> {
        let mut sorted: Vec<&(usize, usize, T)> = self.entries.iter().collect();
        sorted.sort_by_key(|e| (e.0, e.1));

        let mut data: Vec<Vec<(usize, T)>> = (0..self.rows).map(|_| Vec::new()).collect();
        for entry in &sorted {
            data[entry.0].push((entry.1, entry.2));
        }

        let nnz = self.entries.len();
        CompressedSparseRowMatrix {
            rows: self.rows,
            cols: self.cols,
            nnz,
            data,
        }
    }
}
