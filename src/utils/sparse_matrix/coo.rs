//! Coordinate (COO) sparse matrix format

use super::csr::CompressedSparseRowMatrix;

/// Coordinate format sparse matrix (triplet format)
#[derive(Debug, Clone))]
pub struct CoordinateMatrix {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Row indices
    pub row_indices: Vec<usize>,
    /// Column indices
    pub col_indices: Vec<usize>,
    /// Values
    pub values: Vec<f64>,
}

impl CoordinateMatrix {
    /// Create coordinate matrix
    pub fn create(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Add triplet (row, col, value)
    pub fn add_triplet(&mut self, row: usize, col: usize, value: f64) {
        if row < self.rows && col < self.cols && value.abs() > 1e-14 {
            self.row_indices.push(row);
            self.col_indices.push(col);
            self.values.push(value);
        }
    }

    /// Convert to CSR format
    pub fn to_csr(&self) -> CompressedSparseRowMatrix {
        // Sort by row, then column
        let mut triplets: Vec<(usize, usize, f64)> = self
            .row_indices
            .iter()
            .zip(&self.col_indices)
            .zip(&self.values)
            .map(|((r, c), v)| (*r, *c, *v))
            .collect();

        triplets.sort_by_key(|&(r, c, _)| (r, c));

        // Build CSR
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_pointers = vec![0; self.rows + 1];

        let mut current_row = 0;
        for (row, col, val) in triplets {
            while current_row < row {
                current_row += 1;
                row_pointers[current_row] = values.len();
            }
            values.push(val);
            col_indices.push(col);
        }

        while current_row < self.rows {
            current_row += 1;
            row_pointers[current_row] = values.len();
        }

        let nnz = values.len();
        CompressedSparseRowMatrix {
            rows: self.rows,
            cols: self.cols,
            values,
            col_indices,
            row_pointers,
            nnz,
        }
    }

    /// Get number of non-zeros
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.row_indices.clear();
        self.col_indices.clear();
        self.values.clear();
    }
}
