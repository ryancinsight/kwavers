//! Compressed Sparse Row (CSR) matrix format

use crate::core::error::KwaversResult;
use ndarray::{Array1, ArrayView1, ArrayView2};

/// Compressed Sparse Row matrix format
#[derive(Debug, Clone)]
pub struct CompressedSparseRowMatrix {
    /// Number of rows
    pub rows: usize,
    /// Number of columns  
    pub cols: usize,
    /// Non-zero values
    pub values: Vec<f64>,
    /// Column indices for each value
    pub col_indices: Vec<usize>,
    /// Row pointers (start index for each row)
    pub row_pointers: Vec<usize>,
    /// Number of non-zero elements
    pub nnz: usize,
}

impl CompressedSparseRowMatrix {
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

    /// Matrix-vector multiplication: y = A * x
    pub fn multiply_vector(&self, x: ArrayView1<f64>) -> KwaversResult<Array1<f64>> {
        if x.len() != self.cols {
            return Err(crate::domain::core::error::KwaversError::Numerical(
                crate::domain::core::error::NumericalError::Instability {
                    operation: "csr_matvec".to_string(),
                    condition: x.len() as f64,
                },
            ));
        }

        let mut y = Array1::zeros(self.rows);

        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in self.row_pointers[i]..self.row_pointers[i + 1] {
                sum += self.values[j] * x[self.col_indices[j]];
            }
            y[i] = sum;
        }

        Ok(y)
    }

    /// Get row as slice
    #[must_use]
    pub fn get_row(&self, row: usize) -> (&[f64], &[usize]) {
        let start = self.row_pointers[row];
        let end = self.row_pointers[row + 1];
        (&self.values[start..end], &self.col_indices[start..end])
    }

    /// Compute Frobenius norm
    #[must_use]
    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Get sparsity ratio
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        1.0 - (self.nnz as f64) / ((self.rows * self.cols) as f64)
    }

    /// Convert to dense matrix
    #[must_use]
    pub fn to_dense(&self) -> ndarray::Array2<f64> {
        let mut dense = ndarray::Array2::zeros((self.rows, self.cols));

        for i in 0..self.rows {
            for j in self.row_pointers[i]..self.row_pointers[i + 1] {
                dense[[i, self.col_indices[j]]] = self.values[j];
            }
        }

        dense
    }
}
