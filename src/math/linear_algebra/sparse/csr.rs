//! Compressed Sparse Row (CSR) matrix format

use crate::core::error::KwaversResult;
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::Zero;
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

pub trait CsrScalar: Copy + Zero + AddAssign + Mul<Output = Self> {
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
    pub fn multiply_vector(&self, x: ArrayView1<T>) -> KwaversResult<Array1<T>>
    where
        T: CsrScalar,
    {
        if x.len() != self.cols {
            return Err(crate::core::error::KwaversError::Numerical(
                crate::core::error::NumericalError::Instability {
                    operation: "csr_matvec".to_string(),
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
    pub fn to_dense(&self) -> ndarray::Array2<T>
    where
        T: Copy + Zero,
    {
        let mut dense = ndarray::Array2::from_elem((self.rows, self.cols), T::zero());

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
        T: Copy + Zero,
    {
        let row_start = self.row_pointers[row];
        let row_end = self.row_pointers[row + 1];

        for i in row_start..row_end {
            if self.col_indices[i] == row {
                return self.values[i];
            }
        }

        T::zero()
    }

    /// Zero out row (except diagonal)
    pub fn zero_row_off_diagonals(&mut self, row: usize) {
        let row_start = self.row_pointers[row];
        let row_end = self.row_pointers[row + 1];

        let mut i = row_start;
        while i < row_end {
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
