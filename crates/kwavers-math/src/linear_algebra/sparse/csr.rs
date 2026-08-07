//! Compressed Sparse Row matrix implementation for BEM/FEM solvers.
//!
//! This module provides the domain-specific `CompressedSparseRowMatrix` type
//! used throughout the boundary element and finite element solvers.

pub use eunomia::Complex64;

/// Compressed Sparse Row matrix for BEM/FEM systems.
///
/// This is a domain-specific wrapper around CSR format that adds BEM-specific
/// functionality while maintaining compatibility with leto-ops sparse kernels.
#[derive(Debug, Clone)]
pub struct CompressedSparseRowMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<T>,
    pub col_indices: Vec<usize>,
    pub row_pointers: Vec<usize>,
    pub nnz: usize,
}

impl<T: leto_ops::Scalar + Clone> CompressedSparseRowMatrix<T> {
    /// Create a new CSR matrix with the given dimensions and data.
    #[inline]
    pub fn new(
        rows: usize,
        cols: usize,
        values: Vec<T>,
        col_indices: Vec<usize>,
        row_pointers: Vec<usize>,
        nnz: usize,
    ) -> Self {
        Self {
            rows,
            cols,
            values,
            col_indices,
            row_pointers,
            nnz,
        }
    }

    /// Create a new CSR matrix with the given dimensions and pre-allocated capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Self {
        let values = Vec::with_capacity(capacity);
        let col_indices = Vec::with_capacity(capacity);
        let row_pointers = vec![0; rows + 1];
        Self {
            rows,
            cols,
            values,
            col_indices,
            row_pointers,
            nnz: 0,
        }
    }

    /// Create a new empty CSR matrix with the given dimensions.
    #[inline]
    #[must_use]
    pub fn create(rows: usize, cols: usize) -> Self {
        let values = Vec::new();
        let col_indices = Vec::new();
        let row_pointers = vec![0; rows + 1];
        Self {
            rows,
            cols,
            values,
            col_indices,
            row_pointers,
            nnz: 0,
        }
    }

    /// Get the number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the number of nonzeros.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Convert to a leto-ops CsrMatrix for use with sparse kernels.
    pub fn to_csr(&self) -> Result<leto_ops::application::sparse::CsrMatrix<T>, String> {
        leto_ops::application::sparse::CsrMatrix::from_parts(
            self.values.clone(),
            self.col_indices.clone(),
            self.row_pointers.clone(),
            self.rows,
            self.cols,
        )
        .map_err(|e| e.to_string())
    }

    /// Zero out all entries in the given row.
    pub fn zero_row(&mut self, row: usize) {
        if row >= self.rows {
            return;
        }
        let start = self.row_pointers[row];
        let end = if row + 1 < self.rows {
            self.row_pointers[row + 1]
        } else {
            self.nnz
        };
        for i in start..end {
            self.values[i] = <T as eunomia::NumericElement>::ZERO;
        }
    }

    /// Set a diagonal entry. If the diagonal is not stored, it is added.
    pub fn set_diagonal(&mut self, row: usize, value: T) {
        if row >= self.rows || row >= self.cols {
            return;
        }
        let start = self.row_pointers[row];
        let end = if row + 1 < self.rows {
            self.row_pointers[row + 1]
        } else {
            self.nnz
        };
        for i in start..end {
            if self.col_indices[i] == row {
                self.values[i] = value;
                return;
            }
        }
        // Diagonal not found, add it
        self.values.push(value);
        self.col_indices.push(row);
        // Update row pointers for all subsequent rows
        for rp in &mut self.row_pointers[row + 1..] {
            *rp += 1;
        }
        self.nnz += 1;
    }

    /// Get a diagonal entry. Returns None if the diagonal is not stored.
    #[must_use]
    pub fn get_diagonal(&self, row: usize) -> Option<T> {
        if row >= self.rows || row >= self.cols {
            return None;
        }
        let start = self.row_pointers[row];
        let end = if row + 1 < self.rows {
            self.row_pointers[row + 1]
        } else {
            self.nnz
        };
        for i in start..end {
            if self.col_indices[i] == row {
                return Some(self.values[i]);
            }
        }
        None
    }

    /// Get all non-zero values and column indices for a given row.
    /// Returns a tuple of (values, col_indices) for the specified row.
    #[must_use]
    pub fn get_row(&self, row: usize) -> (&[T], &[usize]) {
        if row >= self.rows {
            return (&[], &[]);
        }
        let start = self.row_pointers[row];
        let end = if row + 1 < self.rows {
            self.row_pointers[row + 1]
        } else {
            self.nnz
        };
        (&self.values[start..end], &self.col_indices[start..end])
    }

    /// Add a value to an existing entry. If the entry doesn't exist, it is added.
    pub fn add_value(&mut self, row: usize, col: usize, value: T) {
        if row >= self.rows || col >= self.cols {
            return;
        }
        let start = self.row_pointers[row];
        let end = if row + 1 < self.rows {
            self.row_pointers[row + 1]
        } else {
            self.nnz
        };
        for i in start..end {
            if self.col_indices[i] == col {
                self.values[i] += value;
                return;
            }
        }
        // Entry not found, add it
        self.values.push(value);
        self.col_indices.push(col);
        // Update row pointers for all subsequent rows
        for rp in &mut self.row_pointers[row + 1..] {
            *rp += 1;
        }
        self.nnz += 1;
    }

    /// Zero out all off-diagonal entries in the given row.
    pub fn zero_row_off_diagonals(&mut self, row: usize) {
        if row >= self.rows {
            return;
        }
        let start = self.row_pointers[row];
        let end = if row + 1 < self.rows {
            self.row_pointers[row + 1]
        } else {
            self.nnz
        };
        for i in start..end {
            if self.col_indices[i] != row {
                self.values[i] = <T as eunomia::NumericElement>::ZERO;
            }
        }
    }

    /// Compress the matrix by removing explicit zeros.
    /// This removes entries that are exactly zero from the sparse representation.
    #[must_use]
    pub fn compress(&self) -> Self {
        let mut new_values = Vec::new();
        let mut new_col_indices = Vec::new();
        let mut new_row_pointers = vec![0; self.rows + 1];

        for row in 0..self.rows {
            let start = self.row_pointers[row];
            let end = if row + 1 < self.rows {
                self.row_pointers[row + 1]
            } else {
                self.nnz
            };

            let mut row_nnz = 0;
            for i in start..end {
                let val = self.values[i];
                if val != <T as eunomia::NumericElement>::ZERO {
                    new_values.push(val);
                    new_col_indices.push(self.col_indices[i]);
                    row_nnz += 1;
                }
            }

            // Update row pointer
            let current_ptr = if row > 0 { new_row_pointers[row] } else { 0 };
            new_row_pointers[row + 1] = current_ptr + row_nnz;
        }

        let nnz = new_values.len();
        Self {
            rows: self.rows,
            cols: self.cols,
            values: new_values,
            col_indices: new_col_indices,
            row_pointers: new_row_pointers,
            nnz,
        }
    }
}

/// Type alias for Complex64 CSR matrices used in BEM solvers.
pub type CsrMatrixComplex = CompressedSparseRowMatrix<Complex64>;
