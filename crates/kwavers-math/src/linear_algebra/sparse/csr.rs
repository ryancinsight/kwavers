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
    pub fn new(rows: usize, cols: usize, values: Vec<T>, col_indices: Vec<usize>, row_pointers: Vec<usize>, nnz: usize) -> Self {
        Self { rows, cols, values, col_indices, row_pointers, nnz }
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
        ).map_err(|e| e.to_string())
    }
}

/// Type alias for Complex64 CSR matrices used in BEM solvers.
pub type CsrMatrixComplex = CompressedSparseRowMatrix<Complex64>;
