//! Row-major sensitivity matrix operations for inverse ultrasound.

use super::linear_operator::{dot, LinearOperator};

#[derive(Clone, Debug)]
pub struct RowMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl RowMatrix {
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[must_use]
    pub fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [f32] {
        let start = row * self.cols;
        &mut self.data[start..start + self.cols]
    }

    fn matvec_impl(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(out.len(), self.rows);
        for (row, out_value) in out.iter_mut().enumerate().take(self.rows) {
            *out_value = dot(self.row(row), x);
        }
    }

    fn t_matvec_impl(&self, y: &[f32], out: &mut [f32]) {
        debug_assert_eq!(y.len(), self.rows);
        debug_assert_eq!(out.len(), self.cols);
        out.fill(0.0);
        for (row, y_value) in y.iter().copied().enumerate().take(self.rows) {
            for (dst, value) in out.iter_mut().zip(self.row(row).iter()) {
                *dst += y_value * *value;
            }
        }
    }

    fn normal_diag_impl(&self) -> Vec<f32> {
        let mut diag = vec![0.0; self.cols];
        for row in 0..self.rows {
            for (dst, value) in diag.iter_mut().zip(self.row(row).iter()) {
                *dst += *value * *value;
            }
        }
        diag
    }

    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        self.matvec_impl(x, out);
    }

    pub fn t_matvec(&self, y: &[f32], out: &mut [f32]) {
        self.t_matvec_impl(y, out);
    }

    #[must_use]
    pub fn normal_diag(&self) -> Vec<f32> {
        self.normal_diag_impl()
    }
}

impl LinearOperator for RowMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn matvec(&self, x: &[f32], out: &mut [f32]) {
        self.matvec_impl(x, out);
    }

    fn t_matvec(&self, y: &[f32], out: &mut [f32]) {
        self.t_matvec_impl(y, out);
    }

    fn row_values(&self, row: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.cols);
        out.copy_from_slice(self.row(row));
    }

    fn normal_diag(&self) -> Vec<f32> {
        self.normal_diag_impl()
    }

    fn storage_values(&self) -> usize {
        self.data.len()
    }
}
