//! Field operations and array creation
//!
//! This module provides utilities for creating and manipulating field arrays.

use crate::grid::structure::Grid;
use ndarray::{Array3, Array4};

/// Field operations for grid-based arrays
#[derive(Debug)]
pub struct FieldOperations;

impl FieldOperations {
    /// Create a zero-initialized 3D field array
    #[inline]
    pub fn create_field(grid: &Grid) -> Array3<f64> {
        Array3::zeros((grid.nx, grid.ny, grid.nz))
    }

    /// Create a zero-initialized complex 3D field array
    #[inline]
    pub fn create_complex_field(grid: &Grid) -> Array3<num_complex::Complex<f64>> {
        Array3::zeros((grid.nx, grid.ny, grid.nz))
    }

    /// Create multiple fields as a 4D array
    #[inline]
    pub fn create_field_bundle(grid: &Grid, n_fields: usize) -> Array4<f64> {
        Array4::zeros((n_fields, grid.nx, grid.ny, grid.nz))
    }

    /// Create a field initialized with a constant value
    #[inline]
    pub fn create_constant_field(grid: &Grid, value: f64) -> Array3<f64> {
        Array3::from_elem((grid.nx, grid.ny, grid.nz), value)
    }

    /// Apply periodic boundary conditions to a field
    pub fn apply_periodic_boundary(field: &mut Array3<f64>) {
        let (nx, ny, nz) = field.dim();

        // X boundaries - proper periodic wrapping
        for j in 0..ny {
            for k in 0..nz {
                let temp = field[[0, j, k]];
                field[[0, j, k]] = field[[nx - 1, j, k]];
                field[[nx - 1, j, k]] = temp;
            }
        }

        // Y boundaries - proper periodic wrapping
        for i in 0..nx {
            for k in 0..nz {
                let temp = field[[i, 0, k]];
                field[[i, 0, k]] = field[[i, ny - 1, k]];
                field[[i, ny - 1, k]] = temp;
            }
        }

        // Z boundaries - proper periodic wrapping
        for i in 0..nx {
            for j in 0..ny {
                let temp = field[[i, j, 0]];
                field[[i, j, 0]] = field[[i, j, nz - 1]];
                field[[i, j, nz - 1]] = temp;
            }
        }
    }

    /// Calculate field statistics
    #[must_use]
    pub fn field_statistics(field: &Array3<f64>) -> FieldStatistics {
        let min = field.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = field.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum = field.iter().sum::<f64>();
        let count = field.len() as f64;
        let mean = sum / count;

        let variance = field.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count;
        let std_dev = variance.sqrt();

        FieldStatistics {
            min,
            max,
            mean,
            std_dev,
            total_energy: field.iter().map(|&x| x.powi(2)).sum::<f64>(),
        }
    }
}

/// Statistics for a field array
#[derive(Debug, Clone)]
pub struct FieldStatistics {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub total_energy: f64,
}
