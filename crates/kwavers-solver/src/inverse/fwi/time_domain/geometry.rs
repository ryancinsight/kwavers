//! `FwiGeometry` — source/receiver geometry for acoustic FWI.
//!
//! Encapsulates the forward source term and receiver layout.  The
//! `receiver_row_to_sensor_row` permutation converts residual data from the
//! recorder's Fortran-order convention into the row order required by the
//! pressure-source injector.

use std::collections::HashMap;

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_domain::source::GridSource;
use ndarray::Array3;

/// Source and receiver geometry used by acoustic FWI.
#[derive(Debug, Clone)]
pub struct FwiGeometry {
    pub source: GridSource,
    pub sensor_mask: Array3<bool>,
    pub(super) receiver_row_to_sensor_row: Vec<usize>,
}

impl FwiGeometry {
    /// Build geometry from a forward source and a receiver mask.
    ///
    /// Constructs the Fortran-to-row-major permutation once so that every
    /// adjoint pass can reorder residual rows in O(n_receivers) time.
    /// # Panics
    /// - Panics if `receiver mask ordering mismatch`.
    ///
    #[must_use]
    pub fn new(source: GridSource, sensor_mask: Array3<bool>) -> Self {
        let sensor_indices = Self::collect_fortran_indices(&sensor_mask);
        let receiver_indices = Self::collect_row_major_indices(&sensor_mask);

        let sensor_lookup: HashMap<(usize, usize, usize), usize> = sensor_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(row, coord)| (coord, row))
            .collect();

        let receiver_row_to_sensor_row = receiver_indices
            .iter()
            .map(|coord| {
                *sensor_lookup
                    .get(coord)
                    .expect("receiver mask ordering mismatch")
            })
            .collect();

        Self {
            source,
            sensor_mask,
            receiver_row_to_sensor_row,
        }
    }

    #[must_use]
    pub(super) fn receiver_count(&self) -> usize {
        self.receiver_row_to_sensor_row.len()
    }

    pub(super) fn collect_fortran_indices(mask: &Array3<bool>) -> Vec<(usize, usize, usize)> {
        let (nx, ny, nz) = mask.dim();
        let mut indices = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if mask[[i, j, k]] {
                        indices.push((i, j, k));
                    }
                }
            }
        }
        indices
    }

    fn collect_row_major_indices(mask: &Array3<bool>) -> Vec<(usize, usize, usize)> {
        let mut indices = Vec::new();
        for ((i, j, k), &active) in mask.indexed_iter() {
            if active {
                indices.push((i, j, k));
            }
        }
        indices
    }
    /// Validate.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if `validated above`.
    ///
    pub(super) fn validate(&self, grid: &Grid, nt: usize) -> KwaversResult<()> {
        let expected_shape = grid.dimensions();
        if self.sensor_mask.dim() != expected_shape {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Receiver mask shape mismatch: expected {:?}, got {:?}",
                        expected_shape,
                        self.sensor_mask.dim()
                    ),
                },
            ));
        }

        let Some(source_mask) = self.source.p_mask.as_ref() else {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires a time-varying pressure source mask".to_owned(),
                },
            ));
        };
        if source_mask.dim() != expected_shape {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Source mask shape mismatch: expected {:?}, got {:?}",
                        expected_shape,
                        source_mask.dim()
                    ),
                },
            ));
        }

        if self.source.p_signal.as_ref().is_none() {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "FWI requires a time-varying pressure source signal".to_owned(),
                },
            ));
        }
        let source_signal = self.source.p_signal.as_ref().expect("validated above");
        if source_signal.shape()[1] < nt {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Source signal must contain at least {nt} samples, got {}",
                        source_signal.shape()[1]
                    ),
                },
            ));
        }

        if self.receiver_count() == 0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Receiver mask contains no active points".to_owned(),
                },
            ));
        }

        Ok(())
    }
}
