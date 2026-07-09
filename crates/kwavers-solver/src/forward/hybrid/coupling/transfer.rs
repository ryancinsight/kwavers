//! Transfer operators for field coupling

use super::InterfaceGeometry;
use kwavers_core::error::{KwaversResult, ValidationError};
use leto::Array3;
use std::collections::HashMap;

/// Paired `(source, target)` grid-index lists for an interface transfer.
type IndexPair = (Vec<(usize, usize, usize)>, Vec<(usize, usize, usize)>);

/// Transfer operators for field coupling
#[derive(Debug, Clone)]
pub struct TransferOperators {
    /// Operators for each field type
    operators: HashMap<String, TransferOperator>,
}

/// Individual transfer operator
#[derive(Debug, Clone)]
pub struct TransferOperator {
    /// Interpolation weights
    pub weights: Vec<f64>,
    /// Source indices
    pub source_indices: Vec<(usize, usize, usize)>,
    /// Target indices
    pub target_indices: Vec<(usize, usize, usize)>,
}

impl TransferOperators {
    /// Create new transfer operators
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(geometry: &InterfaceGeometry) -> KwaversResult<Self> {
        let mut operators = HashMap::new();

        // Create operators for standard fields
        operators.insert(
            "pressure".to_owned(),
            Self::create_operator(geometry, "pressure")?,
        );
        operators.insert(
            "velocity_x".to_owned(),
            Self::create_operator(geometry, "velocity_x")?,
        );
        operators.insert(
            "velocity_y".to_owned(),
            Self::create_operator(geometry, "velocity_y")?,
        );
        operators.insert(
            "velocity_z".to_owned(),
            Self::create_operator(geometry, "velocity_z")?,
        );

        Ok(Self { operators })
    }

    /// Create a transfer operator for a specific field
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn create_operator(
        geometry: &InterfaceGeometry,
        field_type: &str,
    ) -> KwaversResult<TransferOperator> {
        // Calculate weights and indices based on geometry
        let num_points = geometry.num_points;
        let weights = vec![1.0 / num_points as f64; num_points];

        // Generate indices based on interface normal
        let (source_indices, target_indices) = Self::generate_indices(geometry, field_type)?;

        Ok(TransferOperator {
            weights,
            source_indices,
            target_indices,
        })
    }

    /// Generate source and target indices
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn generate_indices(
        geometry: &InterfaceGeometry,
        _field_type: &str,
    ) -> KwaversResult<IndexPair> {
        let mut source_indices = Vec::new();
        let mut target_indices = Vec::new();

        // Generate based on interface normal direction
        // Generate interface point indices based on geometry
        let grid_size = (geometry.num_points as f64).sqrt() as usize;

        match geometry.normal_direction {
            0 => {
                // X-normal interface
                for j in 0..grid_size {
                    for k in 0..grid_size {
                        source_indices.push((0, j, k));
                        target_indices.push((0, j, k));
                    }
                }
            }
            1 => {
                // Y-normal interface
                for i in 0..grid_size {
                    for k in 0..grid_size {
                        source_indices.push((i, 0, k));
                        target_indices.push((i, 0, k));
                    }
                }
            }
            2 => {
                // Z-normal interface
                for i in 0..grid_size {
                    for j in 0..grid_size {
                        source_indices.push((i, j, 0));
                        target_indices.push((i, j, 0));
                    }
                }
            }
            _ => {
                return Err(ValidationError::FieldValidation {
                    field: "normal_direction".to_owned(),
                    value: format!("{}", geometry.normal_direction),
                    constraint: "Must be 0, 1, or 2".to_owned(),
                }
                .into());
            }
        }

        Ok((source_indices, target_indices))
    }

    /// Apply transfer operators to fields
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply(&self, source: &Array3<f64>, target: &mut Array3<f64>) -> KwaversResult<()> {
        // Apply pressure transfer operator
        if let Some(op) = self.operators.get("pressure") {
            Self::apply_operator(op, source, target)?;
        }

        Ok(())
    }

    /// Apply a single transfer operator
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_operator(
        operator: &TransferOperator,
        source: &Array3<f64>,
        target: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        for (idx, (&(si, sj, sk), &(ti, tj, tk))) in operator
            .source_indices
            .iter()
            .zip(operator.target_indices.iter())
            .enumerate()
        {
            if idx < operator.weights.len() {
                target[[ti, tj, tk]] += source[[si, sj, sk]] * operator.weights[idx];
            }
        }

        Ok(())
    }
}
