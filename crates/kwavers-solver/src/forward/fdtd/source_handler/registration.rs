use super::{collect_pressure_indices_fortran, SourceHandler};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use kwavers_source::{Source, SourceField};
use ndarray::Array2;

impl SourceHandler {
    /// Add source.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn add_source(
        &mut self,
        source: std::sync::Arc<dyn Source>,
        grid: &Grid,
        nt: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        let mask = source.create_mask(grid);
        let shape = (grid.nx, grid.ny, grid.nz);
        if mask.dim() != shape {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: format!(
                        "Source mask shape mismatch: expected {:?}, got {:?}",
                        shape,
                        mask.dim()
                    ),
                },
            ));
        }

        match source.source_type() {
            SourceField::Pressure => {
                if self.source.p_mask.is_some() || self.source.p_signal.is_some() {
                    return Err(KwaversError::Validation(
                        ValidationError::ConstraintViolation {
                            message: "Multiple pressure sources are not supported in SourceHandler"
                                .to_owned(),
                        },
                    ));
                }

                let indices = collect_pressure_indices_fortran(&mask);
                if indices.is_empty() {
                    return Err(KwaversError::Validation(
                        ValidationError::ConstraintViolation {
                            message: "Source mask contains no active source points".to_owned(),
                        },
                    ));
                }

                let mut signal = Array2::zeros((1, nt));
                for step in 0..nt {
                    let t = step as f64 * dt;
                    signal[[0, step]] = source.amplitude(t);
                }

                self.source.p_mask = Some(mask);
                self.source.p_signal = Some(signal);
                self.p_indices = indices;

                Ok(())
            }
            SourceField::VelocityX | SourceField::VelocityY | SourceField::VelocityZ => {
                if self.source.u_mask.is_some() || self.source.u_signal.is_some() {
                    return Err(KwaversError::Validation(
                        ValidationError::ConstraintViolation {
                            message: "Multiple velocity sources are not supported in SourceHandler"
                                .to_owned(),
                        },
                    ));
                }

                let mut indices = Vec::new();
                for ((i, j, k), &val) in mask.indexed_iter() {
                    if val != 0.0 {
                        indices.push((i, j, k, val));
                    }
                }
                if indices.is_empty() {
                    return Err(KwaversError::Validation(
                        ValidationError::ConstraintViolation {
                            message: "Source mask contains no active source points".to_owned(),
                        },
                    ));
                }

                let mut signal = ndarray::Array3::zeros((3, 1, nt));
                let comp = match source.source_type() {
                    SourceField::VelocityX => 0,
                    SourceField::VelocityY => 1,
                    SourceField::VelocityZ => 2,
                    SourceField::Pressure => {
                        return Err(KwaversError::Validation(
                            ValidationError::ConstraintViolation {
                                message: "Pressure source cannot be used as velocity source in this context.".to_owned(),
                            },
                        ));
                    }
                };
                for step in 0..nt {
                    let t = step as f64 * dt;
                    signal[[comp, 0, step]] = source.amplitude(t);
                }

                self.source.u_mask = Some(mask);
                self.source.u_signal = Some(signal);
                self.u_indices = indices;

                Ok(())
            }
        }
    }
}
