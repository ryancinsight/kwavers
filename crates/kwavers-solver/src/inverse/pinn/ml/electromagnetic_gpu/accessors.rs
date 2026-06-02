//! Field accessors and export utilities for `GPUEMSolver`.
//!
//! SRP: changes when the energy density formula, VTK layout, or output format changes.

use super::solver::GPUEMSolver;
use kwavers_core::error::{KwaversError, KwaversResult};

impl GPUEMSolver {
    /// Return (E, H) at the given `time_index` and `position`, or `None` if out of range.
    pub fn get_field_at(
        &self,
        time_index: usize,
        position: [usize; 3],
    ) -> Option<([f64; 3], [f64; 3])> {
        let field_data = self.field_data.as_ref()?;
        let shape = field_data.electric_field.shape();
        if time_index >= shape[0] {
            return None;
        }
        let [i, j, k] = position;
        if i >= shape[1] || j >= shape[2] || k >= shape[3] {
            return None;
        }
        let e = [
            field_data.electric_field[[time_index, i, j, k, 0]],
            field_data.electric_field[[time_index, i, j, k, 1]],
            field_data.electric_field[[time_index, i, j, k, 2]],
        ];
        let h = [
            field_data.magnetic_field[[time_index, i, j, k, 0]],
            field_data.magnetic_field[[time_index, i, j, k, 1]],
            field_data.magnetic_field[[time_index, i, j, k, 2]],
        ];
        Some((e, h))
    }

    /// Integrate 0.5·(ε|E|² + μ|H|²)·dV over the domain at `time_index`.
    pub fn compute_energy(&self, time_index: usize) -> Option<f64> {
        let field_data = self.field_data.as_ref()?;
        if time_index >= field_data.electric_field.shape()[0] {
            return None;
        }
        let mut energy = 0.0_f64;
        for i in 0..self.config.grid_size[0] {
            for j in 0..self.config.grid_size[1] {
                for k in 0..self.config.grid_size[2] {
                    let e_sq = field_data.electric_field[[time_index, i, j, k, 0]].powi(2)
                        + field_data.electric_field[[time_index, i, j, k, 1]].powi(2)
                        + field_data.electric_field[[time_index, i, j, k, 2]].powi(2);
                    let h_sq = field_data.magnetic_field[[time_index, i, j, k, 0]].powi(2)
                        + field_data.magnetic_field[[time_index, i, j, k, 1]].powi(2)
                        + field_data.magnetic_field[[time_index, i, j, k, 2]].powi(2);
                    energy +=
                        0.5 * (self.config.permittivity * e_sq + self.config.permeability * h_sq);
                }
            }
        }
        Some(energy * self.config.spatial_steps.iter().product::<f64>())
    }

    /// Export field data at `time_index` to a VTK ASCII structured-points file.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn export_vtk(&self, filename: &str, time_index: usize) -> KwaversResult<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let field_data = self.field_data.as_ref().ok_or_else(|| {
            KwaversError::System(kwavers_core::error::SystemError::InvalidOperation {
                operation: "export_vtk".to_string(),
                reason: "No field data available".to_string(),
            })
        })?;
        if time_index >= field_data.electric_field.shape()[0] {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::ConstraintViolation {
                    message: format!("time_index {} out of range", time_index),
                },
            ));
        }

        let mut file = BufWriter::new(File::create(filename)?);
        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "Electromagnetic Field Data")?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET STRUCTURED_POINTS")?;
        writeln!(
            file,
            "DIMENSIONS {} {} {}",
            self.config.grid_size[0], self.config.grid_size[1], self.config.grid_size[2]
        )?;
        writeln!(file, "ORIGIN 0.0 0.0 0.0")?;
        writeln!(
            file,
            "SPACING {} {} {}",
            self.config.spatial_steps[0],
            self.config.spatial_steps[1],
            self.config.spatial_steps[2]
        )?;
        writeln!(
            file,
            "POINT_DATA {}",
            self.config.grid_size.iter().product::<usize>()
        )?;

        writeln!(file, "VECTORS E_Field float")?;
        for k in 0..self.config.grid_size[2] {
            for j in 0..self.config.grid_size[1] {
                for i in 0..self.config.grid_size[0] {
                    writeln!(
                        file,
                        "{} {} {}",
                        field_data.electric_field[[time_index, i, j, k, 0]],
                        field_data.electric_field[[time_index, i, j, k, 1]],
                        field_data.electric_field[[time_index, i, j, k, 2]]
                    )?;
                }
            }
        }

        writeln!(file, "VECTORS H_Field float")?;
        for k in 0..self.config.grid_size[2] {
            for j in 0..self.config.grid_size[1] {
                for i in 0..self.config.grid_size[0] {
                    writeln!(
                        file,
                        "{} {} {}",
                        field_data.magnetic_field[[time_index, i, j, k, 0]],
                        field_data.magnetic_field[[time_index, i, j, k, 1]],
                        field_data.magnetic_field[[time_index, i, j, k, 2]]
                    )?;
                }
            }
        }

        file.flush()?;
        Ok(())
    }
}
