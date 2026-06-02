//! FEM/BEM interface data transfer — extraction and relaxed application.

use num_complex::{Complex64, ComplexFloat};

use kwavers_core::error::KwaversResult;
use kwavers_domain::mesh::tetrahedral::TetrahedralMesh;

use super::BemFemCoupler;

impl BemFemCoupler {
    /// Extract FEM field values at the coupling interface nodes.
    ///
    /// # Errors
    /// Returns `Err` when an interface node index exceeds the field length.
    pub(super) fn extract_fem_interface(
        &self,
        fem_field: &[Complex64],
    ) -> KwaversResult<Vec<Complex64>> {
        let mut interface_values = Vec::new();
        for &node_idx in &self.interface.fem_interface_nodes {
            let value = fem_field.get(node_idx).ok_or_else(|| {
                kwavers_core::error::KwaversError::InvalidInput(format!(
                    "FEM interface node index {} is out of bounds (fem_field len {})",
                    node_idx,
                    fem_field.len()
                ))
            })?;
            interface_values.push(*value);
        }
        Ok(interface_values)
    }

    /// Apply FEM interface values to the BEM boundary with relaxation.
    ///
    /// For each FEM interface node, the corresponding BEM element value is
    /// updated as `α · fem + (1 − α) · current` where `α` is
    /// `config.relaxation_factor`.
    ///
    /// # Errors
    /// Always returns `Ok`; signature matches the error-propagation chain.
    pub(super) fn apply_to_bem_boundary(
        &self,
        fem_values: &[Complex64],
        bem_boundary_values: &mut [Complex64],
    ) -> KwaversResult<()> {
        for (i, &fem_value) in fem_values.iter().enumerate() {
            if i < self.interface.fem_interface_nodes.len() {
                let fem_node_idx = self.interface.fem_interface_nodes[i];
                if let Some(&bem_element_idx) =
                    self.interface.node_element_mapping.get(&fem_node_idx)
                {
                    if bem_element_idx < bem_boundary_values.len() {
                        let current_value = bem_boundary_values[bem_element_idx];
                        bem_boundary_values[bem_element_idx] = self.config.relaxation_factor
                            * fem_value
                            + (1.0 - self.config.relaxation_factor) * current_value;
                    }
                }
            }
        }
        Ok(())
    }

    /// Extract BEM solution at the coupling interface elements.
    ///
    /// # Errors
    /// Returns `Err` when a BEM interface element index exceeds the boundary
    /// value slice length.
    pub(super) fn extract_bem_interface(
        &self,
        bem_boundary_values: &[Complex64],
    ) -> KwaversResult<Vec<Complex64>> {
        let mut interface_values = Vec::new();
        for &bem_element_idx in &self.interface.bem_interface_elements {
            let value = bem_boundary_values.get(bem_element_idx).ok_or_else(|| {
                kwavers_core::error::KwaversError::InvalidInput(format!(
                    "BEM interface element index {} out of bounds (len {})",
                    bem_element_idx,
                    bem_boundary_values.len()
                ))
            })?;
            interface_values.push(*value);
        }
        Ok(interface_values)
    }

    /// Apply BEM interface values to the FEM boundary with relaxation.
    ///
    /// Returns the maximum pointwise residual `max_i |new_i − old_i|`.
    ///
    /// # Errors
    /// Always returns `Ok`; signature matches the error-propagation chain.
    pub(super) fn apply_to_fem_boundary(
        &self,
        bem_values: &[Complex64],
        fem_field: &mut [Complex64],
        _fem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<f64> {
        let mut max_residual: f64 = 0.0;
        for (i, &bem_value) in bem_values.iter().enumerate() {
            if i < self.interface.fem_interface_nodes.len() {
                let fem_node_idx = self.interface.fem_interface_nodes[i];
                if fem_node_idx < fem_field.len() {
                    let current_value = fem_field[fem_node_idx];
                    let new_value = self.config.relaxation_factor * bem_value
                        + (1.0 - self.config.relaxation_factor) * current_value;
                    let residual = (new_value - current_value).abs();
                    max_residual = max_residual.max(residual);
                    fem_field[fem_node_idx] = new_value;
                }
            }
        }
        Ok(max_residual)
    }
}
