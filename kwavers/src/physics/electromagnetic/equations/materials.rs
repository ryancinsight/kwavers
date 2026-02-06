//! Electromagnetic material properties utilities
//!
//! This module provides utilities for working with spatially-distributed
//! electromagnetic material properties in physics simulations.

/// Material distribution utilities for electromagnetic physics
///
/// This module provides helper functions for creating and working with
/// electromagnetic material property distributions. The actual material
/// properties are defined in the domain layer.
#[derive(Debug)]
pub struct EMMaterialUtils;

impl EMMaterialUtils {
    /// Create uniform material distribution from canonical domain property
    ///
    /// # Arguments
    ///
    /// - `shape`: Grid dimensions (e.g., `[nx, ny, nz]` for 3D)
    /// - `props`: Canonical electromagnetic property data
    pub fn create_uniform_distribution(
        shape: &[usize],
        props: crate::domain::medium::properties::ElectromagneticPropertyData,
    ) -> EMMaterialDistribution {
        let grid_shape = ndarray::IxDyn(shape);
        EMMaterialDistribution {
            permittivity: ndarray::ArrayD::from_elem(grid_shape.clone(), props.permittivity),
            permeability: ndarray::ArrayD::from_elem(grid_shape.clone(), props.permeability),
            conductivity: ndarray::ArrayD::from_elem(grid_shape.clone(), props.conductivity),
            relaxation_time: props
                .relaxation_time
                .map(|tau| ndarray::ArrayD::from_elem(grid_shape, tau)),
        }
    }
}

/// Spatially-distributed electromagnetic material properties
///
/// This struct represents electromagnetic properties as N-dimensional arrays
/// for use in numerical solvers (FDTD, FEM, etc.) that require spatially-varying
/// material distributions.
#[derive(Debug, Clone)]
pub struct EMMaterialDistribution {
    /// Relative permittivity ε_r (dimensionless)
    pub permittivity: ndarray::ArrayD<f64>,
    /// Relative permeability μ_r (dimensionless)
    pub permeability: ndarray::ArrayD<f64>,
    /// Electrical conductivity σ (S/m)
    pub conductivity: ndarray::ArrayD<f64>,
    /// Dielectric relaxation time τ (s)
    pub relaxation_time: Option<ndarray::ArrayD<f64>>,
}

impl EMMaterialDistribution {
    #[inline]
    pub fn vacuum(shape: &[usize]) -> Self {
        let props = crate::domain::medium::properties::ElectromagneticPropertyData::vacuum();
        EMMaterialUtils::create_uniform_distribution(shape, props)
    }

    /// Extract canonical domain property at specific grid location
    ///
    /// # Arguments
    ///
    /// - `index`: Grid indices (e.g., `[i, j, k]` for 3D)
    ///
    /// # Returns
    ///
    /// `Ok(ElectromagneticPropertyData)` if index is valid, `Err` otherwise
    pub fn at(
        &self,
        index: &[usize],
    ) -> Result<crate::domain::medium::properties::ElectromagneticPropertyData, String> {
        // Validate index bounds
        if index.len() != self.permittivity.ndim() {
            return Err(format!(
                "Index dimension {} does not match array dimension {}",
                index.len(),
                self.permittivity.ndim()
            ));
        }

        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.permittivity.shape()[i] {
                return Err(format!(
                    "Index {} = {} out of bounds for dimension with size {}",
                    i,
                    idx,
                    self.permittivity.shape()[i]
                ));
            }
        }

        // Extract values at index
        let permittivity = self.permittivity[index];
        let permeability = self.permeability[index];
        let conductivity = self.conductivity[index];
        let relaxation_time = self.relaxation_time.as_ref().map(|arr| arr[index]);

        // Construct canonical domain property with validation
        crate::domain::medium::properties::ElectromagneticPropertyData::new(
            permittivity,
            permeability,
            conductivity,
            relaxation_time,
        )
    }

    /// Get shape of the material distribution grid
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.permittivity.shape()
    }

    /// Get number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.permittivity.ndim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_material_creation() {
        let vacuum_props = crate::domain::medium::properties::ElectromagneticPropertyData::vacuum();
        let material = EMMaterialUtils::create_uniform_distribution(&[10, 10], vacuum_props);

        assert_eq!(material.shape(), &[10, 10]);
        assert_eq!(material.ndim(), 2);

        // Check that all values are the vacuum values
        assert!(material
            .permittivity
            .iter()
            .all(|&x| (x - 1.0).abs() < 1e-10));
        assert!(material
            .permeability
            .iter()
            .all(|&x| (x - 1.0).abs() < 1e-10));
        assert!(material.conductivity.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_at_method() {
        let tissue_props = crate::domain::medium::properties::ElectromagneticPropertyData::tissue();
        let material = EMMaterialUtils::create_uniform_distribution(&[3, 3], tissue_props);
        let props = material.at(&[1, 1]).unwrap();

        // Tissue should have higher permittivity than vacuum
        assert!(props.permittivity > 1.0);
        assert_eq!(props.permeability, 1.0); // Non-magnetic
        assert!(props.conductivity >= 0.0);
    }
}
