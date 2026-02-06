//! Electromagnetic field utilities and energy calculations
//!
//! This module provides utilities for working with electromagnetic fields
//! and computing energy-related quantities like the Poynting vector.

use crate::domain::field::PoyntingVector;

/// Electromagnetic field utilities for physics computations
///
/// This module provides helper functions for electromagnetic field
/// operations, complementing the domain-level field structures.
#[derive(Debug)]
pub struct EMFieldUtils;

impl EMFieldUtils {
    /// Compute Poynting vector from EM fields
    ///
    /// S = E × H (in direction of energy flow)
    ///
    /// # Arguments
    ///
    /// * `electric` - Electric field E [Nx, Ny, 2] or [Nx, Ny, Nz, 3]
    /// * `magnetic` - Magnetic field H [Nx, Ny, 2] or [Nx, Ny, Nz, 3]
    /// * `permittivity` - Relative permittivity ε_r
    /// * `permeability` - Relative permeability μ_r
    pub fn compute_poynting_vector(
        electric: &ndarray::ArrayD<f64>,
        magnetic: &ndarray::ArrayD<f64>,
        permittivity: f64,
        permeability: f64,
    ) -> Result<PoyntingVector, String> {
        PoyntingVector::from_fields(electric, magnetic, permittivity, permeability)
    }

    /// Validate electromagnetic field compatibility
    ///
    /// Checks that electric and magnetic field arrays have compatible shapes
    /// for electromagnetic wave propagation.
    pub fn validate_em_compatibility(
        electric: &ndarray::ArrayD<f64>,
        magnetic: &ndarray::ArrayD<f64>,
    ) -> Result<(), String> {
        let e_shape = electric.shape();
        let h_shape = magnetic.shape();

        if e_shape != h_shape {
            return Err(format!(
                "Electric field shape {:?} does not match magnetic field shape {:?}",
                e_shape, h_shape
            ));
        }

        let ndim = e_shape.len();
        if ndim < 3 {
            return Err(format!(
                "EM fields must be 3D arrays [spatial..., components], got {}D",
                ndim
            ));
        }

        let components = *e_shape.last().unwrap();
        if components != 2 && components != 3 {
            return Err(format!(
                "Field components must be 2 (2D) or 3 (3D), got {}",
                components
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_em_field_compatibility() {
        // Valid 2D fields
        let electric = ArrayD::zeros(ndarray::IxDyn(&[10, 10, 2]));
        let magnetic = ArrayD::zeros(ndarray::IxDyn(&[10, 10, 2]));

        assert!(EMFieldUtils::validate_em_compatibility(&electric, &magnetic).is_ok());

        // Invalid: mismatched shapes
        let electric_wrong = ArrayD::zeros(ndarray::IxDyn(&[10, 10, 3]));
        assert!(EMFieldUtils::validate_em_compatibility(&electric_wrong, &magnetic).is_err());
    }

    #[test]
    fn test_poynting_vector_computation() {
        // Simple 2D case: E = [1, 0], H = [0, 1] → S_z = 1*1 - 0*0 = 1
        let mut electric = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2]));
        let mut magnetic = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2]));

        electric[[0, 0, 0]] = 1.0; // Ex = 1
        magnetic[[0, 0, 1]] = 1.0; // Hy = 1

        let poynting =
            EMFieldUtils::compute_poynting_vector(&electric, &magnetic, 1.0, 1.0).unwrap();

        // For 2D, S should be [0, 0, 1] with magnitude 1
        assert_eq!(poynting.magnitude[[0, 0]], 1.0);
    }
}
