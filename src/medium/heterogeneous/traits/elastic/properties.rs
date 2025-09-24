//! Elastic properties implementation for heterogeneous media
//!
//! **Separation of Concerns**: Pure elastic behavior per Clean Architecture
//! Following TSE 2025 "Domain-Specific Module Organization"

use crate::medium::{
    elastic::{ElasticArrayAccess, ElasticProperties},
    heterogeneous::core::HeterogeneousMedium,
};
use ndarray::Array3;

impl ElasticProperties for HeterogeneousMedium {
    /// Get shear modulus at grid point
    #[inline]
    fn shear_modulus(&self, i: usize, j: usize, k: usize) -> f64 {
        self.lame_mu[[i, j, k]]
    }

    /// Get bulk modulus at grid point  
    #[inline]
    fn bulk_modulus(&self, i: usize, j: usize, k: usize) -> f64 {
        self.lame_lambda[[i, j, k]]
    }
}

impl ElasticArrayAccess for HeterogeneousMedium {
    /// Get shear sound speed array (zero-copy access)
    fn shear_sound_speed_array(&self) -> &Array3<f64> {
        &self.shear_sound_speed
    }

    /// Get shear viscosity coefficient array (zero-copy access)  
    fn shear_viscosity_coeff_array(&self) -> &Array3<f64> {
        &self.shear_viscosity_coeff
    }

    /// Get bulk viscosity coefficient array (zero-copy access)
    fn bulk_viscosity_coeff_array(&self) -> &Array3<f64> {
        &self.bulk_viscosity_coeff
    }

    /// Get Lamé λ parameter array (zero-copy access)
    fn lame_lambda_array(&self) -> &Array3<f64> {
        &self.lame_lambda
    }

    /// Get Lamé μ parameter array (zero-copy access) 
    fn lame_mu_array(&self) -> &Array3<f64> {
        &self.lame_mu
    }
}