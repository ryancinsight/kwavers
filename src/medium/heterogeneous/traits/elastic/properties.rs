//! Elastic properties implementation for heterogeneous media
//!
//! **Separation of Concerns**: Pure elastic behavior per Clean Architecture
//! Following TSE 2025 "Domain-Specific Module Organization"

use crate::grid::Grid;
use crate::medium::elastic::{ElasticArrayAccess, ElasticProperties};
use crate::medium::heterogeneous::{
    core::HeterogeneousMedium, interpolation::TrilinearInterpolator,
};
use ndarray::Array3;

impl ElasticProperties for HeterogeneousMedium {
    /// Lamé's first parameter λ at continuous coordinates (Pa)
    #[inline]
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
            &self.lame_lambda,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    /// Lamé's second parameter μ (shear modulus) at continuous coordinates (Pa)
    #[inline]
    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
            &self.lame_mu,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }
}

impl ElasticArrayAccess for HeterogeneousMedium {
    /// Shear sound speed array (m/s)
    fn shear_sound_speed_array(&self) -> Array3<f64> {
        self.shear_sound_speed.clone()
    }

    /// Shear viscosity coefficient array (Pa·s)
    fn shear_viscosity_coeff_array(&self) -> Array3<f64> {
        self.shear_viscosity_coeff.clone()
    }

    /// Bulk viscosity coefficient array (Pa·s)
    fn bulk_viscosity_coeff_array(&self) -> Array3<f64> {
        self.bulk_viscosity_coeff.clone()
    }

    /// Lamé λ parameter array (Pa)
    fn lame_lambda_array(&self) -> Array3<f64> {
        self.lame_lambda.clone()
    }

    /// Lamé μ parameter array (Pa)
    fn lame_mu_array(&self) -> Array3<f64> {
        self.lame_mu.clone()
    }
}
