//! Elastic properties trait for solid media
//!
//! This module defines traits for elastic wave propagation in solid media,
//! including Lamé parameters and wave speeds.

use crate::grid::Grid;
use crate::medium::core::{CoreMedium, ArrayAccess};
use ndarray::{Array3, Zip};

/// Trait for elastic medium properties
pub trait ElasticProperties: CoreMedium {
    /// Returns Lamé's first parameter λ (Pa)
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Returns Lamé's second parameter μ (shear modulus) (Pa)
    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Calculates shear wave speed (m/s)
    fn shear_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mu = self.lame_mu(x, y, z, grid);
        let rho = self.density(x, y, z, grid);
        if rho > 0.0 {
            (mu / rho).sqrt()
        } else {
            0.0
        }
    }

    /// Calculates compressional wave speed (m/s)
    fn compressional_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let lambda = self.lame_lambda(x, y, z, grid);
        let mu = self.lame_mu(x, y, z, grid);
        let rho = self.density(x, y, z, grid);
        if rho > 0.0 {
            ((lambda + 2.0 * mu) / rho).sqrt()
        } else {
            0.0
        }
    }
}

/// Trait for array-based elastic property access
pub trait ElasticArrayAccess: ElasticProperties + ArrayAccess {
    /// Returns a 3D array of Lamé's first parameter λ values (Pa)
    fn lame_lambda_array(&self) -> Array3<f64>;

    /// Returns a 3D array of Lamé's second parameter μ values (Pa)
    fn lame_mu_array(&self) -> Array3<f64>;

    /// Returns a 3D array of shear wave speeds (m/s)
    fn shear_sound_speed_array(&self) -> Array3<f64> {
        let mu_arr = self.lame_mu_array();
        let rho_arr = self.density_array();
        let mut s_speed_arr = Array3::zeros(rho_arr.dim());
        Zip::from(&mut s_speed_arr)
            .and(&mu_arr)
            .and(rho_arr)
            .for_each(|s_speed, &mu, &rho| {
                if rho > 0.0 {
                    *s_speed = (mu / rho).sqrt();
                } else {
                    *s_speed = 0.0;
                }
            });
        s_speed_arr
    }

    /// Returns a 3D array of shear viscosity coefficients
    fn shear_viscosity_coeff_array(&self) -> Array3<f64> {
        let shape = self.density_array().dim();
        Array3::zeros(shape)
    }

    /// Returns a 3D array of bulk viscosity coefficients
    fn bulk_viscosity_coeff_array(&self) -> Array3<f64> {
        let shape = self.density_array().dim();
        Array3::zeros(shape)
    }
}