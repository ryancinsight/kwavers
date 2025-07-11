// src/medium/mod.rs
use crate::grid::Grid;
use ndarray::{Array3, Zip}; // Added Zip
use std::fmt::Debug;

pub mod absorption;
pub mod heterogeneous;
pub mod homogeneous;

pub use absorption::power_law_absorption;
pub use absorption::tissue_specific;

pub trait Medium: Debug + Sync + Send {
    fn density(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn sound_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn is_homogeneous(&self) -> bool { false }
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn ambient_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64;
    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn absorption_coefficient_light(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn reduced_scattering_coefficient_light(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn reference_frequency(&self) -> f64; // Added for absorption calculations
    /// Get the tissue type at a specific position (if medium supports tissue types)
    fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<tissue_specific::TissueType> { None }

    fn update_temperature(&mut self, temperature: &Array3<f64>);
    fn temperature(&self) -> &Array3<f64>;
    fn bubble_radius(&self) -> &Array3<f64>;
    fn bubble_velocity(&self) -> &Array3<f64>;
    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>);
    fn density_array(&self) -> Array3<f64>;
    fn sound_speed_array(&self) -> Array3<f64>;

    // --- Elastic Properties ---

    /// Returns Lamé's first parameter (lambda) at the given spatial coordinates (Pa).
    ///
    /// Lambda is one of the Lamé parameters and is related to the material's
    /// incompressibility and Young's modulus.
    ///
    /// # Arguments
    /// * `x`, `y`, `z` - Spatial coordinates (m).
    /// * `grid` - Reference to the simulation grid.
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Returns Lamé's second parameter (mu), also known as the shear modulus,
    /// at the given spatial coordinates (Pa).
    ///
    /// Mu measures the material's resistance to shear deformation. For ideal fluids, mu is 0.
    ///
    /// # Arguments
    /// * `x`, `y`, `z` - Spatial coordinates (m).
    /// * `grid` - Reference to the simulation grid.
    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Calculates and returns the shear wave speed (m/s) at the given spatial coordinates.
    ///
    /// Shear waves (S-waves) are waves that involve oscillation perpendicular
    /// to the direction of propagation. They can only propagate in materials with a non-zero
    /// shear modulus (mu > 0), i.e., solids.
    /// The speed is calculated as `sqrt(mu / rho)`.
    ///
    /// A default implementation is provided.
    ///
    /// # Arguments
    /// * `x`, `y`, `z` - Spatial coordinates (m).
    /// * `grid` - Reference to the simulation grid.
    fn shear_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mu = self.lame_mu(x, y, z, grid);
        let rho = self.density(x, y, z, grid);
        if rho > 0.0 {
            (mu / rho).sqrt()
        } else {
            0.0
        }
    }

    /// Calculates and returns the compressional wave speed (m/s) at the given spatial coordinates.
    ///
    /// Compressional waves (P-waves) involve oscillation parallel to the direction
    /// of propagation. These are the primary sound waves in fluids and can also propagate in solids.
    /// The speed is calculated as `sqrt((lambda + 2*mu) / rho)`.
    /// For fluids where mu = 0, this simplifies to `sqrt(lambda / rho)`, where lambda is the bulk modulus K.
    /// This method can be considered the more general form of `sound_speed()` when dealing with elastic media.
    ///
    /// A default implementation is provided.
    ///
    /// # Arguments
    /// * `x`, `y`, `z` - Spatial coordinates (m).
    /// * `grid` - Reference to the simulation grid.
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

    // --- Array-based Elastic Properties (primarily for solver optimization) ---

    /// Returns a 3D array of Lamé's first parameter (lambda) values (Pa) over the entire grid.
    ///
    /// This is typically used by solvers for efficient computation, avoiding repeated point-wise queries.
    /// Implementations are expected to cache this array if the medium's properties are static.
    fn lame_lambda_array(&self) -> Array3<f64>;

    /// Returns a 3D array of Lamé's second parameter (mu, shear modulus) values (Pa) over the entire grid.
    ///
    /// This is typically used by solvers for efficient computation.
    /// Implementations are expected to cache this array if the medium's properties are static.
    fn lame_mu_array(&self) -> Array3<f64>;

    // Existing viscoelastic properties - can be related to or used alongside Lamé parameters
    // For instance, shear_sound_speed_array could be derived from lame_mu_array and density_array.
    // Keeping them distinct for now to allow different levels of model complexity.

    /// Returns a 3D array of shear wave speeds (m/s) over the entire grid.
    ///
    /// The default implementation calculates this from `lame_mu_array()` and `density_array()`.
    /// This is typically used by solvers for efficient computation.
    fn shear_sound_speed_array(&self) -> Array3<f64> {
        let mu_arr = self.lame_mu_array();
        let rho_arr = self.density_array();
        let mut s_speed_arr = Array3::zeros(rho_arr.dim());
        Zip::from(&mut s_speed_arr)
            .and(&mu_arr)
            .and(&rho_arr)
            .for_each(|s_speed, &mu, &rho| {
                if rho > 0.0 {
                    *s_speed = (mu / rho).sqrt();
                } else {
                    *s_speed = 0.0;
                }
            });
        s_speed_arr
    }

    /// Returns a 3D array of shear viscosity coefficients over the grid.
    /// This represents damping of shear waves.
    fn shear_viscosity_coeff_array(&self) -> Array3<f64> {
        // Default implementation: assumes no shear viscosity if not specified.
        let shape = self.density_array().dim();
        Array3::zeros(shape)
    }

    /// Returns a 3D array of bulk viscosity coefficients over the grid.
    /// This represents damping of compressional waves beyond typical absorption.
    fn bulk_viscosity_coeff_array(&self) -> Array3<f64> {
        // Default implementation: assumes no bulk viscosity if not specified.
        let shape = self.density_array().dim();
        Array3::zeros(shape)
    }
}