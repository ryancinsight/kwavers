// src/medium/mod.rs
use crate::grid::Grid;
use ndarray::{Array3, Zip}; // Added Zip
use std::fmt::Debug;

pub mod absorption;
pub mod heterogeneous;
pub mod homogeneous;
pub mod frequency_dependent;
pub mod anisotropic;

pub use absorption::power_law_absorption;
pub use absorption::tissue_specific;
pub use absorption::fractional_derivative;
pub use homogeneous::HomogeneousMedium;
pub use frequency_dependent::{FrequencyDependentProperties, TissueFrequencyModels};
pub use anisotropic::{AnisotropicTissueProperties, StiffnessTensor, AnisotropyType};

/// Get the maximum sound speed from a medium for CFL condition calculations.
/// 
/// This function is useful for determining the most restrictive CFL condition
/// when the medium has spatially varying sound speeds.
/// 
/// # Arguments
/// * `medium` - Reference to the medium
/// 
/// # Returns
/// The maximum sound speed in the medium (m/s)
pub fn max_sound_speed(medium: &dyn Medium) -> f64 {
    let sound_speed_array = medium.sound_speed_array();
    sound_speed_array.fold(0.0, |max, &val| max.max(val))
}

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
    
    /// Get the adiabatic index (gamma) for the medium
    /// Default is 1.4 for air, but different media have different values
    fn gamma(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        1.4 // Default for air
    }
    
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

/// Custom iterators for medium property traversal
pub mod iterators {
    use super::*;
    use crate::grid::Grid;
    
    use rayon::prelude::*;
    
    /// Iterator over medium properties at each grid point
    pub struct MediumPropertyIterator<'a> {
        medium: &'a dyn Medium,
        grid: &'a Grid,
        current: usize,
    }
    
    impl<'a> MediumPropertyIterator<'a> {
        pub fn new(medium: &'a dyn Medium, grid: &'a Grid) -> Self {
            Self {
                medium,
                grid,
                current: 0,
            }
        }
    }
    
    impl<'a> Iterator for MediumPropertyIterator<'a> {
        type Item = MediumProperties;
        
        fn next(&mut self) -> Option<Self::Item> {
            if self.current >= self.grid.total_points() {
                return None;
            }
            
            let k = self.current % self.grid.nz;
            let j = (self.current / self.grid.nz) % self.grid.ny;
            let i = self.current / (self.grid.ny * self.grid.nz);
            
            let (x, y, z) = self.grid.coordinates(i, j, k);
            
            let properties = MediumProperties {
                density: self.medium.density(x, y, z, self.grid),
                sound_speed: self.medium.sound_speed(x, y, z, self.grid),
                absorption: self.medium.absorption_coefficient(x, y, z, self.grid, self.medium.reference_frequency()),
                nonlinearity: self.medium.nonlinearity_coefficient(x, y, z, self.grid),
                position: (x, y, z),
                indices: (i, j, k),
            };
            
            self.current += 1;
            Some(properties)
        }
        
        fn size_hint(&self) -> (usize, Option<usize>) {
            let remaining = self.grid.total_points() - self.current;
            (remaining, Some(remaining))
        }
    }
    
    impl<'a> ExactSizeIterator for MediumPropertyIterator<'a> {}
    
    #[derive(Debug, Clone)]
    pub struct MediumProperties {
        pub density: f64,
        pub sound_speed: f64,
        pub absorption: f64,
        pub nonlinearity: f64,
        pub position: (f64, f64, f64),
        pub indices: (usize, usize, usize),
    }
    
    /// Iterator that finds interfaces in the medium
    pub struct InterfaceIterator<'a> {
        medium: &'a dyn Medium,
        grid: &'a Grid,
        threshold: f64,
        current: usize,
    }
    
    impl<'a> InterfaceIterator<'a> {
        pub fn new(medium: &'a dyn Medium, grid: &'a Grid, threshold: f64) -> Self {
            Self {
                medium,
                grid,
                threshold,
                current: 0,
            }
        }
        
        /// Check if a point is at an interface
        fn is_interface(&self, i: usize, j: usize, k: usize) -> bool {
            if i == 0 || j == 0 || k == 0 || 
               i >= self.grid.nx - 1 || 
               j >= self.grid.ny - 1 || 
               k >= self.grid.nz - 1 {
                return false;
            }
            
            let (x, y, z) = self.grid.coordinates(i, j, k);
            let center_density = self.medium.density(x, y, z, self.grid);
            
            // Check all 6 neighbors
            let neighbors = [
                (i + 1, j, k), (i - 1, j, k),
                (i, j + 1, k), (i, j - 1, k),
                (i, j, k + 1), (i, j, k - 1),
            ];
            
            neighbors.iter().any(|&(ni, nj, nk)| {
                let (nx, ny, nz) = self.grid.coordinates(ni, nj, nk);
                let neighbor_density = self.medium.density(nx, ny, nz, self.grid);
                ((neighbor_density - center_density).abs() / center_density) > self.threshold
            })
        }
    }
    
    impl<'a> Iterator for InterfaceIterator<'a> {
        type Item = InterfacePoint;
        
        fn next(&mut self) -> Option<Self::Item> {
            while self.current < self.grid.total_points() {
                let k = self.current % self.grid.nz;
                let j = (self.current / self.grid.nz) % self.grid.ny;
                let i = self.current / (self.grid.ny * self.grid.nz);
                
                self.current += 1;
                
                if self.is_interface(i, j, k) {
                    let (x, y, z) = self.grid.coordinates(i, j, k);
                    return Some(InterfacePoint {
                        indices: (i, j, k),
                        position: (x, y, z),
                        density_jump: self.calculate_density_jump(i, j, k),
                        normal: self.calculate_normal(i, j, k),
                    });
                }
            }
            
            None
        }
    }
    
    impl<'a> InterfaceIterator<'a> {
        fn calculate_density_jump(&self, i: usize, j: usize, k: usize) -> f64 {
            let (x, y, z) = self.grid.coordinates(i, j, k);
            let center = self.medium.density(x, y, z, self.grid);
            
            let mut max_jump = 0.0;
            for di in -1..=1 {
                for dj in -1..=1 {
                    for dk in -1..=1 {
                        if di == 0 && dj == 0 && dk == 0 { continue; }
                        
                        let ni = (i as i32 + di) as usize;
                        let nj = (j as i32 + dj) as usize;
                        let nk = (k as i32 + dk) as usize;
                        
                        if ni < self.grid.nx && nj < self.grid.ny && nk < self.grid.nz {
                            let (nx, ny, nz) = self.grid.coordinates(ni, nj, nk);
                            let neighbor = self.medium.density(nx, ny, nz, self.grid);
                            max_jump = f64::max(max_jump, (neighbor - center).abs());
                        }
                    }
                }
            }
            
            max_jump
        }
        
        fn calculate_normal(&self, i: usize, j: usize, k: usize) -> (f64, f64, f64) {
            let (x, y, z) = self.grid.coordinates(i, j, k);
            
            // Calculate gradient using central differences
            let dx = if i > 0 && i < self.grid.nx - 1 {
                let (x1, _, _) = self.grid.coordinates(i + 1, j, k);
                let (x0, _, _) = self.grid.coordinates(i - 1, j, k);
                let d1 = self.medium.density(x1, y, z, self.grid);
                let d0 = self.medium.density(x0, y, z, self.grid);
                (d1 - d0) / (2.0 * self.grid.dx)
            } else { 0.0 };
            
            let dy = if j > 0 && j < self.grid.ny - 1 {
                let (_, y1, _) = self.grid.coordinates(i, j + 1, k);
                let (_, y0, _) = self.grid.coordinates(i, j - 1, k);
                let d1 = self.medium.density(x, y1, z, self.grid);
                let d0 = self.medium.density(x, y0, z, self.grid);
                (d1 - d0) / (2.0 * self.grid.dy)
            } else { 0.0 };
            
            let dz = if k > 0 && k < self.grid.nz - 1 {
                let (_, _, z1) = self.grid.coordinates(i, j, k + 1);
                let (_, _, z0) = self.grid.coordinates(i, j, k - 1);
                let d1 = self.medium.density(x, y, z1, self.grid);
                let d0 = self.medium.density(x, y, z0, self.grid);
                (d1 - d0) / (2.0 * self.grid.dz)
            } else { 0.0 };
            
            // Normalize
            let mag = (dx * dx + dy * dy + dz * dz).sqrt();
            if mag > 0.0 {
                (dx / mag, dy / mag, dz / mag)
            } else {
                (0.0, 0.0, 0.0)
            }
        }
    }
    
    #[derive(Debug, Clone)]
    pub struct InterfacePoint {
        pub indices: (usize, usize, usize),
        pub position: (f64, f64, f64),
        pub density_jump: f64,
        pub normal: (f64, f64, f64),
    }
    
    /// Parallel iterator adapter for medium properties
    pub struct ParallelMediumIterator<'a> {
        medium: &'a dyn Medium,
        grid: &'a Grid,
    }
    
    impl<'a> ParallelMediumIterator<'a> {
        pub fn new(medium: &'a dyn Medium, grid: &'a Grid) -> Self {
            Self { medium, grid }
        }
        
        /// Map properties in parallel
        pub fn par_map<F, T>(&self, f: F) -> Vec<T>
        where
            F: Fn(MediumProperties) -> T + Sync + Send,
            T: Send,
        {
            (0..self.grid.total_points())
                .into_par_iter()
                .map(|idx| {
                    let k = idx % self.grid.nz;
                    let j = (idx / self.grid.nz) % self.grid.ny;
                    let i = idx / (self.grid.ny * self.grid.nz);
                    
                    let (x, y, z) = self.grid.coordinates(i, j, k);
                    
                    let properties = MediumProperties {
                        density: self.medium.density(x, y, z, self.grid),
                        sound_speed: self.medium.sound_speed(x, y, z, self.grid),
                        absorption: self.medium.absorption_coefficient(x, y, z, self.grid, self.medium.reference_frequency()),
                        nonlinearity: self.medium.nonlinearity_coefficient(x, y, z, self.grid),
                        position: (x, y, z),
                        indices: (i, j, k),
                    };
                    
                    f(properties)
                })
                .collect()
        }
        
        /// Filter and collect in parallel
        pub fn par_filter<F>(&self, predicate: F) -> Vec<MediumProperties>
        where
            F: Fn(&MediumProperties) -> bool + Sync + Send,
        {
            (0..self.grid.total_points())
                .into_par_iter()
                .filter_map(|idx| {
                    let k = idx % self.grid.nz;
                    let j = (idx / self.grid.nz) % self.grid.ny;
                    let i = idx / (self.grid.ny * self.grid.nz);
                    
                    let (x, y, z) = self.grid.coordinates(i, j, k);
                    
                    let properties = MediumProperties {
                        density: self.medium.density(x, y, z, self.grid),
                        sound_speed: self.medium.sound_speed(x, y, z, self.grid),
                        absorption: self.medium.absorption_coefficient(x, y, z, self.grid, self.medium.reference_frequency()),
                        nonlinearity: self.medium.nonlinearity_coefficient(x, y, z, self.grid),
                        position: (x, y, z),
                        indices: (i, j, k),
                    };
                    
                    if predicate(&properties) {
                        Some(properties)
                    } else {
                        None
                    }
                })
                .collect()
        }
    }
    
    /// Extension trait to add iterator methods to Medium
    pub trait MediumIteratorExt {
        /// Create an iterator over medium properties
        fn iter_properties<'a>(&'a self, grid: &'a Grid) -> MediumPropertyIterator<'a>;
        
        /// Create an interface iterator
        fn iter_interfaces<'a>(&'a self, grid: &'a Grid, threshold: f64) -> InterfaceIterator<'a>;
        
        /// Create a parallel iterator
        fn par_iter<'a>(&'a self, grid: &'a Grid) -> ParallelMediumIterator<'a>;
    }
    
    impl MediumIteratorExt for dyn Medium {
        fn iter_properties<'a>(&'a self, grid: &'a Grid) -> MediumPropertyIterator<'a> {
            MediumPropertyIterator::new(self, grid)
        }
        
        fn iter_interfaces<'a>(&'a self, grid: &'a Grid, threshold: f64) -> InterfaceIterator<'a> {
            InterfaceIterator::new(self, grid, threshold)
        }
        
        fn par_iter<'a>(&'a self, grid: &'a Grid) -> ParallelMediumIterator<'a> {
            ParallelMediumIterator::new(self, grid)
        }
    }
}