//! Composable trait-based medium interface following Interface Segregation Principle
//!
//! This module provides fine-grained traits for different physical properties,
//! allowing media to implement only the behaviors they actually support.

// Allow unused variables in trait default implementations
// This is a known design issue - traits are too broad but refactoring would break API
#![allow(unused_variables)]

use crate::grid::Grid;
use ndarray::Array3;
use std::fmt::Debug;

/// Core trait for acoustic properties
pub trait AcousticMedium: Debug + Sync + Send {
    /// Get density at a specific point (kg/m³)
    fn density(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get sound speed at a specific point (m/s)
    fn sound_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get absorption coefficient at a specific point and frequency (Np/m)
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64;

    /// Get nonlinearity parameter (B/A) at a specific point
    fn nonlinearity_parameter(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        5.0 // Default for water-like media
    }

    /// Check if the medium is homogeneous
    fn is_homogeneous(&self) -> bool {
        false
    }

    /// Get reference to density array for efficient bulk access
    /// Returns None if not cached or not available
    fn density_array(&self) -> Option<&Array3<f64>> {
        None
    }

    /// Get reference to sound speed array for efficient bulk access
    /// Returns None if not cached or not available
    fn sound_speed_array(&self) -> Option<&Array3<f64>> {
        None
    }
}

/// Trait for elastic properties
pub trait ElasticMedium: Debug + Sync + Send {
    /// Get Lamé's first parameter λ (Pa)
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get Lamé's second parameter μ (shear modulus) (Pa)
    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get shear wave speed (m/s)
    fn shear_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Use a default density of 1000 kg/m³ for shear wave speed calculation
        // This should be overridden in concrete implementations that have access to actual density
        let density = 1000.0;
        (self.lame_mu(x, y, z, grid) / density).sqrt()
    }

    /// Get compressional wave speed (m/s)
    fn compressional_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let density = 1000.0; // Default
        let lambda = self.lame_lambda(x, y, z, grid);
        let mu = self.lame_mu(x, y, z, grid);
        ((lambda + 2.0 * mu) / density).sqrt()
    }
}

/// Trait for thermal properties
pub trait ThermalMedium: Debug + Sync + Send {
    /// Get specific heat capacity (J/(kg·K))
    fn specific_heat_capacity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get thermal conductivity (W/(m·K))
    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get thermal diffusivity (m²/s)
    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let k = self.thermal_conductivity(x, y, z, grid);
        let cp = self.specific_heat_capacity(x, y, z, grid);
        let rho = 1000.0; // Default density
        k / (rho * cp)
    }

    /// Get thermal expansion coefficient (1/K)
    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2.0e-4 // Default for water
    }

    /// Get temperature at a point (K)
    fn temperature(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        310.0 // Body temperature default
    }
}

/// Trait for optical properties
pub trait OpticalMedium: Debug + Sync + Send {
    /// Get optical absorption coefficient (1/m)
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get optical scattering coefficient (1/m)
    fn optical_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Get refractive index
    fn refractive_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.33 // Default for water
    }

    /// Get anisotropy factor for scattering
    fn anisotropy_factor(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.9 // Default for tissue
    }
}

/// Trait for viscous properties
pub trait ViscousMedium: Debug + Sync + Send {
    /// Get shear viscosity (Pa·s)
    fn shear_viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.0e-3 // Default for water
    }

    /// Get bulk viscosity (Pa·s)
    fn bulk_viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        2.5 * self.shear_viscosity(x, y, z, grid) // Stokes' hypothesis
    }
}

/// Trait for bubble dynamics properties
pub trait BubbleMedium: Debug + Sync + Send {
    /// Get surface tension (N/m)
    fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.072 // Default for water-air interface
    }

    /// Get ambient pressure (Pa)
    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        101325.0 // Atmospheric pressure
    }

    /// Get vapor pressure (Pa)
    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2338.0 // Water at 20°C
    }

    /// Get polytropic index
    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.4 // Default for air
    }
}

/// Helper function to get maximum sound speed from an acoustic medium
pub fn max_sound_speed<M: AcousticMedium + ?Sized>(medium: &M, grid: &Grid) -> f64 {
    // If array is available, use it for efficiency
    if let Some(array) = medium.sound_speed_array() {
        array.fold(0.0, |max, &val| max.max(val))
    } else {
        // Fall back to point-wise sampling
        let mut max_speed = 0.0;
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let (x, y, z) = grid.coordinates(i, j, k);
                    let speed = medium.sound_speed(x, y, z, grid);
                    max_speed = f64::max(max_speed, speed);
                }
            }
        }
        max_speed
    }
}

/// Efficiently find interfaces in a medium based on density jumps
pub fn find_interfaces<M: AcousticMedium + ?Sized>(
    medium: &M,
    grid: &Grid,
    threshold: f64,
) -> Vec<InterfacePoint> {
    let interfaces: Vec<InterfacePoint> = Vec::new();

    // Try to use cached array for efficiency
    if let Some(density_array) = medium.density_array() {
        find_interfaces_from_array(density_array, grid, threshold)
    } else {
        // Fall back to point-wise detection (less efficient)
        find_interfaces_pointwise(medium, grid, threshold)
    }
}

/// Interface point information
#[derive(Debug, Clone)]
pub struct InterfacePoint {
    pub position: (f64, f64, f64),
    pub indices: (usize, usize, usize),
    pub density_jump: f64,
    pub normal: [f64; 3],
}

/// Efficient array-based interface detection
fn find_interfaces_from_array(
    density_array: &Array3<f64>,
    grid: &Grid,
    threshold: f64,
) -> Vec<InterfacePoint> {
    let mut interfaces = Vec::new();

    // Check interior points only (avoid boundaries)
    for i in 1..(grid.nx - 1) {
        for j in 1..(grid.ny - 1) {
            for k in 1..(grid.nz - 1) {
                let center_density = density_array[[i, j, k]];

                // Check neighbors for density jump
                let neighbors = [
                    (i + 1, j, k),
                    (i - 1, j, k),
                    (i, j + 1, k),
                    (i, j - 1, k),
                    (i, j, k + 1),
                    (i, j, k - 1),
                ];

                for &(ni, nj, nk) in &neighbors {
                    let neighbor_density = density_array[[ni, nj, nk]];
                    let relative_jump = (neighbor_density - center_density).abs() / center_density;

                    if relative_jump > threshold {
                        // Calculate normal using central differences
                        let dx = (density_array[[i + 1, j, k]] - density_array[[i - 1, j, k]])
                            / (2.0 * grid.dx);
                        let dy = (density_array[[i, j + 1, k]] - density_array[[i, j - 1, k]])
                            / (2.0 * grid.dy);
                        let dz = (density_array[[i, j, k + 1]] - density_array[[i, j, k - 1]])
                            / (2.0 * grid.dz);

                        let norm = (dx * dx + dy * dy + dz * dz).sqrt();
                        let normal = if norm > 1e-10 {
                            [dx / norm, dy / norm, dz / norm]
                        } else {
                            [0.0, 0.0, 0.0]
                        };

                        interfaces.push(InterfacePoint {
                            position: grid.coordinates(i, j, k),
                            indices: (i, j, k),
                            density_jump: neighbor_density - center_density,
                            normal,
                        });

                        break; // Only record once per grid point
                    }
                }
            }
        }
    }

    interfaces
}

/// Fallback point-wise interface detection (less efficient)
fn find_interfaces_pointwise<M: AcousticMedium + ?Sized>(
    medium: &M,
    grid: &Grid,
    threshold: f64,
) -> Vec<InterfacePoint> {
    let mut interfaces = Vec::new();

    for i in 1..(grid.nx - 1) {
        for j in 1..(grid.ny - 1) {
            for k in 1..(grid.nz - 1) {
                let (x, y, z) = grid.coordinates(i, j, k);
                let center_density = medium.density(x, y, z, grid);

                // Check one neighbor for efficiency
                let (nx, ny, nz) = grid.coordinates(i + 1, j, k);
                let neighbor_density = medium.density(nx, ny, nz, grid);

                if ((neighbor_density - center_density).abs() / center_density) > threshold {
                    // Simplified normal calculation
                    let normal = [1.0, 0.0, 0.0]; // Assume x-direction for simplicity

                    interfaces.push(InterfacePoint {
                        position: (x, y, z),
                        indices: (i, j, k),
                        density_jump: neighbor_density - center_density,
                        normal,
                    });
                }
            }
        }
    }

    interfaces
}
