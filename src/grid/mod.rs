// grid/mod.rs

use log::debug;
use ndarray::{Array1, Array3};
use std::f64::consts::PI;

/// Defines a 3D Cartesian grid for the simulation domain, optimized for k-space pseudospectral methods.
#[derive(Debug, Clone)]
pub struct Grid {
    pub nx: usize, // Number of points in x-direction (including padding)
    pub ny: usize, // Number of points in y-direction (including padding)
    pub nz: usize, // Number of points in z-direction (including padding)
    pub dx: f64,   // Spacing in x-direction (meters)
    pub dy: f64,   // Spacing in y-direction (meters)
    pub dz: f64,   // Spacing in z-direction (meters)
}

impl Grid {
    /// Creates a new grid with specified dimensions and spacing.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "Grid dimensions must be positive"
        );
        assert!(
            dx > 0.0 && dy > 0.0 && dz > 0.0,
            "Grid spacing must be positive"
        );

        let grid = Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
        };
        debug!(
            "Grid created: {}x{}x{}, dx = {}, dy = {}, dz = {}",
            nx, ny, nz, dx, dy, dz
        );

        if (dx - dy).abs() > 1e-10 || (dy - dz).abs() > 1e-10 {
            debug!("Warning: Non-uniform grid spacing may affect k-space accuracy");
        }
        grid
    }

    /// Returns the total number of grid points.
    pub fn total_points(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Returns grid dimensions as (nx, ny, nz).
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Returns grid spacing as (dx, dy, dz).
    pub fn spacing(&self) -> (f64, f64, f64) {
        (self.dx, self.dy, self.dz)
    }

    /// Returns the physical coordinates at grid indices (i, j, k).
    pub fn coordinates(&self, i: usize, j: usize, k: usize) -> (f64, f64, f64) {
        let x = i as f64 * self.dx;
        let y = j as f64 * self.dy;
        let z = k as f64 * self.dz;
        (x, y, z)
    }

    /// Calculates the physical domain size in each dimension (meters), excluding padding.
    pub fn domain_size(&self) -> (f64, f64, f64) {
        let lx = self.dx * (self.nx - 1) as f64;
        let ly = self.dy * (self.ny - 1) as f64;
        let lz = self.dz * (self.nz - 1) as f64;
        (lx, ly, lz)
    }

    /// Generates 1D arrays of coordinates (meters).
    pub fn x_coordinates(&self) -> Array1<f64> {
        Array1::linspace(0.0, self.dx * (self.nx - 1) as f64, self.nx)
    }

    pub fn y_coordinates(&self) -> Array1<f64> {
        Array1::linspace(0.0, self.dy * (self.ny - 1) as f64, self.ny)
    }

    pub fn z_coordinates(&self) -> Array1<f64> {
        Array1::linspace(0.0, self.dz * (self.nz - 1) as f64, self.nz)
    }

    /// Creates a 3D array initialized with zeros for fields (e.g., pressure, light).
    pub fn create_field(&self) -> Array3<f64> {
        Array3::zeros((self.nx, self.ny, self.nz))
    }

    /// Checks if a point (x, y, z) is within grid bounds (meters).
    pub fn contains_point(&self, x: f64, y: f64, z: f64) -> bool {
        let (lx, ly, lz) = self.domain_size();
        x >= 0.0 && x <= lx && y >= 0.0 && y <= ly && z >= 0.0 && z <= lz
    }

    /// Converts coordinates to grid indices, clamping to bounds.
    pub fn x_idx(&self, x: f64) -> usize {
        (x / self.dx).floor().clamp(0.0, (self.nx - 1) as f64) as usize
    }
    pub fn y_idx(&self, y: f64) -> usize {
        (y / self.dy).floor().clamp(0.0, (self.ny - 1) as f64) as usize
    }
    pub fn z_idx(&self, z: f64) -> usize {
        (z / self.dz).floor().clamp(0.0, (self.nz - 1) as f64) as usize
    }

    /// Converts physical coordinates to grid indices, returning None if out of bounds.
    pub fn to_grid_indices(&self, x: f64, y: f64, z: f64) -> Option<(usize, usize, usize)> {
        if !self.contains_point(x, y, z) {
            return None;
        }
        Some((self.x_idx(x), self.y_idx(y), self.z_idx(z)))
    }

    /// Generates wavenumber arrays for k-space computations (rad/m).
    pub fn kx(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.nx, |i| {
            let k = if i <= self.nx / 2 {
                i as f64
            } else {
                (i as i64 - self.nx as i64) as f64
            };
            2.0 * PI * k / (self.nx as f64 * self.dx)
        })
    }

    pub fn ky(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.ny, |j| {
            let k = if j <= self.ny / 2 {
                j as f64
            } else {
                (j as i64 - self.ny as i64) as f64
            };
            2.0 * PI * k / (self.ny as f64 * self.dy)
        })
    }

    pub fn kz(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.nz, |k| {
            let k = if k <= self.nz / 2 {
                k as f64
            } else {
                (k as i64 - self.nz as i64) as f64
            };
            2.0 * PI * k / (self.nz as f64 * self.dz)
        })
    }

    /// Computes the squared wavenumber magnitude (k^2) for k-space operations (rad^2/m^2).
    pub fn k_squared(&self) -> Array3<f64> {
        debug!(
            "Computing k^2 for k-space: {}x{}x{}",
            self.nx, self.ny, self.nz
        );
        let kx = self.kx();
        let ky = self.ky();
        let kz = self.kz();
        Array3::from_shape_fn((self.nx, self.ny, self.nz), |(i, j, k)| {
            kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k]
        })
    }

    /// Computes the k-space correction factor, sinc(c k dt / 2), given sound speed and time step.
    pub fn kspace_correction(&self, c: f64, dt: f64) -> Array3<f64> {
        let k2 = self.k_squared();
        k2.mapv(|k2_val| {
            let k = k2_val.sqrt();
            (c * k * dt / 2.0).sin() / (k + 1e-10) // Avoid division by zero
        })
    }

    /// Calculates the maximum stable time step based on the CFL condition.
    /// 
    /// # Arguments
    /// * `sound_speed` - The maximum sound speed in the medium (m/s)
    /// * `cfl_factor` - Safety factor for CFL condition (default: 0.3 for k-space methods)
    /// 
    /// # Returns
    /// The maximum stable time step in seconds
    pub fn cfl_timestep(&self, sound_speed: f64, cfl_factor: f64) -> f64 {
        assert!(sound_speed > 0.0, "Sound speed must be positive");
        assert!(cfl_factor > 0.0 && cfl_factor <= 1.0, "CFL factor must be in (0, 1]");
        
        let (dx, dy, dz) = self.spacing();
        let min_spacing = dx.min(dy).min(dz);
        
        // For k-space methods, we use a relaxed CFL condition
        // The factor of 1.414 (sqrt(2)) accounts for the 3D nature of the problem
        min_spacing / (sound_speed * 1.414) * cfl_factor
    }

    /// Calculates the maximum stable time step based on the CFL condition with default CFL factor.
    /// 
    /// # Arguments
    /// * `sound_speed` - The maximum sound speed in the medium (m/s)
    /// 
    /// # Returns
    /// The maximum stable time step in seconds (using default CFL factor of 0.3)
    pub fn cfl_timestep_default(&self, sound_speed: f64) -> f64 {
        self.cfl_timestep(sound_speed, 0.3)
    }

    /// Calculates the maximum stable time step based on the CFL condition using a medium.
    /// 
    /// This method automatically determines the maximum sound speed from the medium
    /// and applies the CFL condition with the default safety factor.
    /// 
    /// # Arguments
    /// * `medium` - Reference to the medium
    /// 
    /// # Returns
    /// The maximum stable time step in seconds
    pub fn cfl_timestep_from_medium(&self, medium: &dyn crate::medium::Medium) -> f64 {
        let max_sound_speed = crate::medium::max_sound_speed(medium);
        self.cfl_timestep_default(max_sound_speed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::homogeneous::HomogeneousMedium;

    #[test]
    fn test_cfl_timestep_methods() {
        let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
        
        // Test with different sound speeds
        let water_speed = 1500.0;
        let bone_speed = 3500.0;
        let air_speed = 343.0;
        
        let dt_water = grid.cfl_timestep_default(water_speed);
        let dt_bone = grid.cfl_timestep_default(bone_speed);
        let dt_air = grid.cfl_timestep_default(air_speed);
        
        // Faster sound speed should give smaller timestep
        assert!(dt_bone < dt_water);
        assert!(dt_water < dt_air);
        
        // Test with custom CFL factor
        let dt_custom = grid.cfl_timestep(water_speed, 0.1);
        assert!(dt_custom < dt_water); // Smaller CFL factor = smaller timestep
    }

    #[test]
    fn test_cfl_timestep_from_medium() {
        let grid = Grid::new(64, 64, 64, 1e-4, 1e-4, 1e-4);
        
        // Create a medium with known sound speed
        let medium = HomogeneousMedium::new(1000.0, 2000.0, &grid, 0.1, 1.0);
        
        let dt_from_medium = grid.cfl_timestep_from_medium(&medium);
        let dt_direct = grid.cfl_timestep_default(2000.0);
        
        // Should be the same
        assert!((dt_from_medium - dt_direct).abs() < 1e-10);
    }
}
