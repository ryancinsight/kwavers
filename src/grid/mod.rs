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

    /// Create a zero-initialized 3D array with the grid dimensions
    /// This follows DRY principle by centralizing array creation
    #[inline]
    pub fn zeros_array(&self) -> Array3<f64> {
        Array3::zeros((self.nx, self.ny, self.nz))
    }

    /// Convert physical position (meters) to grid indices
    pub fn position_to_indices(&self, x: f64, y: f64, z: f64) -> (usize, usize, usize) {
        (
            (x / self.dx).round() as usize,
            (y / self.dy).round() as usize,
            (z / self.dz).round() as usize,
        )
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
            let arg = c * k * dt / 2.0;
            if arg.abs() < 1e-10 {
                1.0  // sinc(0) = 1
            } else {
                arg.sin() / arg  // sinc(x) = sin(x)/x
            }
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

/// Iterator over grid points with their physical coordinates
pub struct GridPointIterator<'a> {
    grid: &'a Grid,
    current: usize,
}

impl<'a> Iterator for GridPointIterator<'a> {
    type Item = ((usize, usize, usize), (f64, f64, f64));
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.grid.total_points() {
            return None;
        }
        
        let k = self.current % self.grid.nz;
        let j = (self.current / self.grid.nz) % self.grid.ny;
        let i = self.current / (self.grid.ny * self.grid.nz);
        
        let coords = self.grid.coordinates(i, j, k);
        self.current += 1;
        
        Some(((i, j, k), coords))
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.grid.total_points() - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for GridPointIterator<'a> {}

/// Iterator over grid slices in a specific dimension
pub struct GridSliceIterator<'a> {
    grid: &'a Grid,
    dimension: Dimension,
    current: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum Dimension {
    X,
    Y,
    Z,
}

impl<'a> Iterator for GridSliceIterator<'a> {
    type Item = GridSlice<'a>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let (max_slices, slice_type) = match self.dimension {
            Dimension::X => (self.grid.nx, SliceType::YZ(self.current)),
            Dimension::Y => (self.grid.ny, SliceType::XZ(self.current)),
            Dimension::Z => (self.grid.nz, SliceType::XY(self.current)),
        };
        
        if self.current >= max_slices {
            return None;
        }
        
        let slice = GridSlice {
            grid: self.grid,
            slice_type,
        };
        
        self.current += 1;
        Some(slice)
    }
}

#[derive(Debug)]
pub struct GridSlice<'a> {
    grid: &'a Grid,
    slice_type: SliceType,
}

#[derive(Debug)]
enum SliceType {
    XY(usize), // z-index
    XZ(usize), // y-index
    YZ(usize), // x-index
}

impl<'a> GridSlice<'a> {
    /// Iterator over points in this slice
    pub fn points(&self) -> impl Iterator<Item = ((usize, usize, usize), (f64, f64, f64))> + '_ {
        let (i_start, i_end, j_start, j_end, k_start, k_end) = match self.slice_type {
            SliceType::XY(z) => (0, self.grid.nx, 0, self.grid.ny, z, z+1),
            SliceType::XZ(y) => (0, self.grid.nx, y, y+1, 0, self.grid.nz),
            SliceType::YZ(x) => (x, x+1, 0, self.grid.ny, 0, self.grid.nz),
        };
        
        (i_start..i_end).flat_map(move |i| {
            (j_start..j_end).flat_map(move |j| {
                (k_start..k_end).map(move |k| {
                    ((i, j, k), self.grid.coordinates(i, j, k))
                })
            })
        })
    }
}

/// Advanced grid methods using iterators
impl Grid {
    /// Returns an iterator over all grid points with their indices and coordinates
    pub fn iter_points(&self) -> GridPointIterator<'_> {
        GridPointIterator {
            grid: self,
            current: 0,
        }
    }
    
    /// Returns an iterator over grid slices in the specified dimension
    pub fn iter_slices(&self, dimension: Dimension) -> GridSliceIterator<'_> {
        GridSliceIterator {
            grid: self,
            dimension,
            current: 0,
        }
    }
    
    /// Find all grid points within a sphere using iterator combinators
    pub fn points_in_sphere(&self, center: (f64, f64, f64), radius: f64) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.iter_points()
            .filter(move |(_, (x, y, z))| {
                let dx = x - center.0;
                let dy = y - center.1;
                let dz = z - center.2;
                dx * dx + dy * dy + dz * dz <= radius * radius
            })
            .map(|(indices, _)| indices)
    }
    
    /// Find all boundary points using iterator combinators
    pub fn boundary_points(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.iter_points()
            .filter(|((i, j, k), _)| {
                *i == 0 || *i == self.nx - 1 ||
                *j == 0 || *j == self.ny - 1 ||
                *k == 0 || *k == self.nz - 1
            })
            .map(|(indices, _)| indices)
    }
    
    /// Apply a function to all grid points in parallel using rayon
    pub fn par_map_points<F, T>(&self, f: F) -> Vec<T>
    where
        F: Fn((usize, usize, usize), (f64, f64, f64)) -> T + Sync + Send,
        T: Send,
    {
        use rayon::prelude::*;
        
        (0..self.total_points())
            .into_par_iter()
            .map(|idx| {
                let k = idx % self.nz;
                let j = (idx / self.nz) % self.ny;
                let i = idx / (self.ny * self.nz);
                let coords = self.coordinates(i, j, k);
                f((i, j, k), coords)
            })
            .collect()
    }
    
    /// Create a field with values computed from coordinates using iterator
    pub fn create_field_from<F>(&self, f: F) -> Array3<f64>
    where
        F: Fn(f64, f64, f64) -> f64,
    {
        let mut field = self.create_field();
        
        self.iter_points()
            .for_each(|((i, j, k), (x, y, z))| {
                field[[i, j, k]] = f(x, y, z);
            });
            
        field
    }
    
    /// Compute statistics over a field using iterator combinators
    pub fn field_statistics(&self, field: &Array3<f64>) -> FieldStatistics {
        let values: Vec<f64> = field.iter().copied().collect();
        
        let (sum, sum_sq, min, max) = values.iter()
            .fold((0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY), 
                  |(sum, sum_sq, min, max), &val| {
                (sum + val, sum_sq + val * val, min.min(val), max.max(val))
            });
        
        let count = values.len() as f64;
        let mean = sum / count;
        let variance = (sum_sq / count) - mean * mean;
        
        FieldStatistics {
            mean,
            variance,
            std_dev: variance.sqrt(),
            min,
            max,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FieldStatistics {
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
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
