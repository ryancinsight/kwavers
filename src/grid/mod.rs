// grid/mod.rs

use crate::error::{KwaversError, KwaversResult};
use log::debug;
use ndarray::{Array1, Array3};
use std::f64::consts::PI;
use std::sync::OnceLock;

/// Dimension selector for coordinate generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    X,
    Y,
    Z,
}

/// Defines a 3D Cartesian grid for the simulation domain, optimized for k-space pseudospectral methods.
#[derive(Debug, Clone)]
pub struct Grid {
    pub nx: usize, // Number of points in x-direction (including padding)
    pub ny: usize, // Number of points in y-direction (including padding)
    pub nz: usize, // Number of points in z-direction (including padding)
    pub dx: f64,   // Spacing in x-direction (meters)
    pub dy: f64,   // Spacing in y-direction (meters)
    pub dz: f64,   // Spacing in z-direction (meters)

    // Cache for k_squared computation
    k_squared_cache: OnceLock<Array3<f64>>,
}

impl Default for Grid {
    /// Creates a default 32x32x32 grid with 1mm spacing in each dimension.
    /// This is intended for use in tests and examples.
    fn default() -> Self {
        Self::new(32, 32, 32, 1e-3, 1e-3, 1e-3)
    }
}

impl Grid {
    /// Creates a new grid with specified dimensions and spacing.
    /// 
    /// # Panics
    /// Panics if dimensions or spacing are not positive.
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self::create(nx, ny, nz, dx, dy, dz)
            .expect("Invalid grid parameters")
    }

    /// Creates a new grid with specified dimensions and spacing, returning an error if invalid.
    pub fn create(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KwaversError::InvalidInput(
                format!("Grid dimensions must be positive, got nx={}, ny={}, nz={}", nx, ny, nz)
            ));
        }
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                format!("Grid spacing must be positive, got dx={}, dy={}, dz={}", dx, dy, dz)
            ));
        }

        let grid = Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            k_squared_cache: OnceLock::new(),
        };
        debug!(
            "Grid created: {}x{}x{}, dx = {}, dy = {}, dz = {}",
            nx, ny, nz, dx, dy, dz
        );

        if (dx - dy).abs() > 1e-10 || (dy - dz).abs() > 1e-10 {
            debug!("Warning: Non-uniform grid spacing may affect k-space accuracy");
        }
        Ok(grid)
    }

    /// Create a zero-initialized 3D array with the grid dimensions
    /// This follows DRY principle by centralizing array creation
    #[inline]
    pub fn create_field(&self) -> Array3<f64> {
        Array3::zeros((self.nx, self.ny, self.nz))
    }

    /// Converts physical position (meters) to grid indices, returning None if out of bounds.
    /// Uses floor() to determine the grid cell containing the position.
    pub fn position_to_indices(&self, x: f64, y: f64, z: f64) -> Option<(usize, usize, usize)> {
        // Check bounds first
        if x < 0.0 || y < 0.0 || z < 0.0 {
            return None;
        }

        // Use floor for correct cell assignment
        let i = (x / self.dx).floor() as usize;
        let j = (y / self.dy).floor() as usize;
        let k = (z / self.dz).floor() as usize;

        // Check upper bounds
        if i >= self.nx || j >= self.ny || k >= self.nz {
            return None;
        }

        Some((i, j, k))
    }

    /// Compute k-space wavenumbers in x-direction
    pub fn compute_kx(&self) -> Array1<f64> {
        let dk = 2.0 * PI / (self.nx as f64 * self.dx);
        
        Array1::from_iter((0..self.nx).map(|i| {
            let idx = if i <= self.nx / 2 {
                i as f64
            } else {
                i as f64 - self.nx as f64
            };
            idx * dk
        }))
    }

    /// Compute k-space wavenumbers in y-direction
    pub fn compute_ky(&self) -> Array1<f64> {
        let dk = 2.0 * PI / (self.ny as f64 * self.dy);
        
        Array1::from_iter((0..self.ny).map(|j| {
            let idx = if j <= self.ny / 2 {
                j as f64
            } else {
                j as f64 - self.ny as f64
            };
            idx * dk
        }))
    }

    /// Compute k-space wavenumbers in z-direction
    pub fn compute_kz(&self) -> Array1<f64> {
        let dk = 2.0 * PI / (self.nz as f64 * self.dz);
        
        Array1::from_iter((0..self.nz).map(|k| {
            let idx = if k <= self.nz / 2 {
                k as f64
            } else {
                k as f64 - self.nz as f64
            };
            idx * dk
        }))
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

    /// Returns the total physical dimensions (meters) of the grid.
    /// This represents the total physical size of the computational domain.
    /// For a grid with nx points and spacing dx, the physical dimension is nx * dx.
    pub fn physical_dimensions(&self) -> (f64, f64, f64) {
        (
            self.dx * self.nx as f64,
            self.dy * self.ny as f64,
            self.dz * self.nz as f64,
        )
    }

    /// Returns the span between the first and last grid points in each dimension (meters).
    /// This represents the distance from the first grid point to the last grid point.
    /// For a grid with nx points and spacing dx, the span is (nx - 1) * dx.
    pub fn grid_span(&self) -> (f64, f64, f64) {
        (
            self.dx * (self.nx.saturating_sub(1)) as f64,
            self.dy * (self.ny.saturating_sub(1)) as f64,
            self.dz * (self.nz.saturating_sub(1)) as f64,
        )
    }



    /// Generates a 1D array of the physical coordinates of the grid points along the x-axis.
    ///
    /// The coordinates represent the center of each grid cell and range from `0.0` to
    /// `(self.nx - 1) * self.dx`, which corresponds to the grid span (not the full physical dimension).
    ///
    /// # Example
    /// ```
    /// use kwavers::grid::Grid;
    /// let grid = Grid::new(5, 1, 1, 0.1, 0.1, 0.1);
    /// let x_coords = grid.x_coordinates();
    /// // x_coords will be [0.0, 0.1, 0.2, 0.3, 0.4]
    /// ```
    pub fn x_coordinates(&self) -> Array1<f64> {
        Array1::linspace(0.0, self.dx * (self.nx - 1) as f64, self.nx)
    }

    /// Generates a 1D array of the physical coordinates of the grid points along the y-axis.
    ///
    /// The coordinates represent the center of each grid cell and range from `0.0` to
    /// `(self.ny - 1) * self.dy`, which corresponds to the grid span (not the full physical dimension).
    pub fn y_coordinates(&self) -> Array1<f64> {
        Array1::linspace(0.0, self.dy * (self.ny - 1) as f64, self.ny)
    }

    /// Generates a 1D array of the physical coordinates of the grid points along the z-axis.
    ///
    /// The coordinates represent the center of each grid cell and range from `0.0` to
    /// `(self.nz - 1) * self.dz`, which corresponds to the grid span (not the full physical dimension).
    pub fn z_coordinates(&self) -> Array1<f64> {
        Array1::linspace(0.0, self.dz * (self.nz - 1) as f64, self.nz)
    }

    /// Check if a point is within the grid bounds
    pub fn contains_point(&self, x: f64, y: f64, z: f64) -> bool {
        let (lx, ly, lz) = self.physical_dimensions();
        x >= 0.0 && x < lx && y >= 0.0 && y < ly && z >= 0.0 && z < lz
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

    /// Returns the squared wavenumber magnitude (k^2) for k-space operations (rad^2/m^2).
    /// This value is cached after first computation for efficiency.
    pub fn k_squared(&self) -> &Array3<f64> {
        self.k_squared_cache.get_or_init(|| {
            debug!(
                "Computing and caching k^2 for k-space: {}x{}x{}",
                self.nx, self.ny, self.nz
            );
            let kx = self.kx();
            let ky = self.ky();
            let kz = self.kz();
            Array3::from_shape_fn((self.nx, self.ny, self.nz), |(i, j, k)| {
                kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k]
            })
        })
    }

    /// Computes the k-space correction factor, sinc(c k dt / 2), given sound speed and time step.
    pub fn kspace_correction(&self, c: f64, dt: f64) -> Array3<f64> {
        let k2 = self.k_squared();
        k2.mapv(|k2_val| {
            let k = k2_val.sqrt();
            let arg = c * k * dt / 2.0;
            if arg.abs() < 1e-10 {
                1.0 // sinc(0) = 1
            } else {
                arg.sin() / arg // sinc(x) = sin(x)/x
            }
        })
    }

    /// Calculates the maximum stable time step based on the CFL condition.
    ///
    /// For 3D simulations, the CFL condition requires:
    /// dt ≤ min(dx, dy, dz) / (c * sqrt(3))
    ///
    /// Reference: Taflove & Hagness, "Computational Electrodynamics: The Finite-Difference
    /// Time-Domain Method", 3rd Edition, Section 4.8 for 3D FDTD stability.
    ///
    /// For k-space methods, the stability criteria can be different, but we use the
    /// conservative 3D FDTD criterion to ensure stability across all methods.
    ///
    /// # Arguments
    /// * `sound_speed` - The maximum sound speed in the medium (m/s)
    /// * `cfl_factor` - Safety factor for CFL condition (default: 0.3 for k-space methods)
    ///
    /// # Returns
    /// The maximum stable time step in seconds
    pub fn cfl_timestep(&self, sound_speed: f64, cfl_factor: f64) -> f64 {
        assert!(sound_speed > 0.0, "Sound speed must be positive");
        assert!(
            cfl_factor > 0.0 && cfl_factor <= 1.0,
            "CFL factor must be in (0, 1]"
        );

        let (dx, dy, dz) = self.spacing();
        let min_spacing = dx.min(dy).min(dz);

        // For 3D simulations, the CFL stability factor is sqrt(3) ≈ 1.732
        // This ensures stability for wave propagation in all diagonal directions
        let stability_factor = 3.0_f64.sqrt();

        min_spacing / (sound_speed * stability_factor) * cfl_factor
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
    i: usize,
    j: usize,
    k: usize,
}

impl<'a> Iterator for GridPointIterator<'a> {
    type Item = ((usize, usize, usize), (f64, f64, f64));

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.grid.nx {
            return None;
        }

        let coords = self.grid.coordinates(self.i, self.j, self.k);
        let indices = (self.i, self.j, self.k);

        // Increment indices - iterate in memory-friendly order (k fastest)
        self.k += 1;
        if self.k >= self.grid.nz {
            self.k = 0;
            self.j += 1;
            if self.j >= self.grid.ny {
                self.j = 0;
                self.i += 1;
            }
        }

        Some((indices, coords))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let current = self.i * self.grid.ny * self.grid.nz + self.j * self.grid.nz + self.k;
        let remaining = self.grid.total_points().saturating_sub(current);
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
            SliceType::XY(z) => (0, self.grid.nx, 0, self.grid.ny, z, z + 1),
            SliceType::XZ(y) => (0, self.grid.nx, y, y + 1, 0, self.grid.nz),
            SliceType::YZ(x) => (x, x + 1, 0, self.grid.ny, 0, self.grid.nz),
        };

        (i_start..i_end).flat_map(move |i| {
            (j_start..j_end).flat_map(move |j| {
                (k_start..k_end).map(move |k| ((i, j, k), self.grid.coordinates(i, j, k)))
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
            i: 0,
            j: 0,
            k: 0,
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
    pub fn points_in_sphere(
        &self,
        center: (f64, f64, f64),
        radius: f64,
    ) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
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
                *i == 0
                    || *i == self.nx - 1
                    || *j == 0
                    || *j == self.ny - 1
                    || *k == 0
                    || *k == self.nz - 1
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

        self.iter_points().for_each(|((i, j, k), (x, y, z))| {
            field[[i, j, k]] = f(x, y, z);
        });

        field
    }

    /// Compute statistics over a field using zero-copy iteration
    ///
    /// This method efficiently computes all statistics in a single pass
    /// without any intermediate allocations, making it suitable for large grids.
    pub fn field_statistics(&self, field: &Array3<f64>) -> FieldStatistics {
        let count = field.len() as f64;

        // Handle empty field case
        if count == 0.0 {
            return FieldStatistics {
                mean: 0.0,
                variance: 0.0,
                std_dev: 0.0,
                min: f64::NAN,
                max: f64::NAN,
            };
        }

        // Compute all statistics in a single pass with no allocations
        let (sum, sum_sq, min, max) = field.iter().fold(
            (0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY),
            |(sum, sum_sq, min, max), &val| {
                (sum + val, sum_sq + val * val, min.min(val), max.max(val))
            },
        );

        let mean = sum / count;
        // Ensure variance is non-negative to avoid sqrt of negative due to floating point errors
        let variance = (sum_sq / count - mean * mean).max(0.0);

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
    fn test_default_grid() {
        let grid = Grid::default();
        assert_eq!(grid.nx, 32);
        assert_eq!(grid.ny, 32);
        assert_eq!(grid.nz, 32);
        assert_eq!(grid.dx, 1e-3);
        assert_eq!(grid.dy, 1e-3);
        assert_eq!(grid.dz, 1e-3);
    }

    #[test]
    fn test_field_statistics_efficiency() {
        // Test that field_statistics handles edge cases correctly
        let grid = Grid::default();

        // Test empty field
        let empty_field = Array3::<f64>::zeros((0, 0, 0));
        let stats = grid.field_statistics(&empty_field);
        assert_eq!(stats.mean, 0.0);
        assert!(stats.min.is_nan());
        assert!(stats.max.is_nan());

        // Test normal field
        let mut field = grid.create_field();
        field.fill(5.0);
        field[[0, 0, 0]] = 1.0;
        field[[1, 0, 0]] = 10.0;

        let stats = grid.field_statistics(&field);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert!(stats.variance >= 0.0); // Ensure non-negative variance

        // Test that variance protection works for floating point errors
        let uniform_field = Array3::from_elem((100, 100, 100), 1.0);
        let uniform_stats = grid.field_statistics(&uniform_field);
        assert_eq!(uniform_stats.variance, 0.0);
        assert_eq!(uniform_stats.std_dev, 0.0);
    }

    #[test]
    fn test_coordinate_generation_ranges() {
        let grid = Grid::new(5, 4, 3, 0.1, 0.2, 0.3);

        // Test x coordinates
        let x_coords = grid.x_coordinates();
        assert_eq!(x_coords.len(), 5);
        assert_eq!(x_coords[0], 0.0);
        assert_eq!(x_coords[4], 0.4); // (5-1) * 0.1

        // Test y coordinates
        let y_coords = grid.y_coordinates();
        assert_eq!(y_coords.len(), 4);
        assert_eq!(y_coords[0], 0.0);
        assert_eq!(y_coords[3], 0.6); // (4-1) * 0.2

        // Test z coordinates
        let z_coords = grid.z_coordinates();
        assert_eq!(z_coords.len(), 3);
        assert_eq!(z_coords[0], 0.0);
        assert_eq!(z_coords[2], 0.6); // (3-1) * 0.3

        // Verify that coordinates are grid span, not physical dimensions
        let (phys_x, phys_y, phys_z) = grid.physical_dimensions();
        assert!(x_coords[x_coords.len() - 1] < phys_x);
        assert!(y_coords[y_coords.len() - 1] < phys_y);
        assert!(z_coords[z_coords.len() - 1] < phys_z);
    }

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
        let medium = HomogeneousMedium::from_minimal(1000.0, 2000.0, &grid);

        let dt_from_medium = grid.cfl_timestep_from_medium(&medium);
        let dt_direct = grid.cfl_timestep_default(2000.0);

        // Should be the same
        assert!((dt_from_medium - dt_direct).abs() < 1e-10);
    }

    #[test]
    fn test_grid_span_vs_physical_dimensions() {
        // Test the distinction between grid_span and physical_dimensions
        let grid = Grid::new(10, 20, 30, 0.1, 0.2, 0.3);

        // physical_dimensions should be nx*dx, ny*dy, nz*dz
        let (phys_x, phys_y, phys_z) = grid.physical_dimensions();
        assert_eq!(phys_x, 10.0 * 0.1); // 1.0
        assert_eq!(phys_y, 20.0 * 0.2); // 4.0
        assert_eq!(phys_z, 30.0 * 0.3); // 9.0

        // grid_span should be (nx-1)*dx, (ny-1)*dy, (nz-1)*dz
        let (span_x, span_y, span_z) = grid.grid_span();
        assert_eq!(span_x, 9.0 * 0.1); // 0.9
        assert_eq!(span_y, 19.0 * 0.2); // 3.8
        assert_eq!(span_z, 29.0 * 0.3); // 8.7

        // They should be different
        assert_ne!(phys_x, span_x);
        assert_ne!(phys_y, span_y);
        assert_ne!(phys_z, span_z);

        // Edge case: single point grid
        let single_grid = Grid::new(1, 1, 1, 1.0, 1.0, 1.0);
        let (single_phys_x, single_phys_y, single_phys_z) = single_grid.physical_dimensions();
        let (single_span_x, single_span_y, single_span_z) = single_grid.grid_span();

        // For a single point, physical dimension is dx but span is 0
        assert_eq!(single_phys_x, 1.0);
        assert_eq!(single_phys_y, 1.0);
        assert_eq!(single_phys_z, 1.0);
        assert_eq!(single_span_x, 0.0);
        assert_eq!(single_span_y, 0.0);
        assert_eq!(single_span_z, 0.0);
    }
}
