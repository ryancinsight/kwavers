//! Properly encapsulated Grid structure with compile-time unit safety
//!
//! This module provides a robust grid implementation with:
//! - Private fields with getter methods (encapsulation)
//! - Compile-time unit safety using uom
//! - No solver-specific caching (SRP)
//! - Fallible constructors (no panics)
//! - Robust floating-point comparisons

use crate::error::{KwaversError, KwaversResult};
use approx::AbsDiffEq;
use log::debug;
use uom::si::f64::Length;
use uom::si::length::meter;

/// Spatial bounds for a region with unit safety
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    /// Minimum coordinates [x, y, z]
    min: [Length; 3],
    /// Maximum coordinates [x, y, z]
    max: [Length; 3],
}

impl Bounds {
    /// Create new bounds with unit-safe lengths
    pub fn new(min: [Length; 3], max: [Length; 3]) -> Self {
        Self { min, max }
    }
    
    /// Get minimum bounds
    pub fn min(&self) -> [Length; 3] {
        self.min
    }
    
    /// Get maximum bounds
    pub fn max(&self) -> [Length; 3] {
        self.max
    }
    
    /// Get the center point
    pub fn center(&self) -> [Length; 3] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
            (self.min[2] + self.max[2]) / 2.0,
        ]
    }
}

/// Dimension selector for coordinate generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    X,
    Y,
    Z,
}

/// Defines a 3D Cartesian grid for the simulation domain
/// 
/// All fields are private to ensure data integrity.
/// Grid instances are immutable once created.
#[derive(Debug, Clone)]
pub struct Grid {
    /// Number of points in x-direction
    nx: usize,
    /// Number of points in y-direction
    ny: usize,
    /// Number of points in z-direction
    nz: usize,
    /// Spacing in x-direction with unit safety
    dx: Length,
    /// Spacing in y-direction with unit safety
    dy: Length,
    /// Spacing in z-direction with unit safety
    dz: Length,
    // Note: k_squared_cache removed - solvers should manage their own caches
}

impl Grid {
    /// Creates a new grid with unit-safe dimensions
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` - Number of grid points in each direction
    /// * `dx`, `dy`, `dz` - Grid spacing with compile-time unit checking
    ///
    /// # Returns
    /// * `Ok(Grid)` if all parameters are valid
    /// * `Err(KwaversError)` if any dimension is zero or spacing is non-positive
    ///
    /// # Example
    /// ```
    /// use uom::si::f64::Length;
    /// use uom::si::length::millimeter;
    /// 
    /// let grid = Grid::new(
    ///     100, 100, 100,
    ///     Length::new::<millimeter>(1.0),
    ///     Length::new::<millimeter>(1.0),
    ///     Length::new::<millimeter>(1.0),
    /// )?;
    /// ```
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: Length,
        dy: Length,
        dz: Length,
    ) -> KwaversResult<Self> {
        // Validate dimensions
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KwaversError::InvalidInput(format!(
                "Grid dimensions must be positive, got nx={}, ny={}, nz={}",
                nx, ny, nz
            )));
        }
        
        // Validate spacing (using Length's comparison operators)
        if dx <= Length::new::<meter>(0.0) || 
           dy <= Length::new::<meter>(0.0) || 
           dz <= Length::new::<meter>(0.0) {
            return Err(KwaversError::InvalidInput(
                "Grid spacing must be positive".to_string()
            ));
        }
        
        debug!(
            "Creating grid: {}x{}x{} points, spacing: {:.3e}m x {:.3e}m x {:.3e}m",
            nx, ny, nz,
            dx.get::<meter>(),
            dy.get::<meter>(),
            dz.get::<meter>()
        );
        
        Ok(Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
        })
    }
    
    /// Creates a grid from raw f64 values (meters)
    /// 
    /// This is a convenience method for backward compatibility.
    /// Prefer using the unit-safe `new` method when possible.
    pub fn from_meters(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<Self> {
        Self::new(
            nx, ny, nz,
            Length::new::<meter>(dx),
            Length::new::<meter>(dy),
            Length::new::<meter>(dz),
        )
    }
    
    // Getter methods for encapsulated fields
    
    /// Get number of grid points in x-direction
    #[inline]
    pub fn nx(&self) -> usize {
        self.nx
    }
    
    /// Get number of grid points in y-direction
    #[inline]
    pub fn ny(&self) -> usize {
        self.ny
    }
    
    /// Get number of grid points in z-direction
    #[inline]
    pub fn nz(&self) -> usize {
        self.nz
    }
    
    /// Get grid spacing in x-direction (unit-safe)
    #[inline]
    pub fn dx(&self) -> Length {
        self.dx
    }
    
    /// Get grid spacing in y-direction (unit-safe)
    #[inline]
    pub fn dy(&self) -> Length {
        self.dy
    }
    
    /// Get grid spacing in z-direction (unit-safe)
    #[inline]
    pub fn dz(&self) -> Length {
        self.dz
    }
    
    /// Get grid spacing in x-direction as meters (for compatibility)
    #[inline]
    pub fn dx_meters(&self) -> f64 {
        self.dx.get::<meter>()
    }
    
    /// Get grid spacing in y-direction as meters (for compatibility)
    #[inline]
    pub fn dy_meters(&self) -> f64 {
        self.dy.get::<meter>()
    }
    
    /// Get grid spacing in z-direction as meters (for compatibility)
    #[inline]
    pub fn dz_meters(&self) -> f64 {
        self.dz.get::<meter>()
    }
    
    /// Get total number of grid points
    #[inline]
    pub fn total_points(&self) -> usize {
        self.nx * self.ny * self.nz
    }
    
    /// Get grid dimensions as tuple
    #[inline]
    pub fn dim(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
    
    /// Get grid spacing as tuple of Lengths
    #[inline]
    pub fn spacing(&self) -> (Length, Length, Length) {
        (self.dx, self.dy, self.dz)
    }
    
    /// Get grid spacing as tuple of meters (for compatibility)
    #[inline]
    pub fn spacing_meters(&self) -> (f64, f64, f64) {
        (
            self.dx.get::<meter>(),
            self.dy.get::<meter>(),
            self.dz.get::<meter>(),
        )
    }
    
    /// Check if grid has uniform spacing
    #[inline]
    pub fn is_uniform(&self) -> bool {
        // Use approx crate for robust floating-point comparison
        let dx_m = self.dx.get::<meter>();
        let dy_m = self.dy.get::<meter>();
        let dz_m = self.dz.get::<meter>();
        
        // Use relative tolerance for better accuracy across scales
        dx_m.abs_diff_eq(&dy_m, dx_m * 1e-12) && 
        dy_m.abs_diff_eq(&dz_m, dy_m * 1e-12)
    }
    
    /// Get minimum grid spacing
    #[inline]
    pub fn min_spacing(&self) -> Length {
        self.dx.min(self.dy).min(self.dz)
    }
    
    /// Get maximum grid spacing
    #[inline]
    pub fn max_spacing(&self) -> Length {
        self.dx.max(self.dy).max(self.dz)
    }
    
    /// Get physical dimensions of the grid
    #[inline]
    pub fn physical_size(&self) -> (Length, Length, Length) {
        (
            self.dx * self.nx as f64,
            self.dy * self.ny as f64,
            self.dz * self.nz as f64,
        )
    }
    
    /// Get bounds of the grid
    pub fn bounds(&self) -> Bounds {
        let (lx, ly, lz) = self.physical_size();
        Bounds::new(
            [Length::new::<meter>(0.0); 3],
            [lx, ly, lz],
        )
    }
}

impl Default for Grid {
    /// Creates a default 32x32x32 grid with 1mm spacing
    /// 
    /// This uses `unwrap` safely because the parameters are known to be valid.
    fn default() -> Self {
        use uom::si::length::millimeter;
        
        Self::new(
            32, 32, 32,
            Length::new::<millimeter>(1.0),
            Length::new::<millimeter>(1.0),
            Length::new::<millimeter>(1.0),
        ).unwrap() // Safe: parameters are known valid
    }
}

// Backward compatibility layer
impl Grid {
    /// Legacy constructor that panics on invalid input
    /// 
    /// # Deprecated
    /// Use `Grid::new` or `Grid::from_meters` instead for proper error handling.
    #[deprecated(note = "Use Grid::new or Grid::from_meters for proper error handling")]
    pub fn create_legacy(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Self {
        Self::from_meters(nx, ny, nz, dx, dy, dz)
            .expect("Invalid grid parameters")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uom::si::length::{millimeter, micrometer};
    
    #[test]
    fn test_grid_creation_with_units() {
        let grid = Grid::new(
            100, 100, 100,
            Length::new::<millimeter>(1.0),
            Length::new::<millimeter>(1.0),
            Length::new::<millimeter>(1.0),
        ).unwrap();
        
        assert_eq!(grid.nx(), 100);
        assert_eq!(grid.dx_meters(), 0.001);
    }
    
    #[test]
    fn test_grid_validation() {
        // Zero dimensions should fail
        let result = Grid::new(
            0, 100, 100,
            Length::new::<meter>(1.0),
            Length::new::<meter>(1.0),
            Length::new::<meter>(1.0),
        );
        assert!(result.is_err());
        
        // Negative spacing should fail
        let result = Grid::new(
            100, 100, 100,
            Length::new::<meter>(-1.0),
            Length::new::<meter>(1.0),
            Length::new::<meter>(1.0),
        );
        assert!(result.is_err());
    }
    
    #[test]
    fn test_uniformity_check() {
        // Uniform grid
        let uniform = Grid::new(
            10, 10, 10,
            Length::new::<micrometer>(5.0),
            Length::new::<micrometer>(5.0),
            Length::new::<micrometer>(5.0),
        ).unwrap();
        assert!(uniform.is_uniform());
        
        // Non-uniform grid
        let non_uniform = Grid::new(
            10, 10, 10,
            Length::new::<micrometer>(5.0),
            Length::new::<micrometer>(10.0),
            Length::new::<micrometer>(5.0),
        ).unwrap();
        assert!(!non_uniform.is_uniform());
    }
    
    #[test]
    fn test_encapsulation() {
        let grid = Grid::default();
        
        // These should not compile (fields are private):
        // grid.nx = 0;  // Compile error!
        // grid.dx = Length::new::<meter>(0.0);  // Compile error!
        
        // Only getters work
        assert_eq!(grid.nx(), 32);
        assert!(grid.dx() > Length::new::<meter>(0.0));
    }
}