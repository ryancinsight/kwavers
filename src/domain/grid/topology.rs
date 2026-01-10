//! Grid topology trait and implementations for different coordinate systems
//!
//! This module defines the `GridTopology` trait that abstracts over different
//! coordinate systems (Cartesian, cylindrical, spherical, etc.) and provides
//! a unified interface for spatial discretization.
//!
//! # Design Philosophy
//!
//! The trait enforces mathematical invariants for coordinate transformations
//! and ensures type-safe access to grid properties across different topologies.
//!
//! # Invariants
//!
//! - Grid spacing must be positive and finite
//! - Dimensions must be non-zero
//! - Coordinate transformations must be bijective within grid bounds
//! - Wavenumber computations must respect Nyquist limits

use crate::core::error::{ConfigError, KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3};
use std::f64::consts::PI;

/// Dimensionality of the simulation domain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyDimension {
    /// One-dimensional (e.g., axial only)
    One,
    /// Two-dimensional (e.g., axisymmetric r-z, or x-y plane)
    Two,
    /// Three-dimensional (full 3D Cartesian)
    Three,
}

/// Grid topology trait for coordinate system abstraction
///
/// This trait provides a unified interface for different grid topologies,
/// enabling solvers to be written generically over coordinate systems.
///
/// # Mathematical Foundation
///
/// Each topology must:
/// - Define a mapping from integer indices to physical coordinates
/// - Provide wavenumber grids for spectral methods
/// - Compute metric coefficients (e.g., volume elements)
/// - Support coordinate bounds checking
pub trait GridTopology: Send + Sync {
    /// Get the dimensionality of this topology
    fn dimensionality(&self) -> TopologyDimension;

    /// Get total number of grid points
    fn size(&self) -> usize;

    /// Get grid dimensions as a slice
    ///
    /// Returns [n0, n1, n2] where unused dimensions should be 1
    fn dimensions(&self) -> [usize; 3];

    /// Get grid spacing as a slice
    ///
    /// Returns [d0, d1, d2] in meters
    fn spacing(&self) -> [f64; 3];

    /// Get maximum extents in each direction (meters)
    fn extents(&self) -> [f64; 3];

    /// Convert multi-dimensional indices to physical coordinates
    ///
    /// # Arguments
    ///
    /// * `indices` - Grid indices [i, j, k]
    ///
    /// # Returns
    ///
    /// Physical coordinates in the native coordinate system
    fn indices_to_coordinates(&self, indices: [usize; 3]) -> [f64; 3];

    /// Convert physical coordinates to grid indices
    ///
    /// Returns `None` if coordinates are outside the grid bounds.
    ///
    /// # Arguments
    ///
    /// * `coords` - Physical coordinates in native system
    ///
    /// # Returns
    ///
    /// Grid indices [i, j, k] or `None` if out of bounds
    fn coordinates_to_indices(&self, coords: [f64; 3]) -> Option<[usize; 3]>;

    /// Get the metric coefficient (volume/area element) at given indices
    ///
    /// For Cartesian: dx * dy * dz
    /// For cylindrical: r * dr * dz (or 2π * r * dr * dz for full volume)
    /// For spherical: r² * sin(θ) * dr * dθ * dφ
    fn metric_coefficient(&self, indices: [usize; 3]) -> f64;

    /// Check if grid has uniform spacing in all active dimensions
    fn is_uniform(&self) -> bool;

    /// Maximum wavenumber supported (Nyquist limit)
    fn k_max(&self) -> f64;

    /// Create a zero-initialized field matching this grid topology
    ///
    /// Returns an Array3 with appropriate dimensions, where unused
    /// dimensions have size 1.
    fn create_field(&self) -> Array3<f64> {
        let [n0, n1, n2] = self.dimensions();
        Array3::zeros((n0, n1, n2))
    }

    /// Validate that indices are within grid bounds
    fn validate_indices(&self, indices: [usize; 3]) -> KwaversResult<()> {
        let [n0, n1, n2] = self.dimensions();
        let [i, j, k] = indices;

        if i >= n0 || j >= n1 || k >= n2 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "grid indices".to_string(),
                value: format!("[{}, {}, {}]", i, j, k),
                constraint: format!("Must be within [0..{}, 0..{}, 0..{}]", n0, n1, n2),
            }));
        }

        Ok(())
    }
}

/// Cartesian grid topology (standard rectilinear grid)
///
/// Maps indices directly to (x, y, z) coordinates with uniform spacing.
#[derive(Debug, Clone)]
pub struct CartesianTopology {
    /// Dimensions [nx, ny, nz]
    pub dimensions: [usize; 3],
    /// Spacing [dx, dy, dz] in meters
    pub spacing: [f64; 3],
    /// Origin offset [x0, y0, z0]
    pub origin: [f64; 3],
}

impl CartesianTopology {
    /// Create a new Cartesian topology
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Number of points [nx, ny, nz]
    /// * `spacing` - Grid spacing [dx, dy, dz] in meters
    /// * `origin` - Origin coordinates [x0, y0, z0]
    pub fn new(dimensions: [usize; 3], spacing: [f64; 3], origin: [f64; 3]) -> KwaversResult<Self> {
        // Validate dimensions
        if dimensions[0] == 0 || dimensions[1] == 0 || dimensions[2] == 0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "grid dimensions".to_string(),
                value: format!("{:?}", dimensions),
                constraint: "All dimensions must be positive".to_string(),
            }));
        }

        // Validate spacing
        if spacing[0] <= 0.0 || spacing[1] <= 0.0 || spacing[2] <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "grid spacing".to_string(),
                value: format!("{:?}", spacing),
                constraint: "All spacing values must be positive".to_string(),
            }));
        }

        // Validate spacing is finite
        if !spacing[0].is_finite() || !spacing[1].is_finite() || !spacing[2].is_finite() {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "grid spacing".to_string(),
                value: format!("{:?}", spacing),
                constraint: "All spacing values must be finite".to_string(),
            }));
        }

        Ok(Self {
            dimensions,
            spacing,
            origin,
        })
    }

    /// Create a uniform Cartesian grid (same spacing in all directions)
    pub fn uniform(n: usize, spacing: f64, origin: [f64; 3]) -> KwaversResult<Self> {
        Self::new([n, n, n], [spacing, spacing, spacing], origin)
    }
}

impl GridTopology for CartesianTopology {
    fn dimensionality(&self) -> TopologyDimension {
        let active = (self.dimensions[0] > 1) as usize
            + (self.dimensions[1] > 1) as usize
            + (self.dimensions[2] > 1) as usize;

        match active {
            0 | 1 => TopologyDimension::One,
            2 => TopologyDimension::Two,
            _ => TopologyDimension::Three,
        }
    }

    fn size(&self) -> usize {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }

    fn dimensions(&self) -> [usize; 3] {
        self.dimensions
    }

    fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    fn extents(&self) -> [f64; 3] {
        [
            (self.dimensions[0] - 1) as f64 * self.spacing[0],
            (self.dimensions[1] - 1) as f64 * self.spacing[1],
            (self.dimensions[2] - 1) as f64 * self.spacing[2],
        ]
    }

    fn indices_to_coordinates(&self, indices: [usize; 3]) -> [f64; 3] {
        [
            self.origin[0] + indices[0] as f64 * self.spacing[0],
            self.origin[1] + indices[1] as f64 * self.spacing[1],
            self.origin[2] + indices[2] as f64 * self.spacing[2],
        ]
    }

    fn coordinates_to_indices(&self, coords: [f64; 3]) -> Option<[usize; 3]> {
        let extents = self.extents();

        // Check bounds
        for i in 0..3 {
            let rel = coords[i] - self.origin[i];
            if rel < 0.0 || rel > extents[i] {
                return None;
            }
        }

        // Convert to indices with clamping
        let i = ((coords[0] - self.origin[0]) / self.spacing[0])
            .floor()
            .min((self.dimensions[0] - 1) as f64) as usize;
        let j = ((coords[1] - self.origin[1]) / self.spacing[1])
            .floor()
            .min((self.dimensions[1] - 1) as f64) as usize;
        let k = ((coords[2] - self.origin[2]) / self.spacing[2])
            .floor()
            .min((self.dimensions[2] - 1) as f64) as usize;

        Some([i, j, k])
    }

    fn metric_coefficient(&self, _indices: [usize; 3]) -> f64 {
        self.spacing[0] * self.spacing[1] * self.spacing[2]
    }

    fn is_uniform(&self) -> bool {
        const EPSILON: f64 = 1e-10;
        (self.spacing[0] - self.spacing[1]).abs() < EPSILON
            && (self.spacing[1] - self.spacing[2]).abs() < EPSILON
    }

    fn k_max(&self) -> f64 {
        let min_spacing = self.spacing[0].min(self.spacing[1]).min(self.spacing[2]);
        PI / min_spacing
    }
}

/// Cylindrical grid topology for axisymmetric simulations
///
/// Maps 2D indices (i, j) to cylindrical coordinates (z, r).
/// The first index corresponds to the axial direction (z),
/// and the second to the radial direction (r).
///
/// # Coordinate Convention
///
/// - Index i → axial position z
/// - Index j → radial position r
/// - Implicit azimuthal symmetry (no φ dependence)
///
/// # Mathematical Properties
///
/// - Metric coefficient: r * dr * dz (for area), 2π * r * dr * dz (for volume)
/// - Singularity at r = 0 requires special handling
/// - Hankel transform for radial spectral methods
#[derive(Debug, Clone)]
pub struct CylindricalTopology {
    /// Number of grid points in axial (z) direction
    pub nz: usize,
    /// Number of grid points in radial (r) direction
    pub nr: usize,
    /// Grid spacing in axial direction (m)
    pub dz: f64,
    /// Grid spacing in radial direction (m)
    pub dr: f64,
    /// Axial origin offset (m)
    pub z0: f64,
    /// Radial origin (typically 0 for axis of symmetry)
    pub r0: f64,
    /// Axial coordinates (m)
    z_coords: Array1<f64>,
    /// Radial coordinates (m)
    r_coords: Array1<f64>,
    /// Axial wavenumbers (1/m)
    kz: Array1<f64>,
    /// Radial wavenumbers for Hankel transform (1/m)
    kr: Array1<f64>,
}

impl CylindricalTopology {
    /// Create a new cylindrical topology
    ///
    /// # Arguments
    ///
    /// * `nz` - Number of axial grid points
    /// * `nr` - Number of radial grid points
    /// * `dz` - Axial grid spacing (m)
    /// * `dr` - Radial grid spacing (m)
    pub fn new(nz: usize, nr: usize, dz: f64, dr: f64) -> KwaversResult<Self> {
        Self::with_origin(nz, nr, dz, dr, 0.0, 0.0)
    }

    /// Create a cylindrical topology with custom origin
    pub fn with_origin(
        nz: usize,
        nr: usize,
        dz: f64,
        dr: f64,
        z0: f64,
        r0: f64,
    ) -> KwaversResult<Self> {
        // Validate dimensions
        if nz == 0 || nr == 0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "cylindrical grid dimensions".to_string(),
                value: format!("nz={}, nr={}", nz, nr),
                constraint: "Must be positive".to_string(),
            }));
        }

        // Validate spacing
        if dz <= 0.0 || dr <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "cylindrical grid spacing".to_string(),
                value: format!("dz={}, dr={}", dz, dr),
                constraint: "Must be positive".to_string(),
            }));
        }

        if !dz.is_finite() || !dr.is_finite() {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "cylindrical grid spacing".to_string(),
                value: format!("dz={}, dr={}", dz, dr),
                constraint: "Must be finite".to_string(),
            }));
        }

        // Create spatial coordinates
        let z_coords = Array1::from_shape_fn(nz, |i| z0 + i as f64 * dz);
        let r_coords = Array1::from_shape_fn(nr, |j| r0 + j as f64 * dr);

        // Create axial wavenumbers (standard FFT frequencies)
        let kz = Self::compute_fft_wavenumbers(nz, dz);

        // Create radial wavenumbers for Hankel transform
        let kr = Self::compute_hankel_wavenumbers(nr, dr, r0);

        Ok(Self {
            nz,
            nr,
            dz,
            dr,
            z0,
            r0,
            z_coords,
            r_coords,
            kz,
            kr,
        })
    }

    /// Compute standard FFT wavenumbers
    ///
    /// Returns wavenumbers k[i] = 2π * freq[i] where freq follows FFT convention:
    /// [0, 1, 2, ..., N/2, -N/2+1, ..., -1] / (N * d)
    fn compute_fft_wavenumbers(n: usize, d: f64) -> Array1<f64> {
        let dk = 2.0 * PI / (n as f64 * d);
        Array1::from_shape_fn(n, |i| {
            if i <= n / 2 {
                i as f64 * dk
            } else {
                (i as f64 - n as f64) * dk
            }
        })
    }

    /// Compute radial wavenumbers for discrete Hankel transform
    ///
    /// Uses approximation for zeros of J₀ Bessel function:
    /// j₀ₘ ≈ (m - 0.25) * π for large m
    ///
    /// # Mathematical Foundation
    ///
    /// The discrete Hankel transform of order 0 requires sampling at
    /// zeros of the Bessel function J₀. For a radial extent r_max,
    /// the wavenumbers are k_m = j₀ₘ / r_max.
    fn compute_hankel_wavenumbers(nr: usize, dr: f64, r0: f64) -> Array1<f64> {
        let r_max = r0 + nr as f64 * dr;
        Array1::from_shape_fn(nr, |m| {
            if m == 0 {
                0.0
            } else {
                // Approximate zeros of J₀
                let j0m = (m as f64 - 0.25) * PI;
                j0m / r_max
            }
        })
    }

    /// Get axial coordinates array
    pub fn z_coordinates(&self) -> &Array1<f64> {
        &self.z_coords
    }

    /// Get radial coordinates array
    pub fn r_coordinates(&self) -> &Array1<f64> {
        &self.r_coords
    }

    /// Get axial wavenumbers
    pub fn kz_wavenumbers(&self) -> &Array1<f64> {
        &self.kz
    }

    /// Get radial wavenumbers (Hankel transform)
    pub fn kr_wavenumbers(&self) -> &Array1<f64> {
        &self.kr
    }

    /// Get physical axial coordinate for grid index
    ///
    /// This is a convenience method for backward compatibility with the old
    /// `CylindricalGrid` API.
    #[inline]
    pub fn z_at(&self, i: usize) -> f64 {
        self.z0 + i as f64 * self.dz
    }

    /// Get physical radial coordinate for grid index
    ///
    /// This is a convenience method for backward compatibility with the old
    /// `CylindricalGrid` API.
    #[inline]
    pub fn r_at(&self, j: usize) -> f64 {
        self.r0 + j as f64 * self.dr
    }

    /// Get grid index for axial coordinate (nearest)
    #[inline]
    pub fn iz_for(&self, z: f64) -> usize {
        let rel = z - self.z0;
        ((rel / self.dz).round() as usize).min(self.nz - 1)
    }

    /// Get grid index for radial coordinate (nearest)
    #[inline]
    pub fn ir_for(&self, r: f64) -> usize {
        let rel = r - self.r0;
        ((rel / self.dr).round() as usize).min(self.nr - 1)
    }

    /// Maximum axial extent (m)
    #[inline]
    pub fn z_max(&self) -> f64 {
        self.z0 + (self.nz - 1) as f64 * self.dz
    }

    /// Maximum radial extent (m)
    #[inline]
    pub fn r_max(&self) -> f64 {
        self.r0 + (self.nr - 1) as f64 * self.dr
    }

    /// Create 2D meshgrid of z coordinates
    pub fn z_mesh(&self) -> Array2<f64> {
        let mut mesh = Array2::zeros((self.nz, self.nr));
        for i in 0..self.nz {
            let z = self.z_coords[i];
            for j in 0..self.nr {
                mesh[[i, j]] = z;
            }
        }
        mesh
    }

    /// Create 2D meshgrid of r coordinates
    pub fn r_mesh(&self) -> Array2<f64> {
        let mut mesh = Array2::zeros((self.nz, self.nr));
        for i in 0..self.nz {
            for j in 0..self.nr {
                mesh[[i, j]] = self.r_coords[j];
            }
        }
        mesh
    }

    /// Calculate area element for integration: r * dr * dz
    ///
    /// At r = 0, uses half-cell width to avoid singularity: 0.5 * dr² * dz
    pub fn area_element(&self, j: usize) -> f64 {
        let r = self.r_coords[j];
        if j == 0 && self.r0.abs() < 1e-15 {
            // Special handling for axis of symmetry
            0.5 * self.dr * self.dr * self.dz
        } else {
            r * self.dr * self.dz
        }
    }

    /// Calculate volume of rotation for full 3D: 2π * r * dr * dz
    pub fn volume_element(&self, j: usize) -> f64 {
        2.0 * PI * self.area_element(j)
    }
}

impl GridTopology for CylindricalTopology {
    fn dimensionality(&self) -> TopologyDimension {
        TopologyDimension::Two
    }

    fn size(&self) -> usize {
        self.nz * self.nr
    }

    fn dimensions(&self) -> [usize; 3] {
        [self.nz, self.nr, 1]
    }

    fn spacing(&self) -> [f64; 3] {
        [self.dz, self.dr, 0.0]
    }

    fn extents(&self) -> [f64; 3] {
        [
            (self.nz - 1) as f64 * self.dz,
            (self.nr - 1) as f64 * self.dr,
            0.0,
        ]
    }

    fn indices_to_coordinates(&self, indices: [usize; 3]) -> [f64; 3] {
        let z = self.z0 + indices[0] as f64 * self.dz;
        let r = self.r0 + indices[1] as f64 * self.dr;
        [z, r, 0.0]
    }

    fn coordinates_to_indices(&self, coords: [f64; 3]) -> Option<[usize; 3]> {
        let z = coords[0];
        let r = coords[1];

        // Check bounds
        if z < self.z0 || r < self.r0 {
            return None;
        }

        let z_rel = z - self.z0;
        let r_rel = r - self.r0;

        let z_max = (self.nz - 1) as f64 * self.dz;
        let r_max = (self.nr - 1) as f64 * self.dr;

        if z_rel > z_max || r_rel > r_max {
            return None;
        }

        let i = (z_rel / self.dz).round().min((self.nz - 1) as f64) as usize;
        let j = (r_rel / self.dr).round().min((self.nr - 1) as f64) as usize;

        Some([i, j, 0])
    }

    fn metric_coefficient(&self, indices: [usize; 3]) -> f64 {
        self.area_element(indices[1])
    }

    fn is_uniform(&self) -> bool {
        const EPSILON: f64 = 1e-10;
        (self.dz - self.dr).abs() < EPSILON
    }

    fn k_max(&self) -> f64 {
        let min_spacing = self.dz.min(self.dr);
        PI / min_spacing
    }

    fn create_field(&self) -> Array3<f64> {
        // Cylindrical fields are stored as 2D arrays with shape (nz, nr)
        // embedded in 3D with third dimension = 1
        Array3::zeros((self.nz, self.nr, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartesian_creation() {
        let topo =
            CartesianTopology::new([10, 10, 10], [1e-3, 1e-3, 1e-3], [0.0, 0.0, 0.0]).unwrap();
        assert_eq!(topo.size(), 1000);
        assert_eq!(topo.dimensionality(), TopologyDimension::Three);
    }

    #[test]
    fn test_cartesian_coordinates() {
        let topo = CartesianTopology::new([10, 10, 10], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]).unwrap();

        let coords = topo.indices_to_coordinates([5, 5, 5]);
        assert!((coords[0] - 0.5).abs() < 1e-10);
        assert!((coords[1] - 0.5).abs() < 1e-10);
        assert!((coords[2] - 0.5).abs() < 1e-10);

        let indices = topo.coordinates_to_indices([0.5, 0.5, 0.5]).unwrap();
        assert_eq!(indices, [5, 5, 5]);
    }

    #[test]
    fn test_cartesian_metric() {
        let topo =
            CartesianTopology::new([10, 10, 10], [1e-3, 2e-3, 3e-3], [0.0, 0.0, 0.0]).unwrap();

        let metric = topo.metric_coefficient([5, 5, 5]);
        assert!((metric - 6e-9).abs() < 1e-15);
    }

    #[test]
    fn test_cylindrical_creation() {
        let topo = CylindricalTopology::new(64, 32, 1e-4, 1e-4).unwrap();
        assert_eq!(topo.nz, 64);
        assert_eq!(topo.nr, 32);
        assert_eq!(topo.size(), 64 * 32);
        assert_eq!(topo.dimensionality(), TopologyDimension::Two);
    }

    #[test]
    fn test_cylindrical_coordinates() {
        let topo = CylindricalTopology::new(10, 10, 0.1, 0.1).unwrap();

        let coords = topo.indices_to_coordinates([5, 3, 0]);
        assert!((coords[0] - 0.5).abs() < 1e-10); // z
        assert!((coords[1] - 0.3).abs() < 1e-10); // r

        let indices = topo.coordinates_to_indices([0.5, 0.3, 0.0]).unwrap();
        assert_eq!(indices[0], 5);
        assert_eq!(indices[1], 3);
    }

    #[test]
    fn test_cylindrical_metric() {
        let topo = CylindricalTopology::new(10, 10, 0.1, 0.1).unwrap();

        // At r = 0 (j = 0), special handling
        let metric_0 = topo.metric_coefficient([5, 0, 0]);
        assert!((metric_0 - 0.5 * 0.1 * 0.1 * 0.1).abs() < 1e-10);

        // At r = 0.3 (j = 3)
        let metric_3 = topo.metric_coefficient([5, 3, 0]);
        assert!((metric_3 - 0.3 * 0.1 * 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_cylindrical_wavenumbers() {
        let topo = CylindricalTopology::new(64, 32, 1e-4, 1e-4).unwrap();

        // DC component should be zero
        assert_eq!(topo.kz_wavenumbers()[0], 0.0);
        assert_eq!(topo.kr_wavenumbers()[0], 0.0);

        // Wavenumbers should be positive for first half
        assert!(topo.kz_wavenumbers()[1] > 0.0);
        assert!(topo.kr_wavenumbers()[1] > 0.0);
    }

    #[test]
    fn test_cylindrical_coordinate_accessors() {
        let topo = CylindricalTopology::new(10, 10, 0.1, 0.1).unwrap();

        assert_eq!(topo.z_at(0), 0.0);
        assert!((topo.z_at(5) - 0.5).abs() < 1e-10);

        assert_eq!(topo.r_at(0), 0.0);
        assert!((topo.r_at(3) - 0.3).abs() < 1e-10);

        assert_eq!(topo.iz_for(0.5), 5);
        assert_eq!(topo.ir_for(0.3), 3);

        assert!((topo.z_max() - 0.9).abs() < 1e-10);
        assert!((topo.r_max() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_cylindrical_meshgrid() {
        let topo = CylindricalTopology::new(4, 3, 1.0, 1.0).unwrap();

        let z_mesh = topo.z_mesh();
        let r_mesh = topo.r_mesh();

        assert_eq!(z_mesh.dim(), (4, 3));
        assert_eq!(r_mesh.dim(), (4, 3));

        assert_eq!(z_mesh[[2, 0]], 2.0);
        assert_eq!(r_mesh[[0, 2]], 2.0);
    }

    #[test]
    fn test_topology_field_creation() {
        let cart =
            CartesianTopology::new([10, 10, 10], [1e-3, 1e-3, 1e-3], [0.0, 0.0, 0.0]).unwrap();
        let field = cart.create_field();
        assert_eq!(field.dim(), (10, 10, 10));

        let cyl = CylindricalTopology::new(64, 32, 1e-4, 1e-4).unwrap();
        let cyl_field = cyl.create_field();
        assert_eq!(cyl_field.dim(), (64, 32, 1));
    }

    #[test]
    fn test_invalid_dimensions() {
        let result = CartesianTopology::new([0, 10, 10], [1e-3, 1e-3, 1e-3], [0.0, 0.0, 0.0]);
        assert!(result.is_err());

        let result = CylindricalTopology::new(0, 10, 1e-3, 1e-3);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_spacing() {
        let result = CartesianTopology::new([10, 10, 10], [0.0, 1e-3, 1e-3], [0.0, 0.0, 0.0]);
        assert!(result.is_err());

        let result = CartesianTopology::new([10, 10, 10], [-1e-3, 1e-3, 1e-3], [0.0, 0.0, 0.0]);
        assert!(result.is_err());

        let result = CylindricalTopology::new(10, 10, f64::INFINITY, 1e-3);
        assert!(result.is_err());
    }
}
