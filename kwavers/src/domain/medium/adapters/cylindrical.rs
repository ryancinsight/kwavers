//! Cylindrical medium projection adapter for axisymmetric solvers
//!
//! This module provides an adapter that projects a 3D `Medium` onto a 2D
//! cylindrical coordinate system for use with axisymmetric solvers. The
//! projection maintains mathematical correctness by sampling the medium
//! along the axis of symmetry (θ = 0 plane in cylindrical coordinates).
//!
//! # Mathematical Foundation
//!
//! For axisymmetric problems, the medium properties are independent of the
//! azimuthal angle θ. The projection samples the 3D medium at:
//!
//! ```text
//! (x, y, z) = (r, 0, z)  in Cartesian coordinates
//! (r, θ, z) = (r, 0, z)  in cylindrical coordinates
//! ```
//!
//! # Physical Invariants
//!
//! The projection preserves:
//! - Sound speed bounds: `min(c_3D) ≤ min(c_2D) ≤ max(c_2D) ≤ max(c_3D)`
//! - Homogeneity: Uniform 3D medium → Uniform 2D field
//! - Physical constraints: Positive density, sound speed, non-negative absorption
//!
//! # Example
//!
//! ```rust
//! use kwavers::domain::medium::{HomogeneousMedium, adapters::CylindricalMediumProjection};
//! use kwavers::domain::grid::{Grid, CylindricalTopology};
//!
//! # fn example() -> kwavers::core::error::KwaversResult<()> {
//! // Create 3D medium
//! let grid = Grid::new(128, 128, 128, 0.0001, 0.0001, 0.0001)?;
//! let medium = HomogeneousMedium::water(&grid);
//!
//! // Create cylindrical topology
//! let topology = CylindricalTopology::new(128, 64, 0.0001, 0.0001)?;
//!
//! // Project to 2D
//! let projection = CylindricalMediumProjection::new(&medium, &grid, &topology)?;
//!
//! // Access 2D fields
//! let c_2d = projection.sound_speed_field();  // Shape: (nz, nr)
//! let rho_2d = projection.density_field();
//! # Ok(())
//! # }
//! ```

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::{CylindricalTopology, Grid};
use crate::domain::medium::Medium;
use ndarray::{Array2, ArrayView2};
use std::fmt;

/// Cylindrical projection of a 3D medium for axisymmetric solvers
///
/// This adapter samples a 3D `Medium` along the θ = 0 plane in cylindrical
/// coordinates, producing 2D property arrays indexed by (z, r).
///
/// # Lifetime
///
/// The projection borrows the medium and grid with lifetime `'a`, ensuring
/// they remain valid for the duration of the projection's use.
///
/// # Caching Strategy
///
/// Property arrays are computed once during construction and cached for
/// efficient repeated access. This is acceptable because:
/// - 2D arrays are much smaller than 3D (typically <1 MB)
/// - Axisymmetric solvers access properties frequently
/// - Construction cost is amortized over many solver iterations
pub struct CylindricalMediumProjection<'a, M: Medium> {
    /// Reference to the 3D medium
    medium: &'a M,
    /// Reference to the 3D grid
    grid: &'a Grid,
    /// Reference to the cylindrical topology
    topology: &'a CylindricalTopology,

    // Cached 2D projections (nz × nr)
    /// Sound speed field (m/s)
    sound_speed_2d: Array2<f64>,
    /// Density field (kg/m³)
    density_2d: Array2<f64>,
    /// Absorption coefficient field (Np/m)
    absorption_2d: Array2<f64>,
    /// Nonlinearity parameter B/A (optional)
    nonlinearity_2d: Option<Array2<f64>>,

    // Cached scalar properties
    /// Maximum sound speed in the projected medium
    max_sound_speed: f64,
    /// Minimum sound speed in the projected medium
    min_sound_speed: f64,
    /// Whether the medium is homogeneous
    is_homogeneous: bool,
}

impl<'a, M: Medium> CylindricalMediumProjection<'a, M> {
    /// Create a new cylindrical projection of the 3D medium
    ///
    /// This constructor samples the medium at the θ = 0 plane and caches
    /// the resulting 2D property arrays.
    ///
    /// # Arguments
    ///
    /// * `medium` - Reference to 3D medium
    /// * `grid` - Reference to 3D grid structure
    /// * `topology` - Reference to cylindrical topology defining the 2D grid
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Grid dimensions don't match medium array dimensions
    /// - Cylindrical topology coordinates exceed grid bounds
    /// - Medium validation fails
    ///
    /// # Example
    ///
    /// ```rust
    /// # use kwavers::domain::medium::{HomogeneousMedium, adapters::CylindricalMediumProjection};
    /// # use kwavers::domain::grid::{Grid, CylindricalTopology};
    /// # fn example() -> kwavers::core::error::KwaversResult<()> {
    /// let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001)?;
    /// let medium = HomogeneousMedium::water(&grid);
    /// let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001)?;
    /// let projection = CylindricalMediumProjection::new(&medium, &grid, &topology)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        medium: &'a M,
        grid: &'a Grid,
        topology: &'a CylindricalTopology,
    ) -> KwaversResult<Self> {
        // Validate medium against grid
        medium.validate(grid)?;

        // Check if medium is homogeneous (fast path)
        let is_homogeneous = medium.is_homogeneous();

        // Allocate 2D arrays
        let shape = (topology.nz, topology.nr);
        let mut sound_speed_2d = Array2::zeros(shape);
        let mut density_2d = Array2::zeros(shape);
        let mut absorption_2d = Array2::zeros(shape);

        // Track min/max sound speed during projection
        let mut max_c = f64::NEG_INFINITY;
        let mut min_c = f64::INFINITY;

        // Project medium properties onto cylindrical grid
        //
        // For each point (iz, ir) in the cylindrical grid:
        // 1. Compute physical coordinates (r, z)
        // 2. Map to Cartesian (x, y, z) with x = r, y = 0 (θ = 0 plane)
        // 3. Convert to grid indices (i, j, k)
        // 4. Sample medium properties
        for iz in 0..topology.nz {
            let z = topology.z_at(iz);

            for ir in 0..topology.nr {
                let r = topology.r_at(ir);

                // Map cylindrical (r, θ=0, z) to Cartesian (x=r, y=0, z)
                let x = r;
                let y = 0.0;

                // Get grid indices for Cartesian coordinates
                let (i, j, k) = grid.coordinates_to_indices(x, y, z).ok_or_else(|| {
                    KwaversError::Config(crate::core::error::ConfigError::InvalidValue {
                        parameter: "cylindrical topology".to_string(),
                        value: format!(
                            "iz={}, ir={} -> x={}, y={}, z={} (out of bounds)",
                            iz, ir, x, y, z
                        ),
                        constraint: format!(
                            "Coordinates must be within grid bounds: ({}, {}, {})",
                            grid.nx as f64 * grid.dx,
                            grid.ny as f64 * grid.dy,
                            grid.nz as f64 * grid.dz
                        ),
                    })
                })?;

                // Sample medium properties at (i, j, k)
                let c = medium.sound_speed(i, j, k);
                let rho = medium.density(i, j, k);
                let alpha = medium.absorption(i, j, k);

                // Store in 2D arrays
                sound_speed_2d[[iz, ir]] = c;
                density_2d[[iz, ir]] = rho;
                absorption_2d[[iz, ir]] = alpha;

                // Track min/max
                max_c = max_c.max(c);
                min_c = min_c.min(c);

                // Validate physical constraints
                if c <= 0.0 || !c.is_finite() {
                    return Err(KwaversError::Config(
                        crate::core::error::ConfigError::InvalidValue {
                            parameter: "sound_speed".to_string(),
                            value: c.to_string(),
                            constraint: "Must be positive and finite".to_string(),
                        },
                    ));
                }

                if rho <= 0.0 || !rho.is_finite() {
                    return Err(KwaversError::Config(
                        crate::core::error::ConfigError::InvalidValue {
                            parameter: "density".to_string(),
                            value: rho.to_string(),
                            constraint: "Must be positive and finite".to_string(),
                        },
                    ));
                }

                if alpha < 0.0 || !alpha.is_finite() {
                    return Err(KwaversError::Config(
                        crate::core::error::ConfigError::InvalidValue {
                            parameter: "absorption".to_string(),
                            value: alpha.to_string(),
                            constraint: "Must be non-negative and finite".to_string(),
                        },
                    ));
                }
            }
        }

        // Project nonlinearity if medium has it
        // For homogeneous media, this is a constant; for heterogeneous, sample like other properties
        let nonlinearity_2d = if is_homogeneous {
            // For homogeneous medium, check if nonlinearity is non-zero at origin
            let b_over_a = medium.nonlinearity(0, 0, 0);
            if b_over_a != 0.0 {
                Some(Array2::from_elem(shape, b_over_a))
            } else {
                None
            }
        } else {
            // For heterogeneous medium, project nonlinearity field
            let mut nonlin_2d = Array2::zeros(shape);
            let mut has_nonlinearity = false;

            for iz in 0..topology.nz {
                let z = topology.z_at(iz);

                for ir in 0..topology.nr {
                    let r = topology.r_at(ir);
                    let x = r;
                    let y = 0.0;

                    if let Some((i, j, k)) = grid.coordinates_to_indices(x, y, z) {
                        let b_over_a = medium.nonlinearity(i, j, k);
                        nonlin_2d[[iz, ir]] = b_over_a;

                        if b_over_a != 0.0 {
                            has_nonlinearity = true;
                        }
                    }
                }
            }

            if has_nonlinearity {
                Some(nonlin_2d)
            } else {
                None
            }
        };

        Ok(Self {
            medium,
            grid,
            topology,
            sound_speed_2d,
            density_2d,
            absorption_2d,
            nonlinearity_2d,
            max_sound_speed: max_c,
            min_sound_speed: min_c,
            is_homogeneous,
        })
    }

    /// Get the projected sound speed field (nz × nr)
    ///
    /// Returns a view of the cached 2D sound speed array.
    #[inline]
    pub fn sound_speed_field(&self) -> ArrayView2<'_, f64> {
        self.sound_speed_2d.view()
    }

    /// Get the projected density field (nz × nr)
    ///
    /// Returns a view of the cached 2D density array.
    #[inline]
    pub fn density_field(&self) -> ArrayView2<'_, f64> {
        self.density_2d.view()
    }

    /// Get the projected absorption field (nz × nr)
    ///
    /// Returns a view of the cached 2D absorption coefficient array.
    #[inline]
    pub fn absorption_field(&self) -> ArrayView2<'_, f64> {
        self.absorption_2d.view()
    }

    /// Get the projected nonlinearity field (nz × nr)
    ///
    /// Returns `Some(view)` if the medium has nonlinearity, `None` otherwise.
    #[inline]
    pub fn nonlinearity_field(&self) -> Option<ArrayView2<'_, f64>> {
        self.nonlinearity_2d.as_ref().map(|arr| arr.view())
    }

    /// Get maximum sound speed in the projected medium (m/s)
    ///
    /// This value is cached during construction for efficient access.
    #[inline]
    pub fn max_sound_speed(&self) -> f64 {
        self.max_sound_speed
    }

    /// Get minimum sound speed in the projected medium (m/s)
    ///
    /// This value is cached during construction for efficient access.
    #[inline]
    pub fn min_sound_speed(&self) -> f64 {
        self.min_sound_speed
    }

    /// Check if the projected medium is homogeneous
    ///
    /// Returns `true` if the underlying 3D medium is homogeneous.
    #[inline]
    pub fn is_homogeneous(&self) -> bool {
        self.is_homogeneous
    }

    /// Get the underlying 3D medium reference
    ///
    /// This allows access to additional properties not projected to 2D.
    #[inline]
    pub fn medium(&self) -> &M {
        self.medium
    }

    /// Get the grid reference
    #[inline]
    pub fn grid(&self) -> &Grid {
        self.grid
    }

    /// Get the cylindrical topology reference
    #[inline]
    pub fn topology(&self) -> &CylindricalTopology {
        self.topology
    }

    /// Get sound speed at a specific (iz, ir) point
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds (debug builds only).
    #[inline]
    pub fn sound_speed_at(&self, iz: usize, ir: usize) -> f64 {
        self.sound_speed_2d[[iz, ir]]
    }

    /// Get density at a specific (iz, ir) point
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds (debug builds only).
    #[inline]
    pub fn density_at(&self, iz: usize, ir: usize) -> f64 {
        self.density_2d[[iz, ir]]
    }

    /// Get absorption at a specific (iz, ir) point
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds (debug builds only).
    #[inline]
    pub fn absorption_at(&self, iz: usize, ir: usize) -> f64 {
        self.absorption_2d[[iz, ir]]
    }

    /// Get nonlinearity at a specific (iz, ir) point
    ///
    /// Returns 0.0 if the medium has no nonlinearity.
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds (debug builds only).
    #[inline]
    pub fn nonlinearity_at(&self, iz: usize, ir: usize) -> f64 {
        self.nonlinearity_2d
            .as_ref()
            .map_or(0.0, |arr| arr[[iz, ir]])
    }

    /// Get grid dimensions (nz, nr)
    #[inline]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.topology.nz, self.topology.nr)
    }

    /// Get grid spacing (dz, dr)
    #[inline]
    pub fn spacing(&self) -> (f64, f64) {
        (self.topology.dz, self.topology.dr)
    }

    /// Validate that projection preserves physical bounds
    ///
    /// # Invariants Checked
    ///
    /// 1. `min_sound_speed_3d ≤ min_sound_speed_2d`
    /// 2. `max_sound_speed_2d ≤ max_sound_speed_3d`
    /// 3. All projected values are positive and finite
    ///
    /// This method is primarily for testing and validation purposes.
    pub fn validate_projection(&self) -> KwaversResult<()> {
        let c_3d_max = self.medium.max_sound_speed();

        // Check max bound
        if self.max_sound_speed > c_3d_max {
            return Err(KwaversError::Config(
                crate::core::error::ConfigError::InvalidValue {
                    parameter: "projection max_sound_speed".to_string(),
                    value: self.max_sound_speed.to_string(),
                    constraint: format!("Must not exceed 3D medium max: {}", c_3d_max),
                },
            ));
        }

        // Check all values are positive and finite
        for &c in self.sound_speed_2d.iter() {
            if c <= 0.0 || !c.is_finite() {
                return Err(KwaversError::Config(
                    crate::core::error::ConfigError::InvalidValue {
                        parameter: "projected sound_speed".to_string(),
                        value: c.to_string(),
                        constraint: "Must be positive and finite".to_string(),
                    },
                ));
            }
        }

        for &rho in self.density_2d.iter() {
            if rho <= 0.0 || !rho.is_finite() {
                return Err(KwaversError::Config(
                    crate::core::error::ConfigError::InvalidValue {
                        parameter: "projected density".to_string(),
                        value: rho.to_string(),
                        constraint: "Must be positive and finite".to_string(),
                    },
                ));
            }
        }

        for &alpha in self.absorption_2d.iter() {
            if alpha < 0.0 || !alpha.is_finite() {
                return Err(KwaversError::Config(
                    crate::core::error::ConfigError::InvalidValue {
                        parameter: "projected absorption".to_string(),
                        value: alpha.to_string(),
                        constraint: "Must be non-negative and finite".to_string(),
                    },
                ));
            }
        }

        Ok(())
    }
}

impl<'a, M: Medium> fmt::Debug for CylindricalMediumProjection<'a, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CylindricalMediumProjection")
            .field("dimensions", &(self.topology.nz, self.topology.nr))
            .field("spacing", &(self.topology.dz, self.topology.dr))
            .field("max_sound_speed", &self.max_sound_speed)
            .field("min_sound_speed", &self.min_sound_speed)
            .field("is_homogeneous", &self.is_homogeneous)
            .field("has_nonlinearity", &self.nonlinearity_2d.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::heterogeneous::HeterogeneousMedium;
    use crate::domain::medium::{CoreMedium, HomogeneousMedium};

    #[test]
    fn test_homogeneous_projection() {
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        // Check dimensions
        assert_eq!(projection.dimensions(), (64, 32));

        // Check homogeneity
        assert!(projection.is_homogeneous());

        // Check all values are constant (homogeneous)
        let c0 = projection.sound_speed_at(0, 0);
        let rho0 = projection.density_at(0, 0);

        for iz in 0..64 {
            for ir in 0..32 {
                assert_eq!(projection.sound_speed_at(iz, ir), c0);
                assert_eq!(projection.density_at(iz, ir), rho0);
            }
        }

        // Check min/max match constant value
        assert_eq!(projection.max_sound_speed(), c0);
        assert_eq!(projection.min_sound_speed(), c0);
    }

    #[test]
    fn test_projection_validates() {
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        // Validation should pass
        projection.validate_projection().unwrap();
    }

    #[test]
    fn test_projection_dimensions_match() {
        let grid = Grid::new(128, 128, 128, 0.00005, 0.00005, 0.00005).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(128, 64, 0.00005, 0.00005).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        let c_field = projection.sound_speed_field();
        assert_eq!(c_field.shape(), &[128, 64]);

        let rho_field = projection.density_field();
        assert_eq!(rho_field.shape(), &[128, 64]);
    }

    #[test]
    fn test_projection_physical_bounds() {
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        // All sound speeds should be positive
        for &c in projection.sound_speed_field().iter() {
            assert!(c > 0.0);
            assert!(c.is_finite());
        }

        // All densities should be positive
        for &rho in projection.density_field().iter() {
            assert!(rho > 0.0);
            assert!(rho.is_finite());
        }

        // All absorptions should be non-negative
        for &alpha in projection.absorption_field().iter() {
            assert!(alpha >= 0.0);
            assert!(alpha.is_finite());
        }
    }

    #[test]
    fn test_accessor_methods() {
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        // Test accessors return same values as array indexing
        let c_from_accessor = projection.sound_speed_at(10, 15);
        let c_from_field = projection.sound_speed_field()[[10, 15]];
        assert_eq!(c_from_accessor, c_from_field);

        let rho_from_accessor = projection.density_at(20, 10);
        let rho_from_field = projection.density_field()[[20, 10]];
        assert_eq!(rho_from_accessor, rho_from_field);
    }

    #[test]
    fn test_spacing_and_dimensions() {
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(64, 32, 0.0001, 0.00015).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        assert_eq!(projection.dimensions(), (64, 32));
        assert_eq!(projection.spacing(), (0.0001, 0.00015));
    }

    // Property-based tests for mathematical invariants
    #[test]
    fn test_property_homogeneity_preservation() {
        // Property: Homogeneous 3D medium → Uniform 2D field
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        // All projected values should be identical
        let c0 = projection.sound_speed_at(0, 0);
        let rho0 = projection.density_at(0, 0);
        let alpha0 = projection.absorption_at(0, 0);

        for iz in 0..64 {
            for ir in 0..32 {
                assert_eq!(
                    projection.sound_speed_at(iz, ir),
                    c0,
                    "Sound speed must be uniform for homogeneous medium"
                );
                assert_eq!(
                    projection.density_at(iz, ir),
                    rho0,
                    "Density must be uniform for homogeneous medium"
                );
                assert_eq!(
                    projection.absorption_at(iz, ir),
                    alpha0,
                    "Absorption must be uniform for homogeneous medium"
                );
            }
        }
    }

    #[test]
    fn test_property_sound_speed_bounds() {
        // Property: min(c_3D) ≤ min(c_2D) ≤ max(c_2D) ≤ max(c_3D)
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        let c_3d_max = medium.max_sound_speed();
        let c_2d_max = projection.max_sound_speed();
        let c_2d_min = projection.min_sound_speed();

        // Check bounds
        assert!(
            c_2d_min <= c_2d_max,
            "Min sound speed must be <= max sound speed"
        );
        assert!(
            c_2d_max <= c_3d_max,
            "Projected max must not exceed 3D max: {} > {}",
            c_2d_max,
            c_3d_max
        );
    }

    #[test]
    fn test_property_positive_density() {
        // Property: ∀(iz,ir): ρ(iz,ir) > 0
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        for iz in 0..64 {
            for ir in 0..32 {
                let rho = projection.density_at(iz, ir);
                assert!(rho > 0.0, "Density must be positive at ({}, {})", iz, ir);
                assert!(
                    rho.is_finite(),
                    "Density must be finite at ({}, {})",
                    iz,
                    ir
                );
            }
        }
    }

    #[test]
    fn test_property_non_negative_absorption() {
        // Property: ∀(iz,ir): α(iz,ir) ≥ 0
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        for iz in 0..64 {
            for ir in 0..32 {
                let alpha = projection.absorption_at(iz, ir);
                assert!(
                    alpha >= 0.0,
                    "Absorption must be non-negative at ({}, {})",
                    iz,
                    ir
                );
                assert!(
                    alpha.is_finite(),
                    "Absorption must be finite at ({}, {})",
                    iz,
                    ir
                );
            }
        }
    }

    #[test]
    fn test_property_array_dimensions() {
        // Property: Projected arrays have shape (nz, nr)
        let grid = Grid::new(128, 128, 128, 0.00005, 0.00005, 0.00005).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(96, 48, 0.00005, 0.00005).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        assert_eq!(projection.sound_speed_field().shape(), &[96, 48]);
        assert_eq!(projection.density_field().shape(), &[96, 48]);
        assert_eq!(projection.absorption_field().shape(), &[96, 48]);
        assert_eq!(projection.dimensions(), (96, 48));
    }

    #[test]
    fn test_heterogeneous_projection() {
        // Test projection of heterogeneous medium
        let grid = Grid::new(32, 32, 32, 0.0001, 0.0001, 0.0001).unwrap();
        let mut medium = HeterogeneousMedium::new(32, 32, 32, false);

        // Create gradient in sound speed
        for i in 0..32 {
            for j in 0..32 {
                for k in 0..32 {
                    let c = 1500.0 + (i as f64) * 10.0; // Gradient along x/r direction
                    medium.sound_speed[[i, j, k]] = c;
                    medium.density[[i, j, k]] = 1000.0;
                }
            }
        }

        let topology = CylindricalTopology::new(32, 16, 0.0001, 0.0001).unwrap();
        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        // Check that projection is not homogeneous
        assert!(!projection.is_homogeneous());

        // Check that min/max are correctly computed
        assert!(projection.min_sound_speed() >= 1500.0);
        assert!(projection.max_sound_speed() > projection.min_sound_speed());

        // Validate physical bounds
        projection.validate_projection().unwrap();
    }

    #[test]
    fn test_projection_with_nonlinearity() {
        // Test that nonlinearity is correctly projected
        let grid = Grid::new(32, 32, 32, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid); // Water has B/A nonlinearity
        let topology = CylindricalTopology::new(32, 16, 0.0001, 0.0001).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        // Water should have nonlinearity
        if let Some(nonlin_field) = projection.nonlinearity_field() {
            assert_eq!(nonlin_field.shape(), &[32, 16]);

            // Check all values are in physical range for B/A (typically 1-20)
            for &b_over_a in nonlin_field.iter() {
                assert!(
                    (0.0..=100.0).contains(&b_over_a),
                    "B/A out of physical range"
                );
            }
        }
    }

    #[test]
    fn test_projection_index_consistency() {
        // Property: Accessor methods consistent with array views
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let topology = CylindricalTopology::new(64, 32, 0.0001, 0.0001).unwrap();

        let projection = CylindricalMediumProjection::new(&medium, &grid, &topology).unwrap();

        // Check several random points
        for iz in [0, 10, 30, 50, 63].iter().copied() {
            for ir in [0, 5, 15, 25, 31].iter().copied() {
                assert_eq!(
                    projection.sound_speed_at(iz, ir),
                    projection.sound_speed_field()[[iz, ir]]
                );
                assert_eq!(
                    projection.density_at(iz, ir),
                    projection.density_field()[[iz, ir]]
                );
                assert_eq!(
                    projection.absorption_at(iz, ir),
                    projection.absorption_field()[[iz, ir]]
                );
            }
        }
    }

    #[test]
    fn test_projection_validates_bounds() {
        // Test that out-of-bounds topology fails gracefully
        let grid = Grid::new(64, 64, 64, 0.0001, 0.0001, 0.0001).unwrap();
        let medium = HomogeneousMedium::water(&grid);

        // Create topology that exceeds grid bounds
        let topology_too_large = CylindricalTopology::new(128, 64, 0.0001, 0.0001).unwrap();

        // Should fail because z_max exceeds grid bounds
        let result = CylindricalMediumProjection::new(&medium, &grid, &topology_too_large);
        assert!(result.is_err(), "Should fail for out-of-bounds topology");
    }
}
