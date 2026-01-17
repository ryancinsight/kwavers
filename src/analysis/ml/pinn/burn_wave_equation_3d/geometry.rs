//! 3D Geometry primitives for PINN domains
//!
//! This module defines geometric primitives for defining spatial domains in 3D wave equation
//! simulations. Supports rectangular, spherical, cylindrical, and multi-region geometries.
//!
//! ## Supported Geometries
//!
//! - **Rectangular**: Axis-aligned boxes (x_min..x_max, y_min..y_max, z_min..z_max)
//! - **Spherical**: Sphere with center and radius
//! - **Cylindrical**: Cylinder with axis parallel to z
//! - **MultiRegion**: Composite geometry with multiple regions and interfaces
//!
//! ## Mathematical Specifications
//!
//! ### Rectangular Domain
//!
//! Ω = {(x,y,z) : x_min ≤ x ≤ x_max, y_min ≤ y ≤ y_max, z_min ≤ z ≤ z_max}
//!
//! ### Spherical Domain
//!
//! Ω = {(x,y,z) : √[(x-xc)² + (y-yc)² + (z-zc)²] ≤ r}
//!
//! ### Cylindrical Domain
//!
//! Ω = {(x,y,z) : (x-xc)² + (y-yc)² ≤ r², z_min ≤ z ≤ z_max}

use super::types::InterfaceCondition3D;

/// 3D Geometry definitions for PINN domains
///
/// Defines the spatial domain Ω ⊂ ℝ³ for wave equation simulation.
/// Supports simple and complex geometries for heterogeneous media.
///
/// # Examples
///
/// ```rust,ignore
/// use kwavers::analysis::ml::pinn::burn_wave_equation_3d::Geometry3D;
///
/// // Unit cube
/// let cube = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
///
/// // Unit sphere at origin
/// let sphere = Geometry3D::spherical(0.0, 0.0, 0.0, 1.0);
///
/// // Cylinder (height 2.0, radius 0.5)
/// let cylinder = Geometry3D::cylindrical(0.0, 0.0, -1.0, 1.0, 0.5);
/// ```
#[derive(Debug, Clone)]
pub enum Geometry3D {
    /// Rectangular box domain
    ///
    /// Domain: {(x,y,z) : x_min ≤ x ≤ x_max, y_min ≤ y ≤ y_max, z_min ≤ z ≤ z_max}
    Rectangular {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_min: f64,
        z_max: f64,
    },

    /// Spherical domain
    ///
    /// Domain: {(x,y,z) : √[(x-xc)² + (y-yc)² + (z-zc)²] ≤ radius}
    Spherical {
        x_center: f64,
        y_center: f64,
        z_center: f64,
        radius: f64,
    },

    /// Cylindrical domain (axis parallel to z)
    ///
    /// Domain: {(x,y,z) : (x-xc)² + (y-yc)² ≤ r², z_min ≤ z ≤ z_max}
    Cylindrical {
        x_center: f64,
        y_center: f64,
        z_min: f64,
        z_max: f64,
        radius: f64,
    },

    /// Multi-region domain for heterogeneous media
    ///
    /// Union of multiple sub-geometries with interface conditions.
    /// Enables simulation of layered or composite materials.
    MultiRegion {
        /// List of sub-regions with their geometries and region IDs
        regions: Vec<(Geometry3D, usize)>,
        /// Interface conditions between regions
        interfaces: Vec<InterfaceCondition3D>,
    },
}

impl Geometry3D {
    /// Create a rectangular geometry
    ///
    /// # Arguments
    ///
    /// - `x_min`, `x_max`: x-coordinate bounds
    /// - `y_min`, `y_max`: y-coordinate bounds
    /// - `z_min`, `z_max`: z-coordinate bounds
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let cube = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    /// ```
    pub fn rectangular(
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_min: f64,
        z_max: f64,
    ) -> Self {
        Self::Rectangular {
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
        }
    }

    /// Create a spherical geometry
    ///
    /// # Arguments
    ///
    /// - `x_center`, `y_center`, `z_center`: Center coordinates
    /// - `radius`: Sphere radius
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let sphere = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
    /// ```
    pub fn spherical(x_center: f64, y_center: f64, z_center: f64, radius: f64) -> Self {
        Self::Spherical {
            x_center,
            y_center,
            z_center,
            radius,
        }
    }

    /// Create a cylindrical geometry (axis parallel to z)
    ///
    /// # Arguments
    ///
    /// - `x_center`, `y_center`: Center coordinates in xy-plane
    /// - `z_min`, `z_max`: Axial extent
    /// - `radius`: Cylinder radius
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let cylinder = Geometry3D::cylindrical(0.0, 0.0, -1.0, 1.0, 0.5);
    /// ```
    pub fn cylindrical(x_center: f64, y_center: f64, z_min: f64, z_max: f64, radius: f64) -> Self {
        Self::Cylindrical {
            x_center,
            y_center,
            z_min,
            z_max,
            radius,
        }
    }

    /// Get bounding box: (x_min, x_max, y_min, y_max, z_min, z_max)
    ///
    /// Returns the axis-aligned bounding box (AABB) that fully contains the geometry.
    /// Useful for sampling collocation points and domain visualization.
    ///
    /// # Returns
    ///
    /// Tuple: `(x_min, x_max, y_min, y_max, z_min, z_max)`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let sphere = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
    /// let (x_min, x_max, y_min, y_max, z_min, z_max) = sphere.bounding_box();
    /// assert_eq!(x_min, 0.2);
    /// assert_eq!(x_max, 0.8);
    /// ```
    pub fn bounding_box(&self) -> (f64, f64, f64, f64, f64, f64) {
        match self {
            Geometry3D::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
                z_min,
                z_max,
            } => (*x_min, *x_max, *y_min, *y_max, *z_min, *z_max),
            Geometry3D::Spherical {
                x_center,
                y_center,
                z_center,
                radius,
            } => (
                x_center - radius,
                x_center + radius,
                y_center - radius,
                y_center + radius,
                z_center - radius,
                z_center + radius,
            ),
            Geometry3D::Cylindrical {
                x_center,
                y_center,
                z_min,
                z_max,
                radius,
            } => (
                x_center - radius,
                x_center + radius,
                y_center - radius,
                y_center + radius,
                *z_min,
                *z_max,
            ),
            Geometry3D::MultiRegion { regions, .. } => {
                let mut x_min = f64::INFINITY;
                let mut x_max = f64::NEG_INFINITY;
                let mut y_min = f64::INFINITY;
                let mut y_max = f64::NEG_INFINITY;
                let mut z_min = f64::INFINITY;
                let mut z_max = f64::NEG_INFINITY;

                for (geom, _) in regions {
                    let (gx_min, gx_max, gy_min, gy_max, gz_min, gz_max) = geom.bounding_box();
                    x_min = x_min.min(gx_min);
                    x_max = x_max.max(gx_max);
                    y_min = y_min.min(gy_min);
                    y_max = y_max.max(gy_max);
                    z_min = z_min.min(gz_min);
                    z_max = z_max.max(gz_max);
                }

                (x_min, x_max, y_min, y_max, z_min, z_max)
            }
        }
    }

    pub fn interior_point(&self) -> (f64, f64, f64) {
        match self {
            Geometry3D::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
                z_min,
                z_max,
            } => (
                (x_min + x_max) * 0.5,
                (y_min + y_max) * 0.5,
                (z_min + z_max) * 0.5,
            ),
            Geometry3D::Spherical {
                x_center,
                y_center,
                z_center,
                ..
            } => (*x_center, *y_center, *z_center),
            Geometry3D::Cylindrical {
                x_center,
                y_center,
                z_min,
                z_max,
                ..
            } => (*x_center, *y_center, (z_min + z_max) * 0.5),
            Geometry3D::MultiRegion { regions, .. } => regions
                .first()
                .map(|(geom, _)| geom.interior_point())
                .unwrap_or((0.0, 0.0, 0.0)),
        }
    }

    /// Check if point (x, y, z) is inside geometry
    ///
    /// Returns `true` if the point is contained within the domain Ω.
    ///
    /// # Mathematical Definition
    ///
    /// - **Rectangular**: x_min ≤ x ≤ x_max ∧ y_min ≤ y ≤ y_max ∧ z_min ≤ z ≤ z_max
    /// - **Spherical**: √[(x-xc)² + (y-yc)² + (z-zc)²] ≤ r
    /// - **Cylindrical**: (x-xc)² + (y-yc)² ≤ r² ∧ z_min ≤ z ≤ z_max
    /// - **MultiRegion**: Point in any sub-region
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let cube = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    /// assert!(cube.contains(0.5, 0.5, 0.5));  // Center point
    /// assert!(!cube.contains(1.5, 0.5, 0.5)); // Outside
    /// ```
    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        match self {
            Geometry3D::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
                z_min,
                z_max,
            } => {
                x >= *x_min
                    && x <= *x_max
                    && y >= *y_min
                    && y <= *y_max
                    && z >= *z_min
                    && z <= *z_max
            }
            Geometry3D::Spherical {
                x_center,
                y_center,
                z_center,
                radius,
            } => {
                let dx = x - x_center;
                let dy = y - y_center;
                let dz = z - z_center;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let radius_sq = radius * radius;
                dist_sq <= radius_sq + 1e-12_f64 * (1.0 + radius_sq.abs())
            }
            Geometry3D::Cylindrical {
                x_center,
                y_center,
                z_min,
                z_max,
                radius,
            } => {
                let dx = x - x_center;
                let dy = y - y_center;
                let r_squared = dx * dx + dy * dy;
                let radius_sq = radius * radius;
                r_squared <= radius_sq + 1e-12_f64 * (1.0 + radius_sq.abs())
                    && z >= *z_min
                    && z <= *z_max
            }
            Geometry3D::MultiRegion { regions, .. } => {
                regions.iter().any(|(geom, _)| geom.contains(x, y, z))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular_geometry() {
        let geom = Geometry3D::rectangular(0.0, 2.0, 1.0, 3.0, -1.0, 1.0);
        if let Geometry3D::Rectangular {
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
        } = geom
        {
            assert_eq!(x_min, 0.0);
            assert_eq!(x_max, 2.0);
            assert_eq!(y_min, 1.0);
            assert_eq!(y_max, 3.0);
            assert_eq!(z_min, -1.0);
            assert_eq!(z_max, 1.0);
        } else {
            panic!("Expected Rectangular geometry");
        }
    }

    #[test]
    fn test_spherical_geometry() {
        let geom = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
        if let Geometry3D::Spherical {
            x_center,
            y_center,
            z_center,
            radius,
        } = geom
        {
            assert_eq!(x_center, 0.5);
            assert_eq!(y_center, 0.5);
            assert_eq!(z_center, 0.5);
            assert_eq!(radius, 0.3);
        } else {
            panic!("Expected Spherical geometry");
        }
    }

    #[test]
    fn test_cylindrical_geometry() {
        let geom = Geometry3D::cylindrical(0.0, 0.0, -1.0, 1.0, 0.5);
        if let Geometry3D::Cylindrical {
            x_center,
            y_center,
            z_min,
            z_max,
            radius,
        } = geom
        {
            assert_eq!(x_center, 0.0);
            assert_eq!(y_center, 0.0);
            assert_eq!(z_min, -1.0);
            assert_eq!(z_max, 1.0);
            assert_eq!(radius, 0.5);
        } else {
            panic!("Expected Cylindrical geometry");
        }
    }

    #[test]
    fn test_bounding_box_rectangular() {
        let geom = Geometry3D::rectangular(0.0, 2.0, 1.0, 3.0, -1.0, 1.0);
        let (x_min, x_max, y_min, y_max, z_min, z_max) = geom.bounding_box();
        assert_eq!(x_min, 0.0);
        assert_eq!(x_max, 2.0);
        assert_eq!(y_min, 1.0);
        assert_eq!(y_max, 3.0);
        assert_eq!(z_min, -1.0);
        assert_eq!(z_max, 1.0);
    }

    #[test]
    fn test_bounding_box_spherical() {
        let geom = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
        let (x_min, x_max, y_min, y_max, z_min, z_max) = geom.bounding_box();
        assert!((x_min - 0.2).abs() < 1e-10);
        assert!((x_max - 0.8).abs() < 1e-10);
        assert!((y_min - 0.2).abs() < 1e-10);
        assert!((y_max - 0.8).abs() < 1e-10);
        assert!((z_min - 0.2).abs() < 1e-10);
        assert!((z_max - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_cylindrical() {
        let geom = Geometry3D::cylindrical(0.0, 0.0, -1.0, 1.0, 0.5);
        let (x_min, x_max, y_min, y_max, z_min, z_max) = geom.bounding_box();
        assert_eq!(x_min, -0.5);
        assert_eq!(x_max, 0.5);
        assert_eq!(y_min, -0.5);
        assert_eq!(y_max, 0.5);
        assert_eq!(z_min, -1.0);
        assert_eq!(z_max, 1.0);
    }

    #[test]
    fn test_contains_rectangular() {
        let geom = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        // Inside points
        assert!(geom.contains(0.5, 0.5, 0.5));
        assert!(geom.contains(0.0, 0.0, 0.0));
        assert!(geom.contains(1.0, 1.0, 1.0));

        // Outside points
        assert!(!geom.contains(1.5, 0.5, 0.5));
        assert!(!geom.contains(0.5, 1.5, 0.5));
        assert!(!geom.contains(0.5, 0.5, 1.5));
        assert!(!geom.contains(-0.1, 0.5, 0.5));
    }

    #[test]
    fn test_contains_spherical() {
        let geom = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);

        // Center point (definitely inside)
        assert!(geom.contains(0.5, 0.5, 0.5));

        // Point on surface (r = 0.3)
        assert!(geom.contains(0.8, 0.5, 0.5));

        // Point outside (r > 0.3)
        assert!(!geom.contains(1.0, 0.5, 0.5));
        assert!(!geom.contains(0.0, 0.5, 0.5));
    }

    #[test]
    fn test_contains_cylindrical() {
        let geom = Geometry3D::cylindrical(0.0, 0.0, -1.0, 1.0, 0.5);

        // Center axis
        assert!(geom.contains(0.0, 0.0, 0.0));

        // On cylindrical surface
        assert!(geom.contains(0.5, 0.0, 0.0));
        assert!(geom.contains(0.0, 0.5, 0.0));

        // Outside radius
        assert!(!geom.contains(1.0, 0.0, 0.0));

        // Outside axial extent
        assert!(!geom.contains(0.0, 0.0, 1.5));
        assert!(!geom.contains(0.0, 0.0, -1.5));

        // Inside radius but outside axial extent
        assert!(!geom.contains(0.2, 0.2, 2.0));
    }
}
