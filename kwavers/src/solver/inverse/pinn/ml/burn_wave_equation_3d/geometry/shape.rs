//! `Geometry3D` — spatial domain primitives for 3D PINN wave problems.

use super::super::types::InterfaceCondition3D;

/// 3D Geometry definitions for PINN domains.
///
/// Defines the spatial domain Ω ⊂ ℝ³ for wave equation simulation.
/// Supports simple and complex geometries for heterogeneous media.
///
/// ## Supported variants
///
/// - **Rectangular**: Axis-aligned box.  Ω = {(x,y,z) : x_min ≤ x ≤ x_max, …}
/// - **Spherical**: Ball. Ω = {(x,y,z) : ‖r-rc‖₂ ≤ radius}
/// - **Cylindrical**: Cylinder with z-parallel axis. (x-xc)² + (y-yc)² ≤ r², z_min ≤ z ≤ z_max
/// - **MultiRegion**: Union of sub-geometries with interface conditions.
#[derive(Debug, Clone)]
pub enum Geometry3D {
    /// Rectangular box domain.
    Rectangular {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_min: f64,
        z_max: f64,
    },

    /// Spherical domain.
    Spherical {
        x_center: f64,
        y_center: f64,
        z_center: f64,
        radius: f64,
    },

    /// Cylindrical domain (axis parallel to z).
    Cylindrical {
        x_center: f64,
        y_center: f64,
        z_min: f64,
        z_max: f64,
        radius: f64,
    },

    /// Multi-region domain for heterogeneous media.
    MultiRegion {
        regions: Vec<(Geometry3D, usize)>,
        interfaces: Vec<InterfaceCondition3D>,
    },
}

impl Geometry3D {
    /// Create a rectangular geometry.
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

    /// Create a spherical geometry.
    pub fn spherical(x_center: f64, y_center: f64, z_center: f64, radius: f64) -> Self {
        Self::Spherical {
            x_center,
            y_center,
            z_center,
            radius,
        }
    }

    /// Create a cylindrical geometry (axis parallel to z).
    pub fn cylindrical(x_center: f64, y_center: f64, z_min: f64, z_max: f64, radius: f64) -> Self {
        Self::Cylindrical {
            x_center,
            y_center,
            z_min,
            z_max,
            radius,
        }
    }

    /// Return the axis-aligned bounding box `(x_min, x_max, y_min, y_max, z_min, z_max)`.
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

    /// Return a canonical interior point.
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

    /// Return `true` if the point `(x, y, z)` lies inside the geometry.
    ///
    /// - **Rectangular**: x_min ≤ x ≤ x_max ∧ y_min ≤ y ≤ y_max ∧ z_min ≤ z ≤ z_max
    /// - **Spherical**: ‖(x,y,z) - center‖₂ ≤ radius
    /// - **Cylindrical**: (x-xc)² + (y-yc)² ≤ r² ∧ z_min ≤ z ≤ z_max
    /// - **MultiRegion**: point in any sub-region
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
