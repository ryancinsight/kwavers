//! Optical Property Map
//!
//! Domain types for spatially heterogeneous optical property maps used in
//! photoacoustic and optical imaging simulations.
//!
//! These are pure data models and spatial primitives, suitable for use across
//! all layers according to clean architecture principles.

use super::properties::OpticalPropertyData;
use crate::domain::grid::GridDimensions;
use ndarray::Array3;

/// 3D optical property map
///
/// Spatially-resolved optical properties on a regular grid.
#[derive(Debug, Clone)]
pub struct OpticalPropertyMap {
    /// Absorption coefficient map (m⁻¹)
    pub mu_a: Array3<f64>,

    /// Reduced scattering coefficient map (m⁻¹)
    pub mu_s_prime: Array3<f64>,

    /// Refractive index map
    pub refractive_index: Array3<f64>,

    /// Grid dimensions
    pub dimensions: GridDimensions,
}

impl OpticalPropertyMap {
    /// Create a new optical property map
    pub fn new(
        mu_a: Array3<f64>,
        mu_s_prime: Array3<f64>,
        refractive_index: Array3<f64>,
        dimensions: GridDimensions,
    ) -> Self {
        Self {
            mu_a,
            mu_s_prime,
            refractive_index,
            dimensions,
        }
    }

    /// Create a homogeneous map with constant properties
    pub fn homogeneous(props: &OpticalPropertyData, dimensions: GridDimensions) -> Self {
        let shape = (dimensions.nx, dimensions.ny, dimensions.nz);
        Self {
            mu_a: Array3::from_elem(shape, props.absorption_coefficient),
            mu_s_prime: Array3::from_elem(shape, props.reduced_scattering_coefficient()),
            refractive_index: Array3::from_elem(shape, props.refractive_index),
            dimensions,
        }
    }

    /// Get properties at a specific grid point
    pub fn get_properties(&self, i: usize, j: usize, k: usize) -> Option<OpticalPropertyData> {
        if i >= self.dimensions.nx || j >= self.dimensions.ny || k >= self.dimensions.nz {
            return None;
        }

        Some(OpticalPropertyData {
            absorption_coefficient: self.mu_a[[i, j, k]],
            scattering_coefficient: self.mu_s_prime[[i, j, k]] / (1.0 - 0.9), // Reconstruct from reduced
            anisotropy: 0.9,                                                  // Default value
            refractive_index: self.refractive_index[[i, j, k]],
        })
    }

    /// Get optical properties at grid coordinates (i, j, k) with bounds checking
    /// Alias for get_properties for backward compatibility
    pub fn get(&self, i: usize, j: usize, k: usize) -> Option<OpticalPropertyData> {
        self.get_properties(i, j, k)
    }

    /// Calculate the physical volume of the domain in cubic meters
    pub fn volume(&self) -> f64 {
        let GridDimensions {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
        } = self.dimensions;
        (nx as f64) * dx * (ny as f64) * dy * (nz as f64) * dz
    }
}

/// Spatial region definition for property assignment
#[derive(Debug, Clone)]
pub enum Region {
    /// Sphere: center (x, y, z), radius
    Sphere { center: [f64; 3], radius: f64 },

    /// Box: min corner (x, y, z), max corner (x, y, z)
    Box { min: [f64; 3], max: [f64; 3] },

    /// Cylinder: start point (x, y, z), end point (x, y, z), radius
    Cylinder {
        start: [f64; 3],
        end: [f64; 3],
        radius: f64,
    },

    /// Ellipsoid: center (x, y, z), semi-axes (a, b, c)
    Ellipsoid {
        center: [f64; 3],
        semi_axes: [f64; 3],
    },
}

impl Region {
    /// Construct sphere region
    pub fn sphere(center: [f64; 3], radius: f64) -> Self {
        Self::Sphere { center, radius }
    }

    /// Construct box region
    pub fn box_region(min: [f64; 3], max: [f64; 3]) -> Self {
        Self::Box { min, max }
    }

    /// Construct cylinder region
    pub fn cylinder(start: [f64; 3], end: [f64; 3], radius: f64) -> Self {
        Self::Cylinder { start, end, radius }
    }

    /// Construct ellipsoid region
    pub fn ellipsoid(center: [f64; 3], semi_axes: [f64; 3]) -> Self {
        Self::Ellipsoid { center, semi_axes }
    }

    /// Test if point is inside region
    pub fn contains(&self, point: [f64; 3]) -> bool {
        match self {
            Self::Sphere { center, radius } => {
                let dx = point[0] - center[0];
                let dy = point[1] - center[1];
                let dz = point[2] - center[2];
                dx * dx + dy * dy + dz * dz <= radius * radius
            }

            Self::Box { min, max } => {
                point[0] >= min[0]
                    && point[0] <= max[0]
                    && point[1] >= min[1]
                    && point[1] <= max[1]
                    && point[2] >= min[2]
                    && point[2] <= max[2]
            }

            Self::Cylinder { start, end, radius } => {
                let axis = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                let axis_len_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];

                if axis_len_sq < 1e-12 {
                    return false;
                }

                let to_point = [
                    point[0] - start[0],
                    point[1] - start[1],
                    point[2] - start[2],
                ];

                let dot = to_point[0] * axis[0] + to_point[1] * axis[1] + to_point[2] * axis[2];
                let t = dot / axis_len_sq;

                if !(0.0..=1.0).contains(&t) {
                    return false;
                }

                let closest = [
                    start[0] + t * axis[0],
                    start[1] + t * axis[1],
                    start[2] + t * axis[2],
                ];

                let dx = point[0] - closest[0];
                let dy = point[1] - closest[1];
                let dz = point[2] - closest[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                dist_sq <= radius * radius
            }

            Self::Ellipsoid { center, semi_axes } => {
                let dx = (point[0] - center[0]) / semi_axes[0];
                let dy = (point[1] - center[1]) / semi_axes[1];
                let dz = (point[2] - center[2]) / semi_axes[2];
                dx * dx + dy * dy + dz * dz <= 1.0
            }
        }
    }
}

/// Horizontal layer for stratified media
#[derive(Debug, Clone)]
pub struct Layer {
    /// Z-coordinate range [z_min, z_max]
    pub z_min: f64,
    pub z_max: f64,

    /// Optical properties for this layer
    pub properties: OpticalPropertyData,
}

impl Layer {
    /// Create a new layer
    pub fn new(z_min: f64, z_max: f64, properties: OpticalPropertyData) -> Self {
        Self {
            z_min,
            z_max,
            properties,
        }
    }
}

/// Builder for constructing optical property maps
#[derive(Debug)]
pub struct OpticalPropertyMapBuilder {
    dimensions: GridDimensions,
    mu_a: Array3<f64>,
    mu_s_prime: Array3<f64>,
    refractive_index: Array3<f64>,
}

impl OpticalPropertyMapBuilder {
    /// Create a new builder with given dimensions
    pub fn new(dimensions: GridDimensions) -> Self {
        let shape = (dimensions.nx, dimensions.ny, dimensions.nz);
        Self {
            dimensions,
            mu_a: Array3::zeros(shape),
            mu_s_prime: Array3::zeros(shape),
            refractive_index: Array3::from_elem(shape, 1.0), // Default: vacuum/air
        }
    }

    /// Set background properties
    pub fn set_background(&mut self, props: OpticalPropertyData) {
        self.mu_a.fill(props.absorption_coefficient);
        self.mu_s_prime.fill(props.reduced_scattering_coefficient());
        self.refractive_index.fill(props.refractive_index);
    }

    /// Add a region with specific properties
    pub fn add_region(&mut self, region: Region, props: OpticalPropertyData) {
        let GridDimensions {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
        } = self.dimensions;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * dx;
                    let y = j as f64 * dy;
                    let z = k as f64 * dz;

                    if region.contains([x, y, z]) {
                        self.mu_a[[i, j, k]] = props.absorption_coefficient;
                        self.mu_s_prime[[i, j, k]] = props.reduced_scattering_coefficient();
                        self.refractive_index[[i, j, k]] = props.refractive_index;
                    }
                }
            }
        }
    }

    /// Add a horizontal layer
    pub fn add_layer(&mut self, layer: Layer) {
        let GridDimensions { nx, ny, nz, dz, .. } = self.dimensions;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let z = k as f64 * dz;

                    if z >= layer.z_min && z <= layer.z_max {
                        self.mu_a[[i, j, k]] = layer.properties.absorption_coefficient;
                        self.mu_s_prime[[i, j, k]] =
                            layer.properties.reduced_scattering_coefficient();
                        self.refractive_index[[i, j, k]] = layer.properties.refractive_index;
                    }
                }
            }
        }
    }

    /// Build the final optical property map
    pub fn build(self) -> OpticalPropertyMap {
        OpticalPropertyMap {
            mu_a: self.mu_a,
            mu_s_prime: self.mu_s_prime,
            refractive_index: self.refractive_index,
            dimensions: self.dimensions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_sphere() {
        let region = Region::sphere([0.0, 0.0, 0.0], 1.0);
        assert!(region.contains([0.0, 0.0, 0.0]));
        assert!(region.contains([0.5, 0.0, 0.0]));
        assert!(!region.contains([2.0, 0.0, 0.0]));
    }

    #[test]
    fn test_region_box() {
        let region = Region::box_region([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(region.contains([0.5, 0.5, 0.5]));
        assert!(!region.contains([1.5, 0.5, 0.5]));
    }

    #[test]
    fn test_optical_map_builder() {
        let dims = GridDimensions {
            nx: 10,
            ny: 10,
            nz: 10,
            dx: 0.001,
            dy: 0.001,
            dz: 0.001,
        };

        let mut builder = OpticalPropertyMapBuilder::new(dims);
        builder.set_background(OpticalPropertyData::soft_tissue());

        let map = builder.build();
        assert_eq!(map.mu_a.shape(), &[10, 10, 10]);
    }
}
