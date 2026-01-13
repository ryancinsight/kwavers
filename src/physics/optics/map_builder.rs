// src/physics/optics/map_builder.rs
//! Optical Property Map Builder
//!
//! Provides utilities for constructing spatially heterogeneous optical property maps
//! for photoacoustic and optical imaging simulations.
//!
//! # Architecture
//!
//! - Domain layer defines canonical `OpticalPropertyData` with validation
//! - Physics layer (this module) provides spatial mapping and region-based construction
//! - Clinical layer uses these tools for realistic tissue phantoms
//!
//! # Key Features
//!
//! - Region-based property assignment (spheres, boxes, cylinders, arbitrary predicates)
//! - Layer stacking for stratified media (e.g., skin/fat/muscle)
//! - Clinical tissue presets with wavelength-specific properties
//! - Validation of physical constraints and spatial consistency
//!
//! # Example
//!
//! ```no_run
//! use kwavers::physics::optics::map_builder::{OpticalPropertyMapBuilder, Region};
//! use kwavers::domain::medium::properties::OpticalPropertyData;
//! use kwavers::domain::grid::Grid3D;
//!
//! // Create a 3D grid
//! let grid = Grid3D::new(50, 50, 50, 0.001)?;
//!
//! // Build heterogeneous optical map
//! let mut builder = OpticalPropertyMapBuilder::new(grid.dimensions());
//!
//! // Background: soft tissue
//! builder.set_background(OpticalPropertyData::soft_tissue());
//!
//! // Add tumor region (sphere at center)
//! builder.add_region(
//!     Region::sphere([0.025, 0.025, 0.025], 0.005),
//!     OpticalPropertyData::tumor()
//! );
//!
//! // Add blood vessel (cylinder)
//! builder.add_region(
//!     Region::cylinder([0.025, 0.025, 0.0], [0.025, 0.025, 0.05], 0.001),
//!     OpticalPropertyData::blood_oxygenated()
//! );
//!
//! let map = builder.build();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::domain::grid::GridDimensions;
use crate::domain::medium::properties::OpticalPropertyData;
use std::fmt;

/// Spatial region definition for property assignment
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

    /// Half-space: point on plane, normal vector (defines positive side)
    HalfSpace { point: [f64; 3], normal: [f64; 3] },

    /// Custom predicate: arbitrary spatial function
    Custom(Box<dyn Fn([f64; 3]) -> bool + Send + Sync>),
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

    /// Construct half-space region
    pub fn half_space(point: [f64; 3], normal: [f64; 3]) -> Self {
        Self::HalfSpace { point, normal }
    }

    /// Construct custom region from predicate
    pub fn custom<F>(predicate: F) -> Self
    where
        F: Fn([f64; 3]) -> bool + Send + Sync + 'static,
    {
        Self::Custom(Box::new(predicate))
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
                // Vector from start to end
                let axis = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
                let axis_len_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];

                if axis_len_sq < 1e-12 {
                    // Degenerate cylinder (point)
                    return false;
                }

                // Vector from start to point
                let to_point = [
                    point[0] - start[0],
                    point[1] - start[1],
                    point[2] - start[2],
                ];

                // Project point onto axis
                let dot = to_point[0] * axis[0] + to_point[1] * axis[1] + to_point[2] * axis[2];
                let t = dot / axis_len_sq;

                // Check if projection is within cylinder bounds
                if !(0.0..=1.0).contains(&t) {
                    return false;
                }

                // Closest point on axis
                let closest = [
                    start[0] + t * axis[0],
                    start[1] + t * axis[1],
                    start[2] + t * axis[2],
                ];

                // Distance from point to axis
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

            Self::HalfSpace {
                point: p,
                normal: n,
            } => {
                // Normalize normal vector
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len < 1e-12 {
                    return false;
                }
                let n_norm = [n[0] / len, n[1] / len, n[2] / len];

                // Vector from plane point to test point
                let vec = [point[0] - p[0], point[1] - p[1], point[2] - p[2]];

                // Dot product with normal
                let dot = vec[0] * n_norm[0] + vec[1] * n_norm[1] + vec[2] * n_norm[2];

                dot >= 0.0
            }

            Self::Custom(predicate) => predicate(point),
        }
    }
}

impl Clone for Region {
    fn clone(&self) -> Self {
        match self {
            Self::Sphere { center, radius } => Self::Sphere {
                center: *center,
                radius: *radius,
            },
            Self::Box { min, max } => Self::Box {
                min: *min,
                max: *max,
            },
            Self::Cylinder { start, end, radius } => Self::Cylinder {
                start: *start,
                end: *end,
                radius: *radius,
            },
            Self::Ellipsoid { center, semi_axes } => Self::Ellipsoid {
                center: *center,
                semi_axes: *semi_axes,
            },
            Self::HalfSpace { point, normal } => Self::HalfSpace {
                point: *point,
                normal: *normal,
            },
            Self::Custom(_) => panic!("Cannot clone Region::Custom"),
        }
    }
}

impl fmt::Debug for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sphere { center, radius } => f
                .debug_struct("Sphere")
                .field("center", center)
                .field("radius", radius)
                .finish(),
            Self::Box { min, max } => f
                .debug_struct("Box")
                .field("min", min)
                .field("max", max)
                .finish(),
            Self::Cylinder { start, end, radius } => f
                .debug_struct("Cylinder")
                .field("start", start)
                .field("end", end)
                .field("radius", radius)
                .finish(),
            Self::Ellipsoid { center, semi_axes } => f
                .debug_struct("Ellipsoid")
                .field("center", center)
                .field("semi_axes", semi_axes)
                .finish(),
            Self::HalfSpace { point, normal } => f
                .debug_struct("HalfSpace")
                .field("point", point)
                .field("normal", normal)
                .finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"<closure>").finish(),
        }
    }
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sphere { center, radius } => {
                write!(
                    f,
                    "Sphere(center=[{:.3}, {:.3}, {:.3}], r={:.3})",
                    center[0], center[1], center[2], radius
                )
            }
            Self::Box { min, max } => {
                write!(
                    f,
                    "Box(min=[{:.3}, {:.3}, {:.3}], max=[{:.3}, {:.3}, {:.3}])",
                    min[0], min[1], min[2], max[0], max[1], max[2]
                )
            }
            Self::Cylinder { start, end, radius } => {
                write!(
                    f,
                    "Cylinder(start=[{:.3}, {:.3}, {:.3}], end=[{:.3}, {:.3}, {:.3}], r={:.3})",
                    start[0], start[1], start[2], end[0], end[1], end[2], radius
                )
            }
            Self::Ellipsoid { center, semi_axes } => {
                write!(
                    f,
                    "Ellipsoid(center=[{:.3}, {:.3}, {:.3}], axes=[{:.3}, {:.3}, {:.3}])",
                    center[0], center[1], center[2], semi_axes[0], semi_axes[1], semi_axes[2]
                )
            }
            Self::HalfSpace { point, normal } => {
                write!(
                    f,
                    "HalfSpace(point=[{:.3}, {:.3}, {:.3}], normal=[{:.3}, {:.3}, {:.3}])",
                    point[0], point[1], point[2], normal[0], normal[1], normal[2]
                )
            }
            Self::Custom(_) => write!(f, "Custom(predicate)"),
        }
    }
}

/// Layer specification for stratified media
#[derive(Debug, Clone)]
pub struct Layer {
    /// Z-coordinate range: (z_min, z_max)
    pub z_range: (f64, f64),
    /// Optical properties for this layer
    pub properties: OpticalPropertyData,
}

impl Layer {
    /// Construct layer
    pub fn new(z_min: f64, z_max: f64, properties: OpticalPropertyData) -> Self {
        Self {
            z_range: (z_min, z_max),
            properties,
        }
    }

    /// Check if z-coordinate is within layer
    pub fn contains_z(&self, z: f64) -> bool {
        z >= self.z_range.0 && z < self.z_range.1
    }
}

/// Builder for heterogeneous optical property maps
///
/// Provides a fluent API for constructing spatially-varying optical property distributions.
#[derive(Debug)]
pub struct OpticalPropertyMapBuilder {
    dimensions: GridDimensions,
    background: Option<OpticalPropertyData>,
    regions: Vec<(Region, OpticalPropertyData)>,
    layers: Vec<Layer>,
}

impl OpticalPropertyMapBuilder {
    /// Create new builder for given grid dimensions
    pub fn new(dimensions: GridDimensions) -> Self {
        Self {
            dimensions,
            background: None,
            regions: Vec::new(),
            layers: Vec::new(),
        }
    }

    /// Set background (default) optical properties
    pub fn set_background(&mut self, properties: OpticalPropertyData) -> &mut Self {
        self.background = Some(properties);
        self
    }

    /// Add region with specified optical properties
    ///
    /// Regions are applied in order, later regions override earlier ones.
    pub fn add_region(&mut self, region: Region, properties: OpticalPropertyData) -> &mut Self {
        self.regions.push((region, properties));
        self
    }

    /// Add multiple regions at once
    pub fn add_regions(
        &mut self,
        regions: impl IntoIterator<Item = (Region, OpticalPropertyData)>,
    ) -> &mut Self {
        self.regions.extend(regions);
        self
    }

    /// Add stratified layer (horizontal slab)
    ///
    /// Layers are processed before regions. If layers overlap, the last-added layer wins.
    pub fn add_layer(&mut self, layer: Layer) -> &mut Self {
        self.layers.push(layer);
        self
    }

    /// Add multiple layers at once
    pub fn add_layers(&mut self, layers: impl IntoIterator<Item = Layer>) -> &mut Self {
        self.layers.extend(layers);
        self
    }

    /// Build 3D optical property map
    ///
    /// Returns flattened array indexed as `[k * (nx * ny) + j * nx + i]`
    /// where (i, j, k) are voxel indices.
    pub fn build(&self) -> OpticalPropertyMap {
        let nx = self.dimensions.nx;
        let ny = self.dimensions.ny;
        let nz = self.dimensions.nz;
        let dx = self.dimensions.dx;
        let dy = self.dimensions.dy;
        let dz = self.dimensions.dz;

        let total_voxels = nx * ny * nz;
        let mut data = Vec::with_capacity(total_voxels);

        let default_props = self
            .background
            .clone()
            .unwrap_or_else(OpticalPropertyData::soft_tissue);

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    // Voxel center coordinates
                    let x = (i as f64 + 0.5) * dx;
                    let y = (j as f64 + 0.5) * dy;
                    let z = (k as f64 + 0.5) * dz;
                    let point = [x, y, z];

                    // Start with background
                    let mut props = default_props.clone();

                    // Apply layers (stratified media)
                    for layer in &self.layers {
                        if layer.contains_z(z) {
                            props = layer.properties.clone();
                        }
                    }

                    // Apply regions (in order, last wins)
                    for (region, region_props) in &self.regions {
                        if region.contains(point) {
                            props = region_props.clone();
                        }
                    }

                    data.push(props);
                }
            }
        }

        OpticalPropertyMap {
            dimensions: self.dimensions,
            data,
        }
    }

    /// Build absorption coefficient map only (for diffusion solver)
    pub fn build_absorption_map(&self) -> Vec<f64> {
        let map = self.build();
        map.data
            .iter()
            .map(|props| props.absorption_coefficient)
            .collect()
    }

    /// Build scattering coefficient map only
    pub fn build_scattering_map(&self) -> Vec<f64> {
        let map = self.build();
        map.data
            .iter()
            .map(|props| props.scattering_coefficient)
            .collect()
    }

    /// Build reduced scattering coefficient map (μ_s' = μ_s(1-g))
    pub fn build_reduced_scattering_map(&self) -> Vec<f64> {
        let map = self.build();
        map.data
            .iter()
            .map(|props| props.reduced_scattering())
            .collect()
    }
}

/// Heterogeneous optical property map
///
/// Stores spatially-varying optical properties on a 3D grid.
#[derive(Clone, Debug)]
pub struct OpticalPropertyMap {
    pub dimensions: GridDimensions,
    pub data: Vec<OpticalPropertyData>,
}

impl OpticalPropertyMap {
    /// Get optical properties at voxel (i, j, k)
    pub fn get(&self, i: usize, j: usize, k: usize) -> Option<&OpticalPropertyData> {
        if i >= self.dimensions.nx || j >= self.dimensions.ny || k >= self.dimensions.nz {
            return None;
        }
        let idx = k * (self.dimensions.nx * self.dimensions.ny) + j * self.dimensions.nx + i;
        self.data.get(idx)
    }

    /// Get mutable reference to optical properties at voxel (i, j, k)
    pub fn get_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut OpticalPropertyData> {
        if i >= self.dimensions.nx || j >= self.dimensions.ny || k >= self.dimensions.nz {
            return None;
        }
        let idx = k * (self.dimensions.nx * self.dimensions.ny) + j * self.dimensions.nx + i;
        self.data.get_mut(idx)
    }

    /// Extract absorption coefficient map
    pub fn absorption_map(&self) -> Vec<f64> {
        self.data
            .iter()
            .map(|props| props.absorption_coefficient)
            .collect()
    }

    /// Extract scattering coefficient map
    pub fn scattering_map(&self) -> Vec<f64> {
        self.data
            .iter()
            .map(|props| props.scattering_coefficient)
            .collect()
    }

    /// Extract reduced scattering coefficient map
    pub fn reduced_scattering_map(&self) -> Vec<f64> {
        self.data
            .iter()
            .map(|props| props.reduced_scattering())
            .collect()
    }

    /// Extract refractive index map
    pub fn refractive_index_map(&self) -> Vec<f64> {
        self.data
            .iter()
            .map(|props| props.refractive_index)
            .collect()
    }

    /// Compute total volume (m³)
    pub fn volume(&self) -> f64 {
        self.dimensions.dx
            * self.dimensions.dy
            * self.dimensions.dz
            * (self.dimensions.nx * self.dimensions.ny * self.dimensions.nz) as f64
    }

    /// Compute statistics for absorption coefficient
    pub fn absorption_stats(&self) -> PropertyStats {
        compute_stats(self.data.iter().map(|p| p.absorption_coefficient))
    }

    /// Compute statistics for scattering coefficient
    pub fn scattering_stats(&self) -> PropertyStats {
        compute_stats(self.data.iter().map(|p| p.scattering_coefficient))
    }
}

impl fmt::Display for OpticalPropertyMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let abs_stats = self.absorption_stats();
        let sca_stats = self.scattering_stats();
        write!(
            f,
            "OpticalPropertyMap({}×{}×{}, μ_a: {:.2}±{:.2} m⁻¹, μ_s: {:.1}±{:.1} m⁻¹)",
            self.dimensions.nx,
            self.dimensions.ny,
            self.dimensions.nz,
            abs_stats.mean,
            abs_stats.std_dev,
            sca_stats.mean,
            sca_stats.std_dev
        )
    }
}

/// Statistical summary of property distribution
#[derive(Debug, Clone)]
pub struct PropertyStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

fn compute_stats(values: impl Iterator<Item = f64>) -> PropertyStats {
    let values: Vec<f64> = values.collect();
    let n = values.len() as f64;

    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    PropertyStats {
        mean,
        std_dev,
        min,
        max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_region() {
        let region = Region::sphere([0.0, 0.0, 0.0], 1.0);
        assert!(region.contains([0.0, 0.0, 0.0]));
        assert!(region.contains([0.5, 0.5, 0.0]));
        assert!(!region.contains([1.0, 1.0, 0.0]));
    }

    #[test]
    fn test_box_region() {
        let region = Region::box_region([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(region.contains([0.5, 0.5, 0.5]));
        assert!(region.contains([0.0, 0.0, 0.0]));
        assert!(region.contains([1.0, 1.0, 1.0]));
        assert!(!region.contains([1.1, 0.5, 0.5]));
    }

    #[test]
    fn test_cylinder_region() {
        let region = Region::cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.5);
        assert!(region.contains([0.0, 0.0, 0.5]));
        assert!(region.contains([0.3, 0.0, 0.5]));
        assert!(!region.contains([0.6, 0.0, 0.5]));
        assert!(!region.contains([0.0, 0.0, 1.5]));
    }

    #[test]
    fn test_ellipsoid_region() {
        let region = Region::ellipsoid([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        assert!(region.contains([0.0, 0.0, 0.0]));
        assert!(region.contains([1.0, 0.0, 0.0]));
        assert!(!region.contains([2.5, 0.0, 0.0]));
    }

    #[test]
    fn test_half_space_region() {
        let region = Region::half_space([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(region.contains([0.0, 0.0, 1.0]));
        assert!(!region.contains([0.0, 0.0, -1.0]));
    }

    #[test]
    fn test_custom_region() {
        let region = Region::custom(|p| p[0] + p[1] + p[2] > 1.0);
        assert!(region.contains([1.0, 1.0, 1.0]));
        assert!(!region.contains([0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_map_builder_background() {
        let dims = GridDimensions::new(10, 10, 10, 0.001, 0.001, 0.001);
        let mut builder = OpticalPropertyMapBuilder::new(dims);
        builder.set_background(OpticalPropertyData::water());

        let map = builder.build();
        assert_eq!(map.data.len(), 1000);

        let props = map.get(5, 5, 5).unwrap();
        assert!((props.absorption_coefficient - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_map_builder_with_sphere() {
        let dims = GridDimensions::new(20, 20, 20, 0.001, 0.001, 0.001);
        let mut builder = OpticalPropertyMapBuilder::new(dims);

        builder.set_background(OpticalPropertyData::water());
        builder.add_region(
            Region::sphere([0.01, 0.01, 0.01], 0.005),
            OpticalPropertyData::tumor(),
        );

        let map = builder.build();

        // Center should be tumor
        let center_props = map.get(10, 10, 10).unwrap();
        assert!((center_props.absorption_coefficient - 10.0).abs() < 1e-6);

        // Corner should be water
        let corner_props = map.get(0, 0, 0).unwrap();
        assert!((corner_props.absorption_coefficient - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_map_builder_with_layers() {
        let dims = GridDimensions::new(10, 10, 20, 0.001, 0.001, 0.001);
        let mut builder = OpticalPropertyMapBuilder::new(dims);

        builder.set_background(OpticalPropertyData::water());
        builder.add_layer(Layer::new(
            0.0,
            0.005,
            OpticalPropertyData::skin_epidermis(),
        ));
        builder.add_layer(Layer::new(0.005, 0.01, OpticalPropertyData::skin_dermis()));
        builder.add_layer(Layer::new(0.01, 0.02, OpticalPropertyData::fat()));

        let map = builder.build();

        // Check layering
        let epidermis = map.get(5, 5, 2).unwrap();
        assert!((epidermis.absorption_coefficient - 5.0).abs() < 1e-6);

        let dermis = map.get(5, 5, 7).unwrap();
        assert!((dermis.absorption_coefficient - 1.0).abs() < 1e-6);

        let fat = map.get(5, 5, 12).unwrap();
        assert!((fat.absorption_coefficient - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_map_absorption_extraction() {
        let dims = GridDimensions::new(5, 5, 5, 0.001, 0.001, 0.001);
        let mut builder = OpticalPropertyMapBuilder::new(dims);
        builder.set_background(OpticalPropertyData::soft_tissue());

        let map = builder.build();
        let abs_map = map.absorption_map();

        assert_eq!(abs_map.len(), 125);
        assert!(abs_map.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_property_stats() {
        let dims = GridDimensions::new(10, 10, 10, 0.001, 0.001, 0.001);
        let mut builder = OpticalPropertyMapBuilder::new(dims);

        builder.set_background(OpticalPropertyData::soft_tissue());
        builder.add_region(
            Region::sphere([0.005, 0.005, 0.005], 0.003),
            OpticalPropertyData::blood_oxygenated(),
        );

        let map = builder.build();
        let stats = map.absorption_stats();

        assert!(stats.min >= 0.0);
        assert!(stats.max <= 100.0);
        assert!(stats.mean > stats.min);
        assert!(stats.mean < stats.max);
    }
}
