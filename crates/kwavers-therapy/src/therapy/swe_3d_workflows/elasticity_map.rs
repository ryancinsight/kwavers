use super::roi::VolumetricROI;
use super::statistics::VolumetricStatistics;
use kwavers_domain::grid::Grid;
use ndarray::{Array3, Axis};

#[derive(Debug, Clone)]
pub struct ElasticityMap3D {
    /// Young's modulus values (Pa) - shape: [nx, ny, nz]
    pub young_modulus: Array3<f64>,
    /// Shear wave speed (m/s) - shape: [nx, ny, nz]
    pub shear_speed: Array3<f64>,
    /// Confidence map (0-1) - shape: [nx, ny, nz]
    pub confidence: Array3<f64>,
    /// Quality map (0-1) - shape: [nx, ny, nz]
    pub quality: Array3<f64>,
    /// Reliability mask (true = reliable) - shape: [nx, ny, nz]
    pub reliability_mask: Array3<bool>,
    /// Computation grid
    pub grid: Grid,
}

impl ElasticityMap3D {
    /// Create new 3D elasticity map
    pub fn new(grid: &Grid) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        Self {
            young_modulus: Array3::zeros((nx, ny, nz)),
            shear_speed: Array3::zeros((nx, ny, nz)),
            confidence: Array3::zeros((nx, ny, nz)),
            quality: Array3::zeros((nx, ny, nz)),
            reliability_mask: Array3::from_elem((nx, ny, nz), false),
            grid: grid.clone(),
        }
    }

    /// Extract 2D slice at given z-index
    pub fn axial_slice(&self, z_index: usize) -> ElasticityMap2D {
        ElasticityMap2D {
            young_modulus: self.young_modulus.index_axis(Axis(2), z_index).to_owned(),
            shear_speed: self.shear_speed.index_axis(Axis(2), z_index).to_owned(),
            confidence: self.confidence.index_axis(Axis(2), z_index).to_owned(),
            quality: self.quality.index_axis(Axis(2), z_index).to_owned(),
            reliability_mask: self
                .reliability_mask
                .index_axis(Axis(2), z_index)
                .to_owned(),
        }
    }

    /// Extract sagittal slice (YZ plane) at given x-index
    pub fn sagittal_slice(&self, x_index: usize) -> ElasticityMap2D {
        ElasticityMap2D {
            young_modulus: self.young_modulus.index_axis(Axis(0), x_index).to_owned(),
            shear_speed: self.shear_speed.index_axis(Axis(0), x_index).to_owned(),
            confidence: self.confidence.index_axis(Axis(0), x_index).to_owned(),
            quality: self.quality.index_axis(Axis(0), x_index).to_owned(),
            reliability_mask: self
                .reliability_mask
                .index_axis(Axis(0), x_index)
                .to_owned(),
        }
    }

    /// Extract coronal slice (XZ plane) at given y-index
    pub fn coronal_slice(&self, y_index: usize) -> ElasticityMap2D {
        ElasticityMap2D {
            young_modulus: self.young_modulus.index_axis(Axis(1), y_index).to_owned(),
            shear_speed: self.shear_speed.index_axis(Axis(1), y_index).to_owned(),
            confidence: self.confidence.index_axis(Axis(1), y_index).to_owned(),
            quality: self.quality.index_axis(Axis(1), y_index).to_owned(),
            reliability_mask: self
                .reliability_mask
                .index_axis(Axis(1), y_index)
                .to_owned(),
        }
    }

    /// Compute volumetric statistics within ROI
    pub fn volumetric_statistics(&self, roi: &VolumetricROI) -> VolumetricStatistics {
        let ([min_x, min_y, min_z], [max_x, max_y, max_z]) = roi.grid_bounds(&self.grid);

        let mut valid_points = 0;
        let mut sum_modulus = 0.0;
        let mut sum_speed = 0.0;
        let mut min_modulus = f64::INFINITY;
        let mut max_modulus = f64::NEG_INFINITY;
        let mut mean_confidence = 0.0;
        let mut mean_quality = 0.0;
        let mut modulus_values: Vec<f64> = Vec::new();

        for k in min_z..=max_z {
            for j in min_y..=max_y {
                for i in min_x..=max_x {
                    if self.reliability_mask[[i, j, k]]
                        && self.confidence[[i, j, k]] >= roi.quality_threshold
                    {
                        let modulus = self.young_modulus[[i, j, k]];
                        let speed = self.shear_speed[[i, j, k]];

                        valid_points += 1;
                        sum_modulus += modulus;
                        sum_speed += speed;
                        min_modulus = min_modulus.min(modulus);
                        max_modulus = max_modulus.max(modulus);
                        mean_confidence += self.confidence[[i, j, k]];
                        mean_quality += self.quality[[i, j, k]];
                        modulus_values.push(modulus);
                    }
                }
            }
        }

        if valid_points == 0 {
            return VolumetricStatistics {
                valid_voxels: 0,
                mean_modulus: 0.0,
                std_modulus: 0.0,
                median_modulus: 0.0,
                min_modulus: 0.0,
                max_modulus: 0.0,
                mean_speed: 0.0,
                mean_confidence: 0.0,
                mean_quality: 0.0,
                volume_coverage: 0.0,
            };
        }

        let mean_modulus = sum_modulus / valid_points as f64;
        let mean_speed = sum_speed / valid_points as f64;
        mean_confidence /= valid_points as f64;
        mean_quality /= valid_points as f64;

        // Compute standard deviation
        let mut sum_squared_diff = 0.0;
        for k in min_z..=max_z {
            for j in min_y..=max_y {
                for i in min_x..=max_x {
                    if self.reliability_mask[[i, j, k]]
                        && self.confidence[[i, j, k]] >= roi.quality_threshold
                    {
                        let diff = self.young_modulus[[i, j, k]] - mean_modulus;
                        sum_squared_diff += diff * diff;
                    }
                }
            }
        }
        let std_modulus = (sum_squared_diff / valid_points as f64).sqrt();

        // Compute true median from collected modulus values
        modulus_values.sort_unstable_by(|a, b| a.total_cmp(b));
        let median_modulus = if modulus_values.len() % 2 == 1 {
            modulus_values[modulus_values.len() / 2]
        } else {
            let mid = modulus_values.len() / 2;
            (modulus_values[mid - 1] + modulus_values[mid]) / 2.0
        };

        let total_roi_voxels = (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1);
        let volume_coverage = valid_points as f64 / total_roi_voxels as f64;

        VolumetricStatistics {
            valid_voxels: valid_points,
            mean_modulus,
            std_modulus,
            median_modulus,
            min_modulus,
            max_modulus,
            mean_speed,
            mean_confidence,
            mean_quality,
            volume_coverage,
        }
    }
}

/// 2D elasticity map for slice visualization
#[derive(Debug, Clone)]
pub struct ElasticityMap2D {
    /// Young's modulus values (Pa)
    pub young_modulus: Array2<f64>,
    /// Shear wave speed (m/s)
    pub shear_speed: Array2<f64>,
    /// Confidence map (0-1)
    pub confidence: Array2<f64>,
    /// Quality map (0-1)
    pub quality: Array2<f64>,
    /// Reliability mask
    pub reliability_mask: Array2<bool>,
}

type Array2<T> = ndarray::Array2<T>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_elasticity_map_3d_creation() {
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
        let map = ElasticityMap3D::new(&grid);

        assert_eq!(map.young_modulus.dim(), (20, 20, 20));
        assert_eq!(map.shear_speed.dim(), (20, 20, 20));
        assert_eq!(map.confidence.dim(), (20, 20, 20));
    }

    #[test]
    fn test_slice_extraction() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let mut map = ElasticityMap3D::new(&grid);

        // Set some test values
        map.young_modulus[[5, 5, 5]] = 10000.0; // 10 kPa
        map.confidence[[5, 5, 5]] = 0.9;

        // Extract axial slice
        let axial_slice = map.axial_slice(5);
        assert_eq!(axial_slice.young_modulus[[5, 5]], 10000.0);
        assert_eq!(axial_slice.confidence[[5, 5]], 0.9);

        // Extract sagittal slice
        let sagittal_slice = map.sagittal_slice(5);
        assert_eq!(sagittal_slice.young_modulus[[5, 5]], 10000.0);

        // Extract coronal slice
        let coronal_slice = map.coronal_slice(5);
        assert_eq!(coronal_slice.young_modulus[[5, 5]], 10000.0);
    }

    #[test]
    fn test_volumetric_statistics() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let mut map = ElasticityMap3D::new(&grid);

        // Set up test data in ROI
        let roi = VolumetricROI {
            center: [0.0045, 0.0045, 0.0045], // Center of 10x10x10 grid
            size: [0.009, 0.009, 0.009],      // Most of the volume
            quality_threshold: 0.5,
            min_depth: 0.0,
            max_depth: 0.01,
            orientation: [0.0, 0.0, 0.0],
        };

        // Set reliable data
        for k in 2..8 {
            for j in 2..8 {
                for i in 2..8 {
                    map.young_modulus[[i, j, k]] = 5000.0; // 5 kPa
                    map.shear_speed[[i, j, k]] = 2.0; // 2 m/s
                    map.confidence[[i, j, k]] = 0.8;
                    map.quality[[i, j, k]] = 0.9;
                    map.reliability_mask[[i, j, k]] = true;
                }
            }
        }

        let stats = map.volumetric_statistics(&roi);

        assert!(stats.valid_voxels > 0);
        assert!((stats.mean_modulus - 5000.0).abs() < 1.0);
        assert!((stats.mean_speed - 2.0).abs() < 0.1);
        assert!(stats.mean_confidence > 0.7);
        assert!(stats.volume_coverage > 0.0);
    }
}
