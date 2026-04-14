use crate::domain::grid::Grid;

#[derive(Debug, Clone)]
pub struct VolumetricROI {
    /// ROI center coordinates [x, y, z] in meters
    pub center: [f64; 3],
    /// ROI dimensions [width, height, depth] in meters
    pub size: [f64; 3],
    /// ROI orientation angles [yaw, pitch, roll] in radians
    pub orientation: [f64; 3],
    /// Quality threshold for inclusion (0-1)
    pub quality_threshold: f64,
    /// Minimum depth for analysis (m)
    pub min_depth: f64,
    /// Maximum depth for analysis (m)
    pub max_depth: f64,
}

impl Default for VolumetricROI {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.04],     // 4cm depth (typical liver)
            size: [0.06, 0.06, 0.04],     // 6x6x4cm volume
            orientation: [0.0, 0.0, 0.0], // Axial orientation
            quality_threshold: 0.7,
            min_depth: 0.02, // 2cm minimum
            max_depth: 0.08, // 8cm maximum
        }
    }
}

impl VolumetricROI {
    /// Create ROI for liver fibrosis assessment
    pub fn liver_roi(center: [f64; 3]) -> Self {
        Self {
            center,
            size: [0.08, 0.08, 0.06], // 8x8x6cm for liver
            quality_threshold: 0.8,
            min_depth: 0.02,
            max_depth: 0.10,
            ..Default::default()
        }
    }

    /// Create ROI for breast lesion assessment
    pub fn breast_roi(center: [f64; 3]) -> Self {
        Self {
            center,
            size: [0.04, 0.04, 0.03], // 4x4x3cm for breast
            quality_threshold: 0.85,
            min_depth: 0.01,
            max_depth: 0.05,
            ..Default::default()
        }
    }

    /// Create ROI for prostate assessment
    pub fn prostate_roi(center: [f64; 3]) -> Self {
        Self {
            center,
            size: [0.05, 0.05, 0.04], // 5x5x4cm for prostate
            quality_threshold: 0.75,
            min_depth: 0.03,
            max_depth: 0.08,
            ..Default::default()
        }
    }

    /// Check if a point is within the ROI
    pub fn contains_point(&self, point: [f64; 3]) -> bool {
        // Simple axis-aligned bounding box check
        // In practice, this should account for orientation
        let dx = (point[0] - self.center[0]).abs();
        let dy = (point[1] - self.center[1]).abs();
        let dz = (point[2] - self.center[2]).abs();

        dx <= self.size[0] / 2.0
            && dy <= self.size[1] / 2.0
            && dz <= self.size[2] / 2.0
            && point[2] >= self.min_depth
            && point[2] <= self.max_depth
    }

    /// Get ROI bounds as grid indices
    pub fn grid_bounds(&self, grid: &Grid) -> ([usize; 3], [usize; 3]) {
        let min_x = ((self.center[0] - self.size[0] / 2.0) / grid.dx).max(0.0) as usize;
        let max_x =
            ((self.center[0] + self.size[0] / 2.0) / grid.dx).min(grid.nx as f64 - 1.0) as usize;
        let min_y = ((self.center[1] - self.size[1] / 2.0) / grid.dy).max(0.0) as usize;
        let max_y =
            ((self.center[1] + self.size[1] / 2.0) / grid.dy).min(grid.ny as f64 - 1.0) as usize;
        let min_z = ((self.center[2] - self.size[2] / 2.0) / grid.dz).max(0.0) as usize;
        let max_z =
            ((self.center[2] + self.size[2] / 2.0) / grid.dz).min(grid.nz as f64 - 1.0) as usize;

        ([min_x, min_y, min_z], [max_x, max_y, max_z])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_volumetric_roi_creation() {
        let roi = VolumetricROI::liver_roi([0.0, 0.0, 0.04]);
        assert_eq!(roi.center, [0.0, 0.0, 0.04]);
        assert_eq!(roi.size, [0.08, 0.08, 0.06]);
        assert_eq!(roi.quality_threshold, 0.8);
    }

    #[test]
    fn test_roi_contains_point() {
        let roi = VolumetricROI {
            center: [0.0, 0.0, 0.04],
            size: [0.06, 0.06, 0.04],
            quality_threshold: 0.7,
            min_depth: 0.02,
            max_depth: 0.06,
            orientation: [0.0, 0.0, 0.0],
        };

        // Point inside ROI
        assert!(roi.contains_point([0.01, 0.01, 0.04]));
        // Point outside ROI (too deep)
        assert!(!roi.contains_point([0.01, 0.01, 0.07]));
        // Point outside ROI (too shallow)
        assert!(!roi.contains_point([0.01, 0.01, 0.01]));
    }
}
