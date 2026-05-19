use super::elasticity_map::{ElasticityMap2D, ElasticityMap3D};

#[derive(Debug)]
pub struct MultiPlanarReconstruction {
    /// Axial slices (XY planes)
    pub axial_slices: Vec<ElasticityMap2D>,
    /// Sagittal slices (YZ planes)
    pub sagittal_slices: Vec<ElasticityMap2D>,
    /// Coronal slices (XZ planes)
    pub coronal_slices: Vec<ElasticityMap2D>,
    /// Slice positions for each plane
    pub slice_positions: SlicePositions,
}

impl MultiPlanarReconstruction {
    /// Create MPR from 3D elasticity map
    pub fn from_elasticity_map(map: &ElasticityMap3D, slice_spacing: f64) -> Self {
        let mut axial_slices = Vec::new();
        let mut sagittal_slices = Vec::new();
        let mut coronal_slices = Vec::new();

        let (nx, ny, nz) = map.grid.dimensions();
        let axial_step = ((slice_spacing / map.grid.dz) as usize).max(1);
        let sagittal_step = ((slice_spacing / map.grid.dx) as usize).max(1);
        let coronal_step = ((slice_spacing / map.grid.dy) as usize).max(1);

        let axial_positions: Vec<f64> = (0..nz)
            .step_by(axial_step)
            .map(|k| k as f64 * map.grid.dz)
            .collect();
        for &z_pos in &axial_positions {
            let z_index = (z_pos / map.grid.dz) as usize;
            if z_index < nz {
                axial_slices.push(map.axial_slice(z_index));
            }
        }

        let sagittal_positions: Vec<f64> = (0..nx)
            .step_by(sagittal_step)
            .map(|i| i as f64 * map.grid.dx)
            .collect();
        for &x_pos in &sagittal_positions {
            let x_index = (x_pos / map.grid.dx) as usize;
            if x_index < nx {
                sagittal_slices.push(map.sagittal_slice(x_index));
            }
        }

        let coronal_positions: Vec<f64> = (0..ny)
            .step_by(coronal_step)
            .map(|j| j as f64 * map.grid.dy)
            .collect();
        for &y_pos in &coronal_positions {
            let y_index = (y_pos / map.grid.dy) as usize;
            if y_index < ny {
                coronal_slices.push(map.coronal_slice(y_index));
            }
        }

        Self {
            axial_slices,
            sagittal_slices,
            coronal_slices,
            slice_positions: SlicePositions {
                axial: axial_positions,
                sagittal: sagittal_positions,
                coronal: coronal_positions,
            },
        }
    }

    /// Get slice at specific position and orientation.
    #[must_use]
    pub fn get_slice(
        &self,
        position: f64,
        orientation: SliceOrientation,
    ) -> Option<&ElasticityMap2D> {
        match orientation {
            SliceOrientation::Axial => {
                let index = self
                    .slice_positions
                    .axial
                    .iter()
                    .position(|&p| (p - position).abs() < 1e-6)?;
                self.axial_slices.get(index)
            }
            SliceOrientation::Sagittal => {
                let index = self
                    .slice_positions
                    .sagittal
                    .iter()
                    .position(|&p| (p - position).abs() < 1e-6)?;
                self.sagittal_slices.get(index)
            }
            SliceOrientation::Coronal => {
                let index = self
                    .slice_positions
                    .coronal
                    .iter()
                    .position(|&p| (p - position).abs() < 1e-6)?;
                self.coronal_slices.get(index)
            }
        }
    }
}

/// Slice orientation for multi-planar reconstruction.
#[derive(Debug, Clone, Copy)]
pub enum SliceOrientation {
    /// Axial (XY plane)
    Axial,
    /// Sagittal (YZ plane)
    Sagittal,
    /// Coronal (XZ plane)
    Coronal,
}

/// Slice positions for each orientation.
#[derive(Debug, Clone)]
pub struct SlicePositions {
    /// Axial slice positions (z-coordinates)
    pub axial: Vec<f64>,
    /// Sagittal slice positions (x-coordinates)
    pub sagittal: Vec<f64>,
    /// Coronal slice positions (y-coordinates)
    pub coronal: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clinical::therapy::swe_3d_workflows::decision_support::Swe3dClinicalDecisionSupport;
    use crate::clinical::therapy::swe_3d_workflows::statistics::VolumetricStatistics;
    use crate::domain::grid::Grid;

    #[test]
    fn test_multi_planar_reconstruction() {
        let grid = Grid::new(10, 10, 10, 0.002, 0.002, 0.002).unwrap();
        let mut map = ElasticityMap3D::new(&grid);
        map.young_modulus[[5, 5, 5]] = 10000.0;
        map.confidence[[5, 5, 5]] = 0.9;
        map.reliability_mask[[5, 5, 5]] = true;

        let mpr = MultiPlanarReconstruction::from_elasticity_map(&map, 0.004);
        assert!(!mpr.axial_slices.is_empty());
        assert!(!mpr.sagittal_slices.is_empty());
        assert!(!mpr.coronal_slices.is_empty());

        let first_axial_pos = mpr.slice_positions.axial[0];
        let axial_slice = mpr
            .get_slice(first_axial_pos, SliceOrientation::Axial)
            .unwrap();
        assert!(!axial_slice.young_modulus.is_empty());
    }

    #[test]
    fn test_clinical_report_generation() {
        let cds = Swe3dClinicalDecisionSupport::default();

        let stats = VolumetricStatistics {
            valid_voxels: 1500,
            mean_modulus: 8000.0,
            std_modulus: 1500.0,
            median_modulus: 8000.0,
            min_modulus: 5000.0,
            max_modulus: 12000.0,
            mean_speed: 1.8,
            mean_confidence: 0.85,
            mean_quality: 0.82,
            volume_coverage: 0.92,
        };

        let report = cds.generate_report("liver", &stats);
        assert!(report.contains("3D SWE Clinical Report"));
        assert!(report.contains("LIVER"));
        assert!(report.contains("Valid voxels: 1500"));
        assert!(report.contains("Mean Young's modulus: 8.0 kPa"));
        assert!(report.contains("Liver Fibrosis Assessment"));
    }
}
