//! Interface geometry for domain coupling

use crate::error::{KwaversResult, ValidationError};
use crate::grid::Grid;
use crate::solver::hybrid::domain_decomposition::DomainRegion;
use serde::{Deserialize, Serialize};

/// Interface geometry description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceGeometry {
    /// Interface normal direction (0=x, 1=y, 2=z)
    pub normal_direction: usize,
    /// Interface plane position
    pub plane_position: f64,
    /// Interface extent in tangential directions
    pub extent: (f64, f64),
    /// Interface area
    pub area: f64,
    /// Number of interface points
    pub num_points: usize,
}

impl InterfaceGeometry {
    /// Create interface geometry from two grids
    pub fn from_grids(source: &Grid, target: &Grid) -> KwaversResult<Self> {
        // Detect interface direction and position
        let (normal_direction, plane_position) = Self::detect_interface(source, target)?;

        // Calculate interface extent
        let extent = Self::calculate_extent(source, target, normal_direction)?;

        // Calculate interface area
        let area = extent.0 * extent.1;

        // Calculate number of interface points
        let num_points = Self::calculate_num_points(source, target, normal_direction);

        Ok(Self {
            normal_direction,
            plane_position,
            extent,
            area,
            num_points,
        })
    }

    fn detect_interface(source: &Grid, target: &Grid) -> KwaversResult<(usize, f64)> {
        // Detect interface by finding the dimension with maximum overlap
        // Returns (dimension_index, position) where dimension is 0=x, 1=y, 2=z
        
        // Calculate grid boundaries
        let source_bounds = (
            (0.0, source.nx as f64 * source.dx),
            (0.0, source.ny as f64 * source.dy),
            (0.0, source.nz as f64 * source.dz),
        );
        
        let target_bounds = (
            (0.0, target.nx as f64 * target.dx),
            (0.0, target.ny as f64 * target.dy),
            (0.0, target.nz as f64 * target.dz),
        );
        
        // Find dimension with smallest overlap (likely the interface)
        let overlaps = [
            (source_bounds.0.1.min(target_bounds.0.1) - f64::max(source_bounds.0.0, target_bounds.0.0)).abs(),
            (source_bounds.1.1.min(target_bounds.1.1) - f64::max(source_bounds.1.0, target_bounds.1.0)).abs(),
            (source_bounds.2.1.min(target_bounds.2.1) - f64::max(source_bounds.2.0, target_bounds.2.0)).abs(),
        ];
        
        let interface_dim = overlaps
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        // Interface position is at the boundary between grids
        let interface_pos = match interface_dim {
            0 => source_bounds.0.1.min(target_bounds.0.1),
            1 => source_bounds.1.1.min(target_bounds.1.1),
            2 => source_bounds.2.1.min(target_bounds.2.1),
            _ => 0.0,
        };
        
        Ok((interface_dim, interface_pos))
    }

    fn calculate_extent(
        source: &Grid,
        target: &Grid,
        normal_direction: usize,
    ) -> KwaversResult<(f64, f64)> {
        // Calculate extent in tangential directions
        match normal_direction {
            0 => Ok((source.dy * source.ny as f64, source.dz * source.nz as f64)),
            1 => Ok((source.dx * source.nx as f64, source.dz * source.nz as f64)),
            2 => Ok((source.dx * source.nx as f64, source.dy * source.ny as f64)),
            _ => Err(ValidationError::FieldValidation {
                field: "normal_direction".to_string(),
                value: format!("{}", normal_direction),
                constraint: "Must be 0, 1, or 2".to_string(),
            }
            .into()),
        }
    }

    fn calculate_num_points(source: &Grid, _target: &Grid, normal_direction: usize) -> usize {
        match normal_direction {
            0 => source.ny * source.nz,
            1 => source.nx * source.nz,
            2 => source.nx * source.ny,
            _ => 0,
        }
    }
}

/// Domain information for coupling
#[derive(Debug, Clone)]
pub struct DomainInfo {
    /// Domain region indices
    pub region: DomainRegion,
    /// Local grid properties
    pub local_grid: LocalGridProperties,
}

/// Local grid properties at interface
#[derive(Debug, Clone)]
pub struct LocalGridProperties {
    /// Grid spacing
    pub spacing: (f64, f64, f64),
    /// Number of points
    pub num_points: (usize, usize, usize),
    /// Origin position
    pub origin: (f64, f64, f64),
}
