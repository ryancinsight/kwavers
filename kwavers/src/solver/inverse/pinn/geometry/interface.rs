use std::sync::Arc;

use ndarray::Array2;

use crate::domain::geometry::{GeometricDomain, PointLocation};

/// Interface condition specification for multi-region PINN training
///
/// Defines how the neural network solution should behave at internal boundaries
/// between regions with different material properties.
#[derive(Clone)]
pub enum InterfaceCondition {
    /// Continuity of displacement and traction (standard elastic interface)
    ElasticContinuity,

    /// Welded contact (same as ElasticContinuity, named for clarity)
    WeldedContact,

    /// Sliding contact (tangential slip allowed, normal stress continuous)
    SlidingContact,

    /// Free boundary (interface with vacuum/air): σ·n = 0
    FreeBoundary,

    /// Acoustic-elastic interface (fluid-solid coupling)
    AcousticElastic { fluid_density: f64 },

    /// Custom interface condition with user-defined residual function
    Custom {
        residual_fn: Arc<
            dyn Fn(&[f64], &[f64], &[[f64; 2]; 2], &[[f64; 2]; 2], &[f64]) -> f64 + Send + Sync,
        >,
    },
}

impl std::fmt::Debug for InterfaceCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ElasticContinuity => write!(f, "ElasticContinuity"),
            Self::WeldedContact => write!(f, "WeldedContact"),
            Self::SlidingContact => write!(f, "SlidingContact"),
            Self::FreeBoundary => write!(f, "FreeBoundary"),
            Self::AcousticElastic { fluid_density } => {
                write!(f, "AcousticElastic(ρ={})", fluid_density)
            }
            Self::Custom { .. } => write!(f, "Custom"),
        }
    }
}

/// Multi-region domain for heterogeneous media
///
/// Represents a domain composed of multiple subdomains with different material
/// properties. The PINN must satisfy interface conditions at region boundaries.
pub struct MultiRegionDomain {
    pub regions: Vec<Box<dyn GeometricDomain>>,
    pub material_ids: Vec<usize>,
    /// interfaces[i] specifies condition between regions i and i+1
    pub interfaces: Vec<InterfaceCondition>,
}

impl std::fmt::Debug for MultiRegionDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiRegionDomain")
            .field("regions", &format!("{} regions", self.regions.len()))
            .field("material_ids", &self.material_ids)
            .field("interfaces", &self.interfaces)
            .finish()
    }
}

impl MultiRegionDomain {
    /// Create a new multi-region domain
    pub fn new(
        regions: Vec<Box<dyn GeometricDomain>>,
        material_ids: Vec<usize>,
        interfaces: Vec<InterfaceCondition>,
    ) -> Self {
        assert_eq!(
            regions.len(),
            material_ids.len(),
            "Each region must have a material ID"
        );
        assert_eq!(
            interfaces.len(),
            regions.len() - 1,
            "Need N-1 interfaces for N regions"
        );
        Self {
            regions,
            material_ids,
            interfaces,
        }
    }

    /// Find which region contains a given point
    pub fn locate_point(&self, point: &[f64], tolerance: f64) -> Option<(usize, PointLocation)> {
        for (i, region) in self.regions.iter().enumerate() {
            let loc = region.classify_point(point, tolerance);
            if loc != PointLocation::Exterior {
                return Some((i, loc));
            }
        }
        None
    }

    /// Sample interface points between regions
    pub fn sample_interface_points(
        &self,
        n_points_per_interface: usize,
        seed: Option<u64>,
    ) -> Array2<f64> {
        let mut all_points = Vec::new();

        for i in 0..self.regions.len() - 1 {
            let boundary_i = self.regions[i].sample_boundary(n_points_per_interface * 2, seed);
            let tolerance = 1e-8;

            for row_idx in 0..boundary_i.nrows() {
                let point = boundary_i.row(row_idx);
                let point_slice: Vec<f64> = point.iter().cloned().collect();

                if self.regions[i + 1].classify_point(&point_slice, tolerance)
                    == PointLocation::Boundary
                {
                    all_points.push(point_slice);
                    if all_points.len() >= n_points_per_interface {
                        break;
                    }
                }
            }
        }

        let n_found = all_points.len();
        let dim = if n_found > 0 { all_points[0].len() } else { 2 };
        let mut result = Array2::zeros((n_found, dim));
        for (i, point) in all_points.iter().enumerate() {
            for (j, &coord) in point.iter().enumerate() {
                result[[i, j]] = coord;
            }
        }

        result
    }
}
