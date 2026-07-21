//! Validated multi-region domain.

use leto::Array2;
use tyche_core::{Counter, Seed, SplitMix64, UserDomain};

use super::{MultiRegionError, PinnGeometryInterfaceCondition};
use kwavers_grid::geometry::{GeometricDomain, PointLocation};

const INTERFACE_SEED_TAG: u64 = u64::from_le_bytes(*b"pinnintr");
// Boundary charts and classification use at most 16 rounded arithmetic steps
// in three dimensions. Fourfold slack covers the chart's transcendental
// evaluations without introducing a unit-dependent absolute tolerance.
const INTERFACE_ROUNDOFF_FACTOR: f64 = 64.0;

/// Heterogeneous PINN domain with one condition between adjacent regions.
///
/// The boxed trait objects are a deliberate cold-boundary exception: a
/// runtime material model may contain heterogeneous domain implementations.
/// Interior collocation remains statically dispatched through
/// [`super::super::CollocationSampler`].
pub struct MultiRegionDomain {
    regions: Vec<Box<dyn GeometricDomain>>,
    material_ids: Vec<usize>,
    interfaces: Vec<PinnGeometryInterfaceCondition>,
    dimension: usize,
}

impl std::fmt::Debug for MultiRegionDomain {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MultiRegionDomain")
            .field("region_count", &self.regions.len())
            .field("material_ids", &self.material_ids)
            .field("interfaces", &self.interfaces)
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl MultiRegionDomain {
    /// Validate and construct a heterogeneous domain.
    ///
    /// # Errors
    ///
    /// Returns [`MultiRegionError`] for an empty region set, count mismatch, or
    /// inconsistent spatial dimension.
    pub fn new(
        regions: Vec<Box<dyn GeometricDomain>>,
        material_ids: Vec<usize>,
        interfaces: Vec<PinnGeometryInterfaceCondition>,
    ) -> Result<Self, MultiRegionError> {
        let Some(first) = regions.first() else {
            return Err(MultiRegionError::Empty);
        };
        if material_ids.len() != regions.len() {
            return Err(MultiRegionError::MaterialCount {
                regions: regions.len(),
                materials: material_ids.len(),
            });
        }
        let expected_interfaces = regions.len() - 1;
        if interfaces.len() != expected_interfaces {
            return Err(MultiRegionError::InterfaceCount {
                regions: regions.len(),
                expected: expected_interfaces,
                actual: interfaces.len(),
            });
        }
        let dimension = first.dimension().as_usize();
        for (region, domain) in regions.iter().enumerate().skip(1) {
            let actual = domain.dimension().as_usize();
            if actual != dimension {
                return Err(MultiRegionError::DimensionMismatch {
                    region,
                    expected: dimension,
                    actual,
                });
            }
        }
        Ok(Self {
            regions,
            material_ids,
            interfaces,
            dimension,
        })
    }

    /// Number of material regions.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Material identifier aligned with each region.
    #[must_use]
    pub fn material_ids(&self) -> &[usize] {
        &self.material_ids
    }

    /// Conditions aligned with adjacent region pairs.
    #[must_use]
    pub fn interfaces(&self) -> &[PinnGeometryInterfaceCondition] {
        &self.interfaces
    }

    /// Locate the first region containing `point`.
    #[must_use]
    pub fn locate_point(&self, point: &[f64], tolerance: f64) -> Option<(usize, PointLocation)> {
        self.regions.iter().enumerate().find_map(|(index, region)| {
            let location = region.classify_point(point, tolerance);
            (location != PointLocation::Exterior).then_some((index, location))
        })
    }

    /// Sample up to `sample_count` shared-boundary points per adjacent pair.
    ///
    /// General heterogeneous domains do not expose a shared interface chart,
    /// so this method filters independently sampled boundary candidates. The
    /// result may contain fewer points when the geometric intersection has
    /// small boundary measure.
    ///
    /// # Errors
    ///
    /// Returns [`MultiRegionError`] when the candidate or result matrix is not
    /// addressable or reservable, or a region boundary sampler fails.
    pub fn sample_interface_points(
        &self,
        sample_count: usize,
        seed: Seed,
    ) -> Result<Array2<f64>, MultiRegionError> {
        if self.interfaces.is_empty() {
            return Ok(Array2::from_shape_vec([0, self.dimension], Vec::new())
                .expect("invariant: an empty point buffer has zero matrix rows"));
        }
        let maximum_rows = sample_count.checked_mul(self.interfaces.len()).ok_or(
            kwavers_grid::geometry::GeometryError::ElementCountOverflow {
                sample_count,
                dimensions: self.interfaces.len(),
            },
        )?;
        let maximum_values = maximum_rows.checked_mul(self.dimension).ok_or(
            kwavers_grid::geometry::GeometryError::ElementCountOverflow {
                sample_count: maximum_rows,
                dimensions: self.dimension,
            },
        )?;
        let candidate_count = sample_count.checked_mul(2).ok_or(
            kwavers_grid::geometry::GeometryError::ElementCountOverflow {
                sample_count,
                dimensions: 2,
            },
        )?;
        let mut values = Vec::new();
        values.try_reserve_exact(maximum_values).map_err(|_| {
            kwavers_grid::geometry::GeometryError::AllocationFailed {
                element_count: maximum_values,
            }
        })?;

        for pair in 0..self.interfaces.len() {
            let address = u64::try_from(pair)
                .expect("invariant: an in-memory region index is representable as u64");
            let pair_seed = Seed::new(Counter::<UserDomain<INTERFACE_SEED_TAG>, SplitMix64>::word(
                seed, address, 0,
            ));
            let boundary = self.regions[pair].sample_boundary(candidate_count, pair_seed)?;
            let tolerance = INTERFACE_ROUNDOFF_FACTOR
                * f64::EPSILON
                * self.regions[pair]
                    .maximum_extent()
                    .max(self.regions[pair + 1].maximum_extent());
            let mut accepted = 0;
            for row in 0..boundary.shape()[0] {
                let mut point = [0.0; 3];
                for column in 0..self.dimension {
                    point[column] = boundary[[row, column]];
                }
                if self.regions[pair + 1].classify_point(&point[..self.dimension], tolerance)
                    == PointLocation::Boundary
                {
                    values.extend_from_slice(&point[..self.dimension]);
                    accepted += 1;
                    if accepted == sample_count {
                        break;
                    }
                }
            }
        }

        let rows = values.len() / self.dimension;
        Ok(Array2::from_shape_vec([rows, self.dimension], values)
            .expect("invariant: accepted interface points are appended by complete rows"))
    }
}
