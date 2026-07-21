//! Axis-aligned interval, rectangle, and cuboid domains.

use std::borrow::Cow;

use leto::{Array1, Array2};
use tyche_core::{Counter, Seed, SplitMix64, UserDomain, WeightedCategorical};

use super::sampling::{collect_points, sample_counter};
use super::{GeometricDomain, GeometryDimension, GeometryError, PointLocation};

const INTERIOR_TAG: u64 = u64::from_le_bytes(*b"rectintr");
const BOUNDARY_COORDINATE_TAG: u64 = u64::from_le_bytes(*b"rectbndc");
const BOUNDARY_FACE_TAG: u64 = u64::from_le_bytes(*b"rectbndf");

/// Validated axis-aligned domain in one, two, or three dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectangularDomain {
    min: [f64; 3],
    max: [f64; 3],
    dimension: GeometryDimension,
}

impl RectangularDomain {
    /// Construct a one-dimensional interval.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] unless the bounds are finite, strictly
    /// ordered, and contain a representable interior value.
    pub fn new_1d(x_min: f64, x_max: f64) -> Result<Self, GeometryError> {
        Self::from_bounds([x_min, 0.0, 0.0], [x_max, 0.0, 0.0], GeometryDimension::One)
    }

    /// Construct a two-dimensional rectangle.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] unless every bound is finite, strictly
    /// ordered, and admits a representable interior and finite measures.
    pub fn new_2d(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Result<Self, GeometryError> {
        Self::from_bounds(
            [x_min, y_min, 0.0],
            [x_max, y_max, 0.0],
            GeometryDimension::Two,
        )
    }

    /// Construct a three-dimensional cuboid.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] unless every bound is finite, strictly
    /// ordered, and admits a representable interior and finite measures.
    pub fn new_3d(
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_min: f64,
        z_max: f64,
    ) -> Result<Self, GeometryError> {
        Self::from_bounds(
            [x_min, y_min, z_min],
            [x_max, y_max, z_max],
            GeometryDimension::Three,
        )
    }

    /// Minimum active coordinates.
    #[must_use]
    pub fn min(&self) -> &[f64] {
        &self.min[..self.dimension.as_usize()]
    }

    /// Maximum active coordinates.
    #[must_use]
    pub fn max(&self) -> &[f64] {
        &self.max[..self.dimension.as_usize()]
    }

    fn from_bounds(
        min: [f64; 3],
        max: [f64; 3],
        dimension: GeometryDimension,
    ) -> Result<Self, GeometryError> {
        let domain = Self {
            min,
            max,
            dimension,
        };
        for axis in 0..dimension.as_usize() {
            let lower = min[axis];
            let upper = max[axis];
            if !lower.is_finite()
                || !upper.is_finite()
                || lower.next_up() >= upper
                || !(upper - lower).is_finite()
            {
                return Err(GeometryError::InvalidBounds {
                    axis,
                    min: lower,
                    max: upper,
                });
            }
        }
        domain.validate_measure("interior", domain.interior_measure())?;
        domain.validate_measure("boundary", domain.boundary_measure())?;
        Ok(domain)
    }

    fn validate_measure(&self, kind: &'static str, value: f64) -> Result<(), GeometryError> {
        if value.is_finite() && value > 0.0 {
            Ok(())
        } else {
            Err(GeometryError::InvalidMeasure { kind, value })
        }
    }

    fn lengths(&self) -> [f64; 3] {
        let mut lengths = [1.0; 3];
        for (axis, length) in lengths
            .iter_mut()
            .enumerate()
            .take(self.dimension.as_usize())
        {
            *length = self.max[axis] - self.min[axis];
        }
        lengths
    }

    fn interior_measure(&self) -> f64 {
        self.lengths()[..self.dimension.as_usize()].iter().product()
    }

    fn face_measure(&self, fixed_axis: usize) -> f64 {
        self.lengths()[..self.dimension.as_usize()]
            .iter()
            .enumerate()
            .filter_map(|(axis, &length)| (axis != fixed_axis).then_some(length))
            .product()
    }

    fn boundary_measure(&self) -> f64 {
        (0..self.dimension.as_usize())
            .map(|axis| 2.0 * self.face_measure(axis))
            .sum()
    }

    fn map_axis(&self, axis: usize, unit: f64) -> f64 {
        let mapped = unit.mul_add(self.max[axis] - self.min[axis], self.min[axis]);
        mapped.clamp(self.min[axis].next_up(), self.max[axis].next_down())
    }

    fn validate_mapping(&self, unit: &[f64], output: &[f64]) -> Result<usize, GeometryError> {
        let dimensions = self.dimension.as_usize();
        for (role, actual) in [("unit input", unit.len()), ("point output", output.len())] {
            if actual != dimensions {
                return Err(GeometryError::DimensionMismatch {
                    role,
                    expected: dimensions,
                    actual,
                });
            }
        }
        for (axis, &coordinate) in unit.iter().enumerate() {
            if !coordinate.is_finite() || !(0.0..1.0).contains(&coordinate) {
                return Err(GeometryError::InvalidUnitCoordinate {
                    axis,
                    value: coordinate,
                });
            }
        }
        Ok(dimensions)
    }
}

impl GeometricDomain for RectangularDomain {
    fn dimension(&self) -> GeometryDimension {
        self.dimension
    }

    fn contains(&self, point: &[f64]) -> bool {
        point.len() == self.dimension.as_usize()
            && point
                .iter()
                .enumerate()
                .all(|(axis, &value)| value >= self.min[axis] && value <= self.max[axis])
    }

    fn classify_point(&self, point: &[f64], tolerance: f64) -> PointLocation {
        if point.len() != self.dimension.as_usize()
            || !tolerance.is_finite()
            || tolerance < 0.0
            || point.iter().any(|value| !value.is_finite())
        {
            return PointLocation::Exterior;
        }

        let mut on_boundary = false;
        for (axis, &value) in point.iter().enumerate() {
            if value < self.min[axis] - tolerance || value > self.max[axis] + tolerance {
                return PointLocation::Exterior;
            }
            on_boundary |= (value - self.min[axis]).abs() <= tolerance
                || (value - self.max[axis]).abs() <= tolerance;
        }
        if on_boundary {
            PointLocation::Boundary
        } else {
            PointLocation::Interior
        }
    }

    fn bounding_box(&self) -> Vec<f64> {
        let mut bounds = Vec::with_capacity(2 * self.dimension.as_usize());
        for axis in 0..self.dimension.as_usize() {
            bounds.extend_from_slice(&[self.min[axis], self.max[axis]]);
        }
        bounds
    }

    fn normal(&self, point: &[f64], tolerance: f64) -> Option<Array1<f64>> {
        if self.classify_point(point, tolerance) != PointLocation::Boundary {
            return None;
        }
        let dimensions = self.dimension.as_usize();
        let mut normal = Array1::zeros([dimensions]);
        for axis in 0..dimensions {
            if (point[axis] - self.min[axis]).abs() <= tolerance {
                normal[axis] = -1.0;
            } else if (point[axis] - self.max[axis]).abs() <= tolerance {
                normal[axis] = 1.0;
            }
        }
        let norm = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm > 0.0 {
            for axis in 0..dimensions {
                normal[axis] /= norm;
            }
        }
        Some(normal)
    }

    fn measure(&self) -> f64 {
        self.interior_measure()
    }

    fn map_unit_interior(&self, unit: &[f64], output: &mut [f64]) -> Result<(), GeometryError> {
        let dimensions = self.validate_mapping(unit, output)?;
        let mut mapped = [0.0; 3];
        for axis in 0..dimensions {
            mapped[axis] = self.map_axis(axis, unit[axis]);
        }
        output.copy_from_slice(&mapped[..dimensions]);
        Ok(())
    }

    fn sample_interior(
        &self,
        sample_count: usize,
        seed: Seed,
    ) -> Result<Array2<f64>, GeometryError> {
        match self.dimension {
            GeometryDimension::One => {
                sample_counter::<1, INTERIOR_TAG, _>(self, sample_count, seed)
            }
            GeometryDimension::Two => {
                sample_counter::<2, INTERIOR_TAG, _>(self, sample_count, seed)
            }
            GeometryDimension::Three => {
                sample_counter::<3, INTERIOR_TAG, _>(self, sample_count, seed)
            }
        }
    }

    fn sample_boundary(
        &self,
        sample_count: usize,
        seed: Seed,
    ) -> Result<Array2<f64>, GeometryError> {
        let dimensions = self.dimension.as_usize();
        let mut weights = [0.0; 6];
        for axis in 0..dimensions {
            let measure = self.face_measure(axis);
            weights[2 * axis] = measure;
            weights[2 * axis + 1] = measure;
        }
        let faces =
            WeightedCategorical::<f64, SplitMix64>::new(Cow::Borrowed(&weights[..2 * dimensions]))
                .expect("invariant: construction validates every finite positive face measure");
        let face_seed = Seed::new(Counter::<UserDomain<BOUNDARY_FACE_TAG>, SplitMix64>::word(
            seed, 0, 0,
        ));

        match self.dimension {
            GeometryDimension::One => {
                collect_points(sample_count, |index, point: &mut [f64; 1]| {
                    self.fill_boundary_point(index, seed, face_seed, &faces, point)
                })
            }
            GeometryDimension::Two => {
                collect_points(sample_count, |index, point: &mut [f64; 2]| {
                    self.fill_boundary_point(index, seed, face_seed, &faces, point)
                })
            }
            GeometryDimension::Three => {
                collect_points(sample_count, |index, point: &mut [f64; 3]| {
                    self.fill_boundary_point(index, seed, face_seed, &faces, point)
                })
            }
        }
    }
}

impl RectangularDomain {
    fn fill_boundary_point<const DIMENSIONS: usize>(
        &self,
        index: usize,
        coordinate_seed: Seed,
        face_seed: Seed,
        faces: &WeightedCategorical<'_, f64, SplitMix64>,
        point: &mut [f64; DIMENSIONS],
    ) -> Result<(), GeometryError> {
        let address = u64::try_from(index)
            .expect("invariant: Tyche-supported targets have at most 64-bit usize");
        let face = faces.at(face_seed, address).get();
        let fixed_axis = face / 2;
        for (axis, coordinate) in point.iter_mut().enumerate() {
            if axis == fixed_axis {
                *coordinate = if face.is_multiple_of(2) {
                    self.min[axis]
                } else {
                    self.max[axis]
                };
            } else {
                let draw = u64::try_from(axis)
                    .expect("invariant: an array dimension is representable as u64");
                let unit = Counter::<UserDomain<BOUNDARY_COORDINATE_TAG>, SplitMix64>::open_unit(
                    coordinate_seed,
                    address,
                    draw,
                );
                *coordinate = self.map_axis(axis, unit);
            }
        }
        Ok(())
    }
}

const _: () = {
    assert!(INTERIOR_TAG != BOUNDARY_COORDINATE_TAG);
    assert!(INTERIOR_TAG != BOUNDARY_FACE_TAG);
    assert!(BOUNDARY_COORDINATE_TAG != BOUNDARY_FACE_TAG);
};
