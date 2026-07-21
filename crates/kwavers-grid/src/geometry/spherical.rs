//! Disk and ball domains.

use std::f64::consts::{PI, TAU};

use leto::{Array1, Array2};
use tyche_core::{Counter, Seed, SplitMix64, UserDomain};

use super::sampling::{collect_points, sample_counter};
use super::{GeometricDomain, GeometryDimension, GeometryError, PointLocation};

const INTERIOR_TAG: u64 = u64::from_le_bytes(*b"sphrintp");
const BOUNDARY_TAG: u64 = u64::from_le_bytes(*b"sphrbndp");

/// Validated disk or ball domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalDomain {
    center: [f64; 3],
    radius: f64,
    dimension: GeometryDimension,
}

impl SphericalDomain {
    /// Construct a two-dimensional disk.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] unless the center is finite and the radius is
    /// finite, positive, representable at the center, and yields finite area
    /// and circumference.
    pub fn new_2d(x: f64, y: f64, radius: f64) -> Result<Self, GeometryError> {
        Self::new([x, y, 0.0], radius, GeometryDimension::Two)
    }

    /// Construct a three-dimensional ball.
    ///
    /// # Errors
    ///
    /// Returns [`GeometryError`] unless the center is finite and the radius is
    /// finite, positive, representable at the center, and yields finite volume
    /// and surface area.
    pub fn new_3d(x: f64, y: f64, z: f64, radius: f64) -> Result<Self, GeometryError> {
        Self::new([x, y, z], radius, GeometryDimension::Three)
    }

    /// Active center coordinates.
    #[must_use]
    pub fn center(&self) -> &[f64] {
        &self.center[..self.dimension.as_usize()]
    }

    /// Domain radius.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    fn new(
        center: [f64; 3],
        radius: f64,
        dimension: GeometryDimension,
    ) -> Result<Self, GeometryError> {
        for (axis, &coordinate) in center.iter().enumerate().take(dimension.as_usize()) {
            if !coordinate.is_finite() {
                return Err(GeometryError::InvalidCenter {
                    axis,
                    value: coordinate,
                });
            }
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(GeometryError::InvalidRadius { radius });
        }
        for &coordinate in center.iter().take(dimension.as_usize()) {
            let lower = coordinate - radius;
            let upper = coordinate + radius;
            if !lower.is_finite()
                || !upper.is_finite()
                || lower >= coordinate
                || upper <= coordinate
            {
                return Err(GeometryError::InvalidRadius { radius });
            }
        }
        let domain = Self {
            center,
            radius,
            dimension,
        };
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

    fn squared_distance(&self, point: &[f64]) -> f64 {
        point
            .iter()
            .enumerate()
            .map(|(axis, &coordinate)| {
                let delta = coordinate - self.center[axis];
                delta * delta
            })
            .sum()
    }

    fn interior_measure(&self) -> f64 {
        match self.dimension {
            GeometryDimension::Two => PI * self.radius.powi(2),
            GeometryDimension::Three => (4.0 / 3.0) * PI * self.radius.powi(3),
            GeometryDimension::One => unreachable!("invariant: spherical domains are 2-D or 3-D"),
        }
    }

    fn boundary_measure(&self) -> f64 {
        match self.dimension {
            GeometryDimension::Two => TAU * self.radius,
            GeometryDimension::Three => 4.0 * PI * self.radius.powi(2),
            GeometryDimension::One => unreachable!("invariant: spherical domains are 2-D or 3-D"),
        }
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

    fn radial_fraction(unit: f64, dimension: GeometryDimension) -> f64 {
        let fraction = match dimension {
            GeometryDimension::Two => unit.sqrt(),
            GeometryDimension::Three => unit.cbrt(),
            GeometryDimension::One => unreachable!("invariant: spherical domains are 2-D or 3-D"),
        };
        fraction.min(1.0_f64.next_down())
    }

    fn project_roundoff_into_interior(&self, point: &mut [f64]) {
        let dimensions = self.dimension.as_usize();
        if self.squared_distance(&point[..dimensions]) < self.radius * self.radius {
            return;
        }

        // Adding a radius just below R to a translated center can round to the
        // closed boundary. Move every non-central component by one
        // representable value toward the center. If the rounded norm remains
        // indistinguishable from R, the center is the guaranteed representable
        // point in the open ball.
        for (axis, coordinate) in point.iter_mut().enumerate().take(dimensions) {
            *coordinate = if *coordinate > self.center[axis] {
                coordinate.next_down()
            } else if *coordinate < self.center[axis] {
                coordinate.next_up()
            } else {
                *coordinate
            };
        }
        if self.squared_distance(&point[..dimensions]) >= self.radius * self.radius {
            point[..dimensions].copy_from_slice(&self.center[..dimensions]);
        }
    }
}

impl GeometricDomain for SphericalDomain {
    fn dimension(&self) -> GeometryDimension {
        self.dimension
    }

    fn contains(&self, point: &[f64]) -> bool {
        point.len() == self.dimension.as_usize()
            && point.iter().all(|coordinate| coordinate.is_finite())
            && self.squared_distance(point) <= self.radius * self.radius
    }

    fn classify_point(&self, point: &[f64], tolerance: f64) -> PointLocation {
        if point.len() != self.dimension.as_usize()
            || !tolerance.is_finite()
            || tolerance < 0.0
            || point.iter().any(|coordinate| !coordinate.is_finite())
        {
            return PointLocation::Exterior;
        }
        let distance = self.squared_distance(point).sqrt();
        if (distance - self.radius).abs() <= tolerance {
            PointLocation::Boundary
        } else if distance < self.radius {
            PointLocation::Interior
        } else {
            PointLocation::Exterior
        }
    }

    fn bounding_box(&self) -> Vec<f64> {
        let mut bounds = Vec::with_capacity(2 * self.dimension.as_usize());
        for axis in 0..self.dimension.as_usize() {
            bounds.extend_from_slice(&[
                self.center[axis] - self.radius,
                self.center[axis] + self.radius,
            ]);
        }
        bounds
    }

    fn normal(&self, point: &[f64], tolerance: f64) -> Option<Array1<f64>> {
        if self.classify_point(point, tolerance) != PointLocation::Boundary {
            return None;
        }
        let distance = self.squared_distance(point).sqrt();
        if distance == 0.0 {
            return None;
        }
        let dimensions = self.dimension.as_usize();
        let mut normal = Array1::zeros([dimensions]);
        for axis in 0..dimensions {
            normal[axis] = (point[axis] - self.center[axis]) / distance;
        }
        Some(normal)
    }

    fn measure(&self) -> f64 {
        self.interior_measure()
    }

    fn maximum_extent(&self) -> f64 {
        2.0 * self.radius
    }

    fn map_unit_interior(&self, unit: &[f64], output: &mut [f64]) -> Result<(), GeometryError> {
        let dimensions = self.validate_mapping(unit, output)?;
        let radial = self.radius * Self::radial_fraction(unit[0], self.dimension);
        let mut mapped = [0.0; 3];
        match self.dimension {
            GeometryDimension::Two => {
                let angle = TAU * unit[1];
                mapped[0] = radial.mul_add(angle.cos(), self.center[0]);
                mapped[1] = radial.mul_add(angle.sin(), self.center[1]);
            }
            GeometryDimension::Three => {
                let axial = 1.0 - 2.0 * unit[1];
                let transverse = axial.mul_add(-axial, 1.0).max(0.0).sqrt();
                let azimuth = TAU * unit[2];
                mapped[0] = (radial * transverse).mul_add(azimuth.cos(), self.center[0]);
                mapped[1] = (radial * transverse).mul_add(azimuth.sin(), self.center[1]);
                mapped[2] = radial.mul_add(axial, self.center[2]);
            }
            GeometryDimension::One => unreachable!("invariant: spherical domains are 2-D or 3-D"),
        }
        self.project_roundoff_into_interior(&mut mapped);
        output.copy_from_slice(&mapped[..dimensions]);
        Ok(())
    }

    fn sample_interior(
        &self,
        sample_count: usize,
        seed: Seed,
    ) -> Result<Array2<f64>, GeometryError> {
        match self.dimension {
            GeometryDimension::Two => {
                sample_counter::<2, INTERIOR_TAG, _>(self, sample_count, seed)
            }
            GeometryDimension::Three => {
                sample_counter::<3, INTERIOR_TAG, _>(self, sample_count, seed)
            }
            GeometryDimension::One => unreachable!("invariant: spherical domains are 2-D or 3-D"),
        }
    }

    fn sample_boundary(
        &self,
        sample_count: usize,
        seed: Seed,
    ) -> Result<Array2<f64>, GeometryError> {
        match self.dimension {
            GeometryDimension::Two => {
                collect_points(sample_count, |index, point: &mut [f64; 2]| {
                    let address = u64::try_from(index)
                        .expect("invariant: Tyche-supported targets have at most 64-bit usize");
                    let unit = Counter::<UserDomain<BOUNDARY_TAG>, SplitMix64>::unit::<f64>(
                        seed, address, 0,
                    );
                    let angle = TAU * unit;
                    point[0] = self.radius.mul_add(angle.cos(), self.center[0]);
                    point[1] = self.radius.mul_add(angle.sin(), self.center[1]);
                    Ok(())
                })
            }
            GeometryDimension::Three => {
                collect_points(sample_count, |index, point: &mut [f64; 3]| {
                    let address = u64::try_from(index)
                        .expect("invariant: Tyche-supported targets have at most 64-bit usize");
                    let axial = 1.0
                        - 2.0
                            * Counter::<UserDomain<BOUNDARY_TAG>, SplitMix64>::unit::<f64>(
                                seed, address, 0,
                            );
                    let azimuth = TAU
                        * Counter::<UserDomain<BOUNDARY_TAG>, SplitMix64>::unit::<f64>(
                            seed, address, 1,
                        );
                    let transverse = axial.mul_add(-axial, 1.0).max(0.0).sqrt();
                    point[0] = (self.radius * transverse).mul_add(azimuth.cos(), self.center[0]);
                    point[1] = (self.radius * transverse).mul_add(azimuth.sin(), self.center[1]);
                    point[2] = self.radius.mul_add(axial, self.center[2]);
                    Ok(())
                })
            }
            GeometryDimension::One => unreachable!("invariant: spherical domains are 2-D or 3-D"),
        }
    }
}

const _: () = assert!(INTERIOR_TAG != BOUNDARY_TAG);
