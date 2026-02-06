//! Grid sampling utilities for high-level simulation and recorder usage.
//!
//! # Purpose
//! This module defines a *simple* sensor set that samples simulation fields on a
//! discrete grid.
//!
//! It supports **both acoustics and optics**:
//! - Acoustic: pressure sampling
//! - Optical: fluence (or light intensity/fluence-rate proxy) sampling
//!
//! # Design constraints
//! - No solver coupling: this is a pure data/selection structure plus sampling.
//! - No silent masking: sampling returns `Option<f64>` per point to avoid
//!   silently fabricating zeros on out-of-bounds access.
//! - Minimal surface: intended for high-level orchestration (`simulation/mod.rs`)
//!   and recorder components that want a straightforward grid probe set.
//!
//! # Relation to mask-based sensors
//! A boolean "sensor mask" specifies a set of grid points. `GridSensorSet` provides
//! the equivalent (a list of grid indices or a derived mask) while being usable
//! for multi-physics (acoustic + optical) sampling.

use ndarray::{Array2, Array3};

/// A point on the solver grid (integer indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GridPoint {
    pub i: usize,
    pub j: usize,
    pub k: usize,
}

impl GridPoint {
    #[must_use]
    pub const fn new(i: usize, j: usize, k: usize) -> Self {
        Self { i, j, k }
    }
}

/// A simple grid sensor set.
///
/// The set can be used to sample multiple fields (acoustic pressure, optical
/// fluence) at identical grid points.
///
/// ## Invariants
/// - `points` is treated as an ordered list. If callers require uniqueness,
///   they must enforce it (or use `dedup_in_place()`).
/// - Sampling does not mutate the sensor set and does not allocate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GridSensorSet {
    pub points: Vec<GridPoint>,
    /// Storage for recorded pressure data [num_points, time_steps]
    #[serde(skip)]
    pub pressure_data: Option<Array2<f64>>,
}

impl GridSensorSet {
    /// Create an empty sensor set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            pressure_data: None,
        }
    }

    /// Create a sensor set from explicit grid points.
    #[must_use]
    pub fn from_points(points: Vec<GridPoint>) -> Self {
        Self {
            points,
            pressure_data: None,
        }
    }

    /// Borrow the sensor points.
    #[must_use]
    pub fn points(&self) -> &[GridPoint] {
        &self.points
    }

    /// Return number of sensor points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Alias for len() for compatibility
    pub fn num_sensors(&self) -> usize {
        self.len()
    }

    /// Alias for sample_scalar
    pub fn sample(&self, field: &Array3<f64>) -> Vec<Option<f64>> {
        self.sample_scalar(field)
    }

    /// Whether the sensor set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Push a new grid point (does not enforce uniqueness).
    pub fn push(&mut self, point: GridPoint) {
        self.points.push(point);
    }

    /// Deduplicate points in-place while preserving the first occurrence order.
    ///
    /// This is O(n²) worst-case; intended for small high-level sensor lists.
    /// If you need large-scale mask processing, prefer boolean masks and
    /// dedicated index extraction.
    pub fn dedup_in_place(&mut self) {
        let mut i = 0;
        while i < self.points.len() {
            let mut j = i + 1;
            while j < self.points.len() {
                if self.points[i] == self.points[j] {
                    self.points.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    /// Convert this sensor set to a boolean mask of shape `(nx, ny, nz)`.
    ///
    /// Points outside the shape are ignored (no panic). This is intentional to
    /// allow composition with validation elsewhere. If you need strictness,
    /// validate points before calling this function.
    #[must_use]
    pub fn to_mask(&self, shape: (usize, usize, usize)) -> Array3<bool> {
        let (nx, ny, nz) = shape;
        let mut mask = Array3::from_elem((nx, ny, nz), false);

        for &p in &self.points {
            if p.i < nx && p.j < ny && p.k < nz {
                mask[[p.i, p.j, p.k]] = true;
            }
        }

        mask
    }

    /// Sample a scalar field at each sensor point.
    ///
    /// Returns a vector aligned with `self.points()` where each entry is `Some(value)`
    /// if sampled in-bounds, else `None`.
    ///
    /// ## Why `Option`?
    /// Returning `0.0` for out-of-bounds reads is error masking and breaks
    /// correctness invariants in downstream algorithms. Callers may map `None`
    /// to a sentinel if they *explicitly* want that behavior.
    #[must_use]
    pub fn sample_scalar(&self, field: &Array3<f64>) -> Vec<Option<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut out = Vec::with_capacity(self.points.len());

        for p in &self.points {
            if p.i < nx && p.j < ny && p.k < nz {
                out.push(Some(field[[p.i, p.j, p.k]]));
            } else {
                out.push(None);
            }
        }

        out
    }

    /// Sample acoustic pressure at each sensor point.
    ///
    /// This is a semantic alias of `sample_scalar` used by high-level orchestration.
    #[must_use]
    pub fn sample_pressure(&self, pressure: &Array3<f64>) -> Vec<Option<f64>> {
        self.sample_scalar(pressure)
    }

    /// Sample optical fluence (or fluence-rate proxy) at each sensor point.
    ///
    /// This is a semantic alias of `sample_scalar` for Kwavers optical support.
    #[must_use]
    pub fn sample_fluence(&self, fluence: &Array3<f64>) -> Vec<Option<f64>> {
        self.sample_scalar(fluence)
    }
}

impl Default for GridSensorSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_mask_marks_in_bounds_points_only() {
        let mut set = GridSensorSet::new();
        set.push(GridPoint::new(0, 0, 0));
        set.push(GridPoint::new(1, 1, 0));
        set.push(GridPoint::new(9, 9, 9)); // out-of-bounds for shape below

        let mask = set.to_mask((2, 2, 1));
        assert!(mask[[0, 0, 0]]);
        assert!(mask[[1, 1, 0]]);
        assert_eq!(mask.len(), 4);

        // Out-of-bounds point must be ignored, so all remaining entries are false.
        assert!(!mask[[0, 1, 0]]);
        assert!(!mask[[1, 0, 0]]);
    }

    #[test]
    fn sample_returns_none_for_out_of_bounds() {
        let field = Array3::from_shape_fn((2, 2, 1), |(i, j, _k)| (10 * i + j) as f64);

        let set = GridSensorSet::from_points(vec![
            GridPoint::new(0, 0, 0),
            GridPoint::new(1, 1, 0),
            GridPoint::new(2, 0, 0), // OOB
        ]);

        let samples = set.sample_scalar(&field);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0], Some(0.0));
        assert_eq!(samples[1], Some(11.0));
        assert_eq!(samples[2], None);
    }

    #[test]
    fn dedup_in_place_removes_duplicates_preserving_first_order() {
        let mut set = GridSensorSet::from_points(vec![
            GridPoint::new(0, 0, 0),
            GridPoint::new(1, 0, 0),
            GridPoint::new(0, 0, 0),
            GridPoint::new(1, 0, 0),
            GridPoint::new(0, 1, 0),
        ]);

        set.dedup_in_place();
        assert_eq!(
            set.points(),
            &[
                GridPoint::new(0, 0, 0),
                GridPoint::new(1, 0, 0),
                GridPoint::new(0, 1, 0)
            ]
        );
    }
}
