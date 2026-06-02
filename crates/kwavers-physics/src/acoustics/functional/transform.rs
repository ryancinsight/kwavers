//! Field Transformation Pipelines
//!
//! This module provides flexible, generic transformation pipelines for fields
//! with support for lazy evaluation and different data structures.

use ndarray::{Array2, Array3};
// LazyField removed - using eager evaluation
use rayon::prelude::*;

/// Generic field transformation pipeline
///
/// This structure supports transformations over any field type F,
/// enabling reuse with Array2<``T``>, Array3<``T``>, or custom field types.
pub struct FieldTransform<F> {
    transforms: Vec<Box<dyn Fn(F) -> F + Send + Sync>>,
}

impl<F> std::fmt::Debug for FieldTransform<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FieldTransform")
            .field("transforms_count", &self.transforms.len())
            .finish()
    }
}

impl<F> FieldTransform<F>
where
    F: Send + Sync + 'static,
{
    /// Create a new empty transformation pipeline
    #[must_use]
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Add a transformation to the pipeline
    pub fn then<G>(mut self, f: G) -> Self
    where
        G: Fn(F) -> F + Send + Sync + 'static,
    {
        self.transforms.push(Box::new(f));
        self
    }

    /// Apply all transformations in sequence
    pub fn apply(&self, field: F) -> F {
        self.transforms
            .iter()
            .fold(field, |acc, transform| transform(acc))
    }

    /// Apply transformations in parallel to multiple fields
    #[must_use]
    pub fn apply_parallel(&self, fields: Vec<F>) -> Vec<F> {
        fields
            .into_par_iter()
            .map(|field| self.apply(field))
            .collect()
    }

    /// Compose this transform with another
    pub fn compose<G>(self, other: G) -> Self
    where
        G: Fn(F) -> F + Send + Sync + 'static,
    {
        self.then(other)
    }

    /// Create a reversible transformation (if possible)
    pub fn reversible<G>(self, inverse: G) -> ReversibleTransform<F>
    where
        G: Fn(F) -> F + Send + Sync + 'static,
    {
        ReversibleTransform::new(self, inverse)
    }
}

impl<F> Default for FieldTransform<F>
where
    F: Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Reversible field transformation
pub struct ReversibleTransform<F> {
    forward: FieldTransform<F>,
    inverse: Box<dyn Fn(F) -> F + Send + Sync>,
}

impl<F> std::fmt::Debug for ReversibleTransform<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReversibleTransform")
            .field("forward", &self.forward)
            .field("inverse", &"<dyn Fn>")
            .finish()
    }
}

impl<F> ReversibleTransform<F>
where
    F: Send + Sync + 'static,
{
    pub fn new<G>(forward: FieldTransform<F>, inverse: G) -> Self
    where
        G: Fn(F) -> F + Send + Sync + 'static,
    {
        Self {
            forward,
            inverse: Box::new(inverse),
        }
    }

    /// Apply the forward transformation
    pub fn forward(&self, field: F) -> F {
        self.forward.apply(field)
    }

    /// Apply the inverse transformation
    pub fn inverse(&self, field: F) -> F {
        (self.inverse)(field)
    }
}

// Convenience type aliases for common field types
pub type Array2Transform<T> = FieldTransform<Array2<T>>;
pub type Array3Transform<T> = FieldTransform<Array3<T>>;

// Predefined transformation builders for common operations
impl<T> Array3Transform<T>
where
    T: Clone + Send + Sync + 'static + std::ops::Mul<f64, Output = T>,
{
    /// Create a scaling transformation
    #[must_use]
    pub fn scale(factor: f64) -> Self {
        Self::new().then(move |field| field.mapv_into(|x| x * factor))
    }
}

impl<T> Array3Transform<T>
where
    T: Copy + Send + Sync + 'static + std::ops::Add<Output = T> + std::ops::Div<f64, Output = T>,
{
    /// Create a smoothing transformation using a simple kernel
    #[must_use]
    pub fn smooth() -> Self {
        Self::new().then(smooth_field)
    }
}

impl<T> Array2Transform<T>
where
    T: Clone + Send + Sync + 'static + std::ops::Mul<f64, Output = T>,
{
    /// Create a scaling transformation for 2D fields
    #[must_use]
    pub fn scale(factor: f64) -> Self {
        Self::new().then(move |field| field.mapv_into(|x| x * factor))
    }
}

/// Apply the seven-point smoothing stencil with one output allocation.
///
/// Boundary cells are copied from the input because the centered stencil is not
/// defined there. Interior cells use the axis-neighbor helper below, whose
/// `AXIS` and `FORWARD` const parameters produce one inlined specialization per
/// structural stencil direction.
fn smooth_field<T>(field: Array3<T>) -> Array3<T>
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Div<f64, Output = T>,
{
    let shape = field.dim();
    Array3::from_shape_fn(shape, |idx| {
        if is_boundary(idx, shape) {
            field[idx]
        } else {
            smooth_interior_cell(&field, idx)
        }
    })
}

fn is_boundary((i, j, k): (usize, usize, usize), (nx, ny, nz): (usize, usize, usize)) -> bool {
    i == 0 || j == 0 || k == 0 || i + 1 == nx || j + 1 == ny || k + 1 == nz
}

#[inline]
fn smooth_interior_cell<T>(field: &Array3<T>, (i, j, k): (usize, usize, usize)) -> T
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Div<f64, Output = T>,
{
    let sum = axis_neighbor::<0, false, T>(field, i, j, k)
        + axis_neighbor::<0, true, T>(field, i, j, k)
        + axis_neighbor::<1, false, T>(field, i, j, k)
        + axis_neighbor::<1, true, T>(field, i, j, k)
        + axis_neighbor::<2, false, T>(field, i, j, k)
        + axis_neighbor::<2, true, T>(field, i, j, k)
        + field[[i, j, k]];
    sum / 7.0
}

#[inline]
fn axis_neighbor<const AXIS: usize, const FORWARD: bool, T: Copy>(
    field: &Array3<T>,
    i: usize,
    j: usize,
    k: usize,
) -> T {
    match (AXIS, FORWARD) {
        (0, false) => field[[i - 1, j, k]],
        (0, true) => field[[i + 1, j, k]],
        (1, false) => field[[i, j - 1, k]],
        (1, true) => field[[i, j + 1, k]],
        (2, false) => field[[i, j, k - 1]],
        (2, true) => field[[i, j, k + 1]],
        _ => unreachable!("Array3 smoothing only supports axes 0, 1, and 2"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array3;

    #[derive(Debug, PartialEq, Eq)]
    struct NonCloneValue(i32);

    #[test]
    fn test_generic_transform() {
        let field = Array3::from_elem((4, 4, 4), 1.0);
        let transform = Array3Transform::new()
            .then(|f| f.mapv(|x: f64| x * 2.0))
            .then(|f| f.mapv(|x: f64| x + 1.0));

        let result = transform.apply(field);
        assert_abs_diff_eq!(result[[2, 2, 2]], 3.0);
    }

    #[test]
    fn transform_pipeline_accepts_non_clone_owned_values() {
        let transform = FieldTransform::new()
            .then(|value: NonCloneValue| NonCloneValue(value.0 + 3))
            .then(|value: NonCloneValue| NonCloneValue(value.0 * 2));

        let result = transform.apply(NonCloneValue(4));

        assert_eq!(result, NonCloneValue(14));
    }

    #[test]
    fn smooth_preserves_boundaries_and_matches_quadratic_stencil() {
        let field = Array3::from_shape_fn((5, 5, 5), |(i, j, k)| (i * i + j * j + k * k) as f64);
        let transform = Array3Transform::<f64>::smooth();

        let result = transform.apply(field);

        assert_abs_diff_eq!(result[[0, 2, 2]], 8.0);
        assert_abs_diff_eq!(result[[2, 2, 2]], 90.0 / 7.0, epsilon = 1e-15);
    }

    #[test]
    fn smooth_handles_domains_without_interior_stencil() {
        let field = Array3::from_shape_fn((1, 2, 3), |(i, j, k)| (i + j + k) as f64);
        let transform = Array3Transform::<f64>::smooth();

        let result = transform.apply(field);

        assert_abs_diff_eq!(result[[0, 0, 0]], 0.0);
        assert_abs_diff_eq!(result[[0, 1, 2]], 3.0);
    }
}
