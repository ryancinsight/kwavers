//! Field Transformation Pipelines
//!
//! This module provides flexible, generic transformation pipelines for fields
//! with support for lazy evaluation and different data structures.

use ndarray::{Array2, Array3};
use crate::solver::lazy::LazyField;
use rayon::prelude::*;

/// Generic field transformation pipeline
/// 
/// This structure supports transformations over any field type F,
/// enabling reuse with Array2<T>, Array3<T>, or custom field types.
pub struct FieldTransform<F> {
    transforms: Vec<Box<dyn Fn(F) -> F + Send + Sync>>,
}

impl<F> FieldTransform<F>
where
    F: Clone + Send + Sync + 'static,
{
    /// Create a new empty transformation pipeline
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
        self.transforms.iter().fold(field, |acc, transform| transform(acc))
    }
    
    /// Apply transformations in parallel to multiple fields
    pub fn apply_parallel(&self, fields: Vec<F>) -> Vec<F> {
        fields.into_par_iter()
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
    F: Clone + Send + Sync + 'static,
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

impl<F> ReversibleTransform<F>
where
    F: Clone + Send + Sync + 'static,
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

/// Lazy field transformation with integrated lazy evaluation
pub struct LazyFieldTransform<T> {
    source: LazyField<T>,
    transforms: Vec<Box<dyn Fn(Array3<T>) -> Array3<T> + Send + Sync>>,
}

impl<T: Clone + Send + Sync + 'static> LazyFieldTransform<T> {
    /// Create a new lazy transformation from a source
    pub fn new(source: LazyField<T>) -> Self {
        Self {
            source,
            transforms: Vec::new(),
        }
    }
    
    /// Add a transformation to the lazy pipeline
    pub fn then<F>(mut self, f: F) -> Self
    where
        F: Fn(Array3<T>) -> Array3<T> + Send + Sync + 'static,
    {
        self.transforms.push(Box::new(f));
        self
    }
    
    /// Convert back to a lazy field
    pub fn to_lazy(self) -> LazyField<T> {
        LazyField::new(move || {
            let mut field = self.source.get();
            for transform in &self.transforms {
                field = transform(field);
            }
            field
        })
    }
    
    /// Evaluate the lazy transformation immediately
    pub fn evaluate(self) -> Array3<T> {
        self.to_lazy().get()
    }
    
    /// Create a cached version that evaluates only once
    pub fn cached(self) -> CachedTransform<T> {
        CachedTransform::new(self.to_lazy())
    }
}

/// Cached transformation that evaluates only once
pub struct CachedTransform<T> {
    lazy_field: LazyField<T>,
    cached_result: std::sync::Mutex<Option<Array3<T>>>,
}

impl<T: Clone + Send + Sync + 'static> CachedTransform<T> {
    fn new(lazy_field: LazyField<T>) -> Self {
        Self {
            lazy_field,
            cached_result: std::sync::Mutex::new(None),
        }
    }
    
    /// Get the cached result, computing it if necessary
    pub fn get(&self) -> Array3<T> {
        let mut cache = self.cached_result.lock().unwrap();
        if let Some(ref result) = *cache {
            result.clone()
        } else {
            let result = self.lazy_field.get();
            *cache = Some(result.clone());
            result
        }
    }
    
    /// Clear the cache, forcing recomputation on next access
    pub fn invalidate(&self) {
        *self.cached_result.lock().unwrap() = None;
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
    pub fn scale(factor: f64) -> Self {
        Self::new().then(move |field| field.mapv(|x| x * factor))
    }
    
    /// Create a smoothing transformation using a simple kernel
    pub fn smooth() -> Self
    where
        T: Default + std::ops::Add<Output = T> + std::ops::Div<f64, Output = T>,
    {
        Self::new().then(|field| {
            let mut result = field.clone();
            let (nx, ny, nz) = field.dim();
            
            for i in 1..nx-1 {
                for j in 1..ny-1 {
                    for k in 1..nz-1 {
                        let sum = field[[i-1,j,k]].clone() + field[[i+1,j,k]].clone() +
                                 field[[i,j-1,k]].clone() + field[[i,j+1,k]].clone() +
                                 field[[i,j,k-1]].clone() + field[[i,j,k+1]].clone() +
                                 field[[i,j,k]].clone();
                        result[[i,j,k]] = sum / 7.0;
                    }
                }
            }
            result
        })
    }
}

impl<T> Array2Transform<T>
where
    T: Clone + Send + Sync + 'static + std::ops::Mul<f64, Output = T>,
{
    /// Create a scaling transformation for 2D fields
    pub fn scale(factor: f64) -> Self {
        Self::new().then(move |field| field.mapv(|x| x * factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_generic_transform() {
        let field = Array3::ones((4, 4, 4));
        let transform = Array3Transform::new()
            .then(|f| f.mapv(|x: f64| x * 2.0))
            .then(|f| f.mapv(|x: f64| x + 1.0));
        
        let result = transform.apply(field);
        assert_abs_diff_eq!(result[[2, 2, 2]], 3.0);
    }
    
    #[test]
    fn test_lazy_transform() {
        let source = LazyField::new(|| Array3::ones((2, 2, 2)));
        let lazy_transform = LazyFieldTransform::new(source)
            .then(|f| f.mapv(|x: f64| x * 2.0));
        
        let result = lazy_transform.evaluate();
        assert_abs_diff_eq!(result[[1, 1, 1]], 2.0);
    }
    
    #[test]
    fn test_cached_transform() {
        let source = LazyField::new(|| Array3::ones((2, 2, 2)));
        let cached = LazyFieldTransform::new(source)
            .then(|f| f.mapv(|x: f64| x * 3.0))
            .cached();
        
        let result1 = cached.get();
        let result2 = cached.get(); // Should use cached value
        
        assert_abs_diff_eq!(result1[[0, 0, 0]], 3.0);
        assert_abs_diff_eq!(result2[[0, 0, 0]], 3.0);
    }
    
    #[test]
    fn test_reversible_transform() {
        let field = Array3::ones((2, 2, 2)) * 5.0;
        let transform = Array3Transform::new()
            .then(|f| f.mapv(|x: f64| x * 2.0))
            .reversible(|f| f.mapv(|x: f64| x / 2.0));
        
        let forward = transform.forward(field.clone());
        let backward = transform.inverse(forward);
        
        assert_abs_diff_eq!(backward[[1, 1, 1]], field[[1, 1, 1]]);
    }
}