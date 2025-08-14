// src/physics/mod.rs
pub mod bubble_dynamics;
pub mod chemistry;
// composable module removed - use plugin system instead
pub mod field_indices;  // Unified field indices (SSOT)
pub mod field_mapping;  // NEW: Unified field mapping system
pub mod heterogeneity;
pub mod mechanics;
// migration module removed - composable has been fully removed
pub mod optics;
pub mod plugin; // NEW: Plugin architecture for extensible physics
pub mod scattering;
pub mod state;
pub mod thermodynamics;
pub mod traits;
pub mod sonoluminescence_detector;

#[cfg(test)]
pub mod analytical_tests;

#[cfg(test)]
pub mod validation_tests;

// Re-export commonly used types
pub use bubble_dynamics::{BubbleField, BubbleState, BubbleParameters};
// Removed composable exports - use plugin system instead
pub use field_mapping::{UnifiedFieldType, FieldAccessor as UnifiedFieldAccessor, FieldAccessorMut};
pub use state::PhysicsState;
pub use traits::*;
pub use plugin::{PhysicsPlugin, PluginManager, PluginMetadata, PluginContext}; // NEW: Plugin exports
pub use optics::sonoluminescence::{SonoluminescenceEmission, EmissionParameters};

/// Functional transformation utilities for physics calculations
pub mod functional {
    use ndarray::Array3;
    use rayon::prelude::*;
    use std::ops::Add;
    
    /// Apply a transformation pipeline to a field
    pub struct FieldTransform<T> {
        transforms: Vec<Box<dyn Fn(&Array3<T>) -> Array3<T> + Send + Sync>>,
    }
    
    impl<T: Clone + Send + Sync + 'static> FieldTransform<T> {
        pub fn new() -> Self {
            Self {
                transforms: Vec::new(),
            }
        }
        
        /// Add a transformation to the pipeline
        pub fn then<F>(mut self, f: F) -> Self
        where
            F: Fn(&Array3<T>) -> Array3<T> + Send + Sync + 'static,
        {
            self.transforms.push(Box::new(f));
            self
        }
        
        /// Apply all transformations in sequence
        pub fn apply(&self, field: &Array3<T>) -> Array3<T> {
            self.transforms.iter()
                .fold(field.clone(), |acc, transform| transform(&acc))
        }
        
        /// Apply transformations in parallel to multiple fields
        pub fn apply_parallel(&self, fields: &[Array3<T>]) -> Vec<Array3<T>> {
            fields.par_iter()
                .map(|field| self.apply(field))
                .collect()
        }
    }
    
    /// Functional field operations using iterator combinators
    pub trait FieldOps {
        type Item;
        
        /// Map a function over all elements
        fn map_field<F, U>(&self, f: F) -> Array3<U>
        where
            F: Fn(&Self::Item) -> U;
        
        /// Filter and collect elements matching a predicate
        fn filter_indices<F>(&self, predicate: F) -> Vec<(usize, usize, usize)>
        where
            F: Fn(&Self::Item) -> bool;
        
        /// Fold over the field with an accumulator
        fn fold_field<F, U>(&self, init: U, f: F) -> U
        where
            F: Fn(U, &Self::Item) -> U;
        
        /// Scan over the field, producing intermediate results
        fn scan_field<F, U>(&self, init: U, f: F) -> Array3<U>
        where
            F: Fn(&U, &Self::Item) -> U,
            U: Clone;
    }
    
    impl<T: Clone + Send + Sync> FieldOps for Array3<T> {
        type Item = T;
        
        fn map_field<F, U>(&self, f: F) -> Array3<U>
        where
            F: Fn(&Self::Item) -> U,
        {
            self.mapv(|x| f(&x))
        }
        
        fn filter_indices<F>(&self, predicate: F) -> Vec<(usize, usize, usize)>
        where
            F: Fn(&Self::Item) -> bool,
        {
            self.indexed_iter()
                .filter_map(|((i, j, k), val)| {
                    predicate(val).then_some((i, j, k))
                })
                .collect()
        }
        
        fn fold_field<F, U>(&self, init: U, f: F) -> U
        where
            F: Fn(U, &Self::Item) -> U,
        {
            self.iter().fold(init, f)
        }
        
        fn scan_field<F, U>(&self, init: U, f: F) -> Array3<U>
        where
            F: Fn(&U, &Self::Item) -> U,
            U: Clone,
        {
            let shape = self.dim();
            let mut result = Array3::from_elem(shape, init.clone());
            let mut accumulator = init;
            
            for ((i, j, k), val) in self.indexed_iter() {
                accumulator = f(&accumulator, val);
                result[[i, j, k]] = accumulator.clone();
            }
            
            result
        }
    }
    
    /// Compose multiple field operations
    pub fn compose_operations<T, F1, F2, U, V>(
        field: &Array3<T>,
        op1: F1,
        op2: F2,
    ) -> Array3<V>
    where
        F1: Fn(&Array3<T>) -> Array3<U>,
        F2: Fn(&Array3<U>) -> Array3<V>,
    {
        op2(&op1(field))
    }
    
    /// Apply a kernel operation using functional style
    pub fn apply_kernel<T, K, U>(
        field: &Array3<T>,
        kernel: &Array3<K>,
        combine: impl Fn(&T, &K) -> U + Sync + Send,
    ) -> Array3<U>
    where
        T: Clone + Send + Sync,
        K: Clone + Send + Sync,
        U: Clone + Send + Sync + Default + Add<Output = U>,
    {
        let (nx, ny, nz) = field.dim();
        let (kx, ky, kz) = kernel.dim();
        let (hx, hy, hz) = (kx / 2, ky / 2, kz / 2);
        
        Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            kernel.indexed_iter()
                .filter_map(|((ki, kj, kk), kval)| {
                    let fi = (i + ki).wrapping_sub(hx);
                    let fj = (j + kj).wrapping_sub(hy);
                    let fk = (k + kk).wrapping_sub(hz);
                    
                    if fi < nx && fj < ny && fk < nz {
                        Some(combine(&field[[fi, fj, fk]], kval))
                    } else {
                        None
                    }
                })
                .fold(U::default(), |acc, val| acc + val)
        })
    }
    
    /// Lazy field iterator that applies transformations on demand
    pub struct LazyFieldIterator<'a, T, F, U> {
        field: &'a Array3<T>,
        transform: F,
        index: usize,
        _phantom: std::marker::PhantomData<U>,
    }
    
    impl<'a, T, F, U> LazyFieldIterator<'a, T, F, U>
    where
        F: Fn(&T) -> U,
    {
        pub fn new(field: &'a Array3<T>, transform: F) -> Self {
            Self {
                field,
                transform,
                index: 0,
                _phantom: std::marker::PhantomData,
            }
        }
    }
    
    impl<'a, T, F, U> Iterator for LazyFieldIterator<'a, T, F, U>
    where
        F: Fn(&T) -> U,
    {
        type Item = ((usize, usize, usize), U);
        
        fn next(&mut self) -> Option<Self::Item> {
            let total = self.field.len();
            if self.index >= total {
                return None;
            }
            
            let shape = self.field.dim();
            let k = self.index % shape.2;
            let j = (self.index / shape.2) % shape.1;
            let i = self.index / (shape.1 * shape.2);
            
            let value = (self.transform)(&self.field[[i, j, k]]);
            self.index += 1;
            
            Some(((i, j, k), value))
        }
    }
    
    /// Monadic operations for Result types in physics calculations
    pub trait ResultOps<T, E> {
        /// Map over successful values
        fn map_ok<F, U>(self, f: F) -> Result<U, E>
        where
            F: FnOnce(T) -> U;
        
        /// Flat map for chaining operations
        fn and_then_ok<F, U>(self, f: F) -> Result<U, E>
        where
            F: FnOnce(T) -> Result<U, E>;
        
        /// Apply a fallback on error
        fn or_else_err<F>(self, f: F) -> Result<T, E>
        where
            F: FnOnce(E) -> Result<T, E>;
    }
    
    impl<T, E> ResultOps<T, E> for Result<T, E> {
        fn map_ok<F, U>(self, f: F) -> Result<U, E>
        where
            F: FnOnce(T) -> U,
        {
            self.map(f)
        }
        
        fn and_then_ok<F, U>(self, f: F) -> Result<U, E>
        where
            F: FnOnce(T) -> Result<U, E>,
        {
            self.and_then(f)
        }
        
        fn or_else_err<F>(self, f: F) -> Result<T, E>
        where
            F: FnOnce(E) -> Result<T, E>,
        {
            self.or_else(f)
        }
    }
}