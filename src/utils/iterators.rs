// src/utils/iterators.rs
//! Zero-cost iterator abstractions for efficient physics simulations
//! 
//! This module provides iterator-based patterns that leverage Rust's zero-cost abstractions
//! to achieve high performance while maintaining readable and maintainable code.

use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};
use rayon::prelude::*;
use num_complex::Complex;

/// Iterator for processing 3D grid points with spatial coordinates
pub struct GridPointIterator<'a, T> {
    array: ArrayViewMut3<'a, T>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl<'a, T> GridPointIterator<'a, T> 
where 
    T: Send + Sync + Clone + Default,
{
    pub fn new(array: ArrayViewMut3<'a, T>) -> Self {
        let (nx, ny, nz) = array.dim();
        Self { array, nx, ny, nz }
    }
    
    /// Process interior points (excluding boundaries) with iterator pattern
    pub fn process_interior<F>(mut self, dx: f64, dy: f64, dz: f64, processor: F)
    where
        F: Fn(usize, usize, usize, f64, f64, f64, &mut T) + Sync + Send,
    {
        (1..self.nx-1)
            .into_par_iter()
            .for_each(|i| {
                for j in 1..self.ny-1 {
                    for k in 1..self.nz-1 {
                        let x = i as f64 * dx;
                        let y = j as f64 * dy;
                        let z = k as f64 * dz;
                        // Safe access within bounds - using unsafe for performance
                        unsafe {
                            let ptr = self.array.as_mut_ptr().add(i * self.ny * self.nz + j * self.nz + k);
                            processor(i, j, k, x, y, z, &mut *ptr);
                        }
                    }
                }
            });
    }
}

/// Iterator for computing gradients with central differences
pub struct GradientComputer<'a> {
    array: ArrayView3<'a, f64>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl<'a> GradientComputer<'a> {
    pub fn new(array: ArrayView3<'a, f64>) -> Self {
        let (nx, ny, nz) = array.dim();
        Self { array, nx, ny, nz }
    }
    
    /// Compute gradients at interior points using iterator pattern
    pub fn compute_interior_gradients<F>(&self, dx: f64, dy: f64, dz: f64, mut processor: F)
    where
        F: FnMut(f64, f64, f64, usize, usize, usize) + Sync + Send,
    {
        let dx_inv = 1.0 / (2.0 * dx);
        let dy_inv = 1.0 / (2.0 * dy);
        let dz_inv = 1.0 / (2.0 * dz);
        
        (1..self.nx-1)
            .into_par_iter()
            .for_each(|i| {
                for j in 1..self.ny-1 {
                    for k in 1..self.nz-1 {
                        let grad_x = (self.array[[i+1, j, k]] - self.array[[i-1, j, k]]) * dx_inv;
                        let grad_y = (self.array[[i, j+1, k]] - self.array[[i, j-1, k]]) * dy_inv;
                        let grad_z = (self.array[[i, j, k+1]] - self.array[[i, j, k-1]]) * dz_inv;
                        
                        processor(grad_x, grad_y, grad_z, i, j, k);
                    }
                }
            });
    }
}

/// Iterator for chunked processing with cache-friendly access patterns
pub struct ChunkedProcessor<T> {
    data: Vec<T>,
    chunk_size: usize,
}

impl<T> ChunkedProcessor<T> 
where 
    T: Send + Sync + Clone,
{
    pub fn new(data: Vec<T>, chunk_size: usize) -> Self {
        Self { data, chunk_size }
    }
    
    /// Process data in chunks for better cache locality
    pub fn process_chunks<F>(mut self, processor: F) -> Vec<T>
    where
        F: Fn(&mut [T]) + Sync + Send,
    {
        self.data
            .par_chunks_mut(self.chunk_size)
            .for_each(|chunk| processor(chunk));
        
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    
    #[test]
    fn test_chunked_processor() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let processor = ChunkedProcessor::new(data, 100);
        
        let result = processor.process_chunks(|chunk| {
            for val in chunk.iter_mut() {
                *val *= 2.0;
            }
        });
        
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[999], 1998.0);
    }
    
    #[test]
    fn test_gradient_computer() {
        let mut array = Array3::<f64>::zeros((10, 10, 10));
        // Initialize with a simple function: f(x,y,z) = x + y + z
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    array[[i, j, k]] = (i + j + k) as f64;
                }
            }
        }
        
        let computer = GradientComputer::new(array.view());
        let gradient_count = std::sync::atomic::AtomicUsize::new(0);
        
        computer.compute_interior_gradients(1.0, 1.0, 1.0, |grad_x, grad_y, grad_z, _i, _j, _k| {
            // For f(x,y,z) = x + y + z, gradients should be (1, 1, 1)
            assert!((grad_x - 1.0).abs() < 1e-10);
            assert!((grad_y - 1.0).abs() < 1e-10);
            assert!((grad_z - 1.0).abs() < 1e-10);
            gradient_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        
        // Should process 8x8x8 = 512 interior points
        assert_eq!(gradient_count.load(std::sync::atomic::Ordering::Relaxed), 512);
    }
}
