// src/utils/iterators.rs
//! Zero-cost iterator abstractions for efficient physics simulations
//! 
//! This module provides iterator-based patterns that leverage Rust's zero-cost abstractions
//! to achieve high performance while maintaining readable and maintainable code.

use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use rayon::prelude::*;

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
    
    /// Get reference to element at position
    pub fn get(&self, i: usize, j: usize, k: usize) -> Option<&T> {
        if i < self.nx && j < self.ny && k < self.nz {
            Some(&self.array[[i, j, k]])
        } else {
            None
        }
    }
    
    /// Get mutable reference to element at position
    pub fn get_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut T> {
        if i < self.nx && j < self.ny && k < self.nz {
            Some(&mut self.array[[i, j, k]])
        } else {
            None
        }
    }
    
    /// Process interior points sequentially to avoid borrowing conflicts
    pub fn process_interior_sequential<F>(&mut self, dx: f64, dy: f64, dz: f64, mut processor: F)
    where
        F: FnMut(usize, usize, usize, f64, f64, f64, &mut T),
    {
        for i in 1..self.nx-1 {
            for j in 1..self.ny-1 {
                for k in 1..self.nz-1 {
                    let x = i as f64 * dx;
                    let y = j as f64 * dy;
                    let z = k as f64 * dz;
                    processor(i, j, k, x, y, z, &mut self.array[[i, j, k]]);
                }
            }
        }
    }
}

/// Iterator for chunked processing with cache-friendly access patterns
pub struct ChunkedProcessor<'a, T> {
    array: ArrayViewMut3<'a, T>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl<'a, T> ChunkedProcessor<'a, T>
where
    T: Sync + Send,
{
    pub fn new(array: ArrayViewMut3<'a, T>) -> Self {
        let (nx, ny, nz) = array.dim();
        Self { array, nx, ny, nz }
    }
    
    /// Process interior points with chunked iteration for cache performance
    pub fn process_interior<F>(&mut self, dx: f64, dy: f64, dz: f64, processor: F)
    where
        F: Fn(usize, usize, usize, f64, f64, f64) + Sync + Send,
    {
        // Use sequential processing to avoid borrowing conflicts
        for i in 1..self.nx-1 {
            for j in 1..self.ny-1 {
                for k in 1..self.nz-1 {
                    let x = i as f64 * dx;
                    let y = j as f64 * dy;
                    let z = k as f64 * dz;
                    processor(i, j, k, x, y, z);
                }
            }
        }
    }
    
    /// Get mutable reference to element at position
    pub fn get_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut T> {
        if i < self.nx && j < self.ny && k < self.nz {
            Some(&mut self.array[[i, j, k]])
        } else {
            None
        }
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
    pub fn compute_interior_gradients<F>(&self, dx: f64, dy: f64, dz: f64, processor: F)
    where
        F: Fn(f64, f64, f64, usize, usize, usize) + Sync + Send,
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
    
    /// Collect gradients into a result array
    pub fn collect_gradients(&self, dx: f64, dy: f64, dz: f64) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let mut grad_x = Array3::zeros((self.nx, self.ny, self.nz));
        let mut grad_y = Array3::zeros((self.nx, self.ny, self.nz));
        let mut grad_z = Array3::zeros((self.nx, self.ny, self.nz));
        
        let dx_inv = 1.0 / (2.0 * dx);
        let dy_inv = 1.0 / (2.0 * dy);
        let dz_inv = 1.0 / (2.0 * dz);
        
        // Sequential computation for interior points
        for i in 1..self.nx-1 {
            for j in 1..self.ny-1 {
                for k in 1..self.nz-1 {
                    grad_x[[i, j, k]] = (self.array[[i+1, j, k]] - self.array[[i-1, j, k]]) * dx_inv;
                    grad_y[[i, j, k]] = (self.array[[i, j+1, k]] - self.array[[i, j-1, k]]) * dy_inv;
                    grad_z[[i, j, k]] = (self.array[[i, j, k+1]] - self.array[[i, j, k-1]]) * dz_inv;
                }
            }
        }
            
        (grad_x, grad_y, grad_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    
    #[test]
    fn test_chunked_processor() {
        // Create a 3D array for testing
        let mut data = Array3::<f64>::zeros((10, 10, 10));
        
        // Fill with test data
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    data[[i, j, k]] = (i * 100 + j * 10 + k) as f64;
                }
            }
        }
        
        // Test chunked processor
        let mut processor = ChunkedProcessor::new(data.view_mut());
        
        // Test element access
        assert_eq!(processor.get_mut(1, 2, 3), Some(&mut 123.0));
        
        // Test interior processing 
        processor.process_interior(1.0, 1.0, 1.0, |i, j, k, x, y, z| {
            // Just verify the coordinates are as expected
            assert_eq!(x, i as f64);
            assert_eq!(y, j as f64);
            assert_eq!(z, k as f64);
        });
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
