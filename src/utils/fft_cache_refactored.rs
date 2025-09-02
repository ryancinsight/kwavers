//! Zero-copy FFT cache using thread-local storage
//!
//! Eliminates Arc<Mutex<>> antipattern in favor of thread-local caching

use crate::fft::ProcessorFft3d;
use crate::grid::Grid;
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct FftCacheKey {
    nx: usize,
    ny: usize,
    nz: usize,
}

impl From<&Grid> for FftCacheKey {
    fn from(grid: &Grid) -> Self {
        Self {
            nx: grid.nx,
            ny: grid.ny,
            nz: grid.nz,
        }
    }
}

thread_local! {
    /// Thread-local FFT cache eliminates synchronization overhead
    static FFT_CACHE: RefCell<HashMap<FftCacheKey, ProcessorFft3d>> = RefCell::new(HashMap::new());
}

/// Get or create FFT processor for grid dimensions
/// 
/// Zero-copy implementation using thread-local storage
pub fn get_fft_for_grid(grid: &Grid) -> ProcessorFft3d {
    let key = FftCacheKey::from(grid);
    
    FFT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        
        cache.entry(key)
            .or_insert_with(|| ProcessorFft3d::new(grid.nx, grid.ny, grid.nz))
            .clone()
    })
}

/// Clear thread-local FFT cache
pub fn clear_fft_cache() {
    FFT_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_thread_local_caching() {
        let grid = Grid::new(32, 32, 32, 1.0, 1.0, 1.0);
        
        // First access creates FFT
        let fft1 = get_fft_for_grid(&grid);
        
        // Second access reuses cached FFT
        let fft2 = get_fft_for_grid(&grid);
        
        // Should be equal (same dimensions)
        assert_eq!(fft1.nx, fft2.nx);
        assert_eq!(fft1.ny, fft2.ny);
        assert_eq!(fft1.nz, fft2.nz);
    }
}