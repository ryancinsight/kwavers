// src/utils/mod.rs
pub mod iterators;

use crate::fft::{fft3d::Fft3d, ifft3d::Ifft3d};
use crate::grid::Grid;
use log::{debug, trace, info};
use ndarray::{Array3, Array4, Axis, Zip};
use rayon::prelude::*;
use num_complex::Complex;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::time::{Instant, Duration};

// Performance tracking for FFT operations
static FFT_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
static FFT_CACHE_MISSES: AtomicUsize = AtomicUsize::new(0);
static IFFT_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
static IFFT_CACHE_MISSES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_FFT_TIME: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
static TOTAL_IFFT_TIME: Mutex<Duration> = Mutex::new(Duration::new(0, 0));

// Thread-local buffer to avoid repeated allocations
thread_local! {
    static FFT_BUFFER: std::cell::RefCell<Option<Array3<Complex<f64>>>> = const { std::cell::RefCell::new(None) };
    static IFFT_BUFFER: std::cell::RefCell<Option<Array3<Complex<f64>>>> = const { std::cell::RefCell::new(None) };
    static RESULT_BUFFER: std::cell::RefCell<Option<Array3<f64>>> = const { std::cell::RefCell::new(None) };
}

// Cache for FFT instances to avoid recreating them for the same grid dimensions
lazy_static! {
    static ref FFT_CACHE: Mutex<HashMap<(usize, usize, usize), Arc<Fft3d>>> = Mutex::new(HashMap::new());
    static ref IFFT_CACHE: Mutex<HashMap<(usize, usize, usize), Arc<Ifft3d>>> = Mutex::new(HashMap::new());
}

/// Initialize and warm up the FFT cache for common grid sizes
pub fn warm_fft_cache(grid: &Grid) {
    debug!("Warming FFT/IFFT cache for grid {}x{}x{}", grid.nx, grid.ny, grid.nz);
    
    // Pre-create FFT instances for the current grid size
    let key = (grid.nx, grid.ny, grid.nz);
    
    // Warm up FFT cache
    {
        let mut cache = FFT_CACHE.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            debug!("Pre-creating FFT3d instance for grid {}x{}x{}", grid.nx, grid.ny, grid.nz);
            Arc::new(Fft3d::new(grid.nx, grid.ny, grid.nz))
        });
    }
    
    // Warm up IFFT cache
    {
        let mut cache = IFFT_CACHE.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            debug!("Pre-creating IFFT3d instance for grid {}x{}x{}", grid.nx, grid.ny, grid.nz);
            Arc::new(Ifft3d::new(grid.nx, grid.ny, grid.nz))
        });
    }
    
    // Initialize thread-local buffers for common grid size
    FFT_BUFFER.with(|buffer| {
        let mut b = buffer.borrow_mut();
        *b = Some(Array3::zeros((grid.nx, grid.ny, grid.nz)));
    });
    
    IFFT_BUFFER.with(|buffer| {
        let mut b = buffer.borrow_mut();
        *b = Some(Array3::zeros((grid.nx, grid.ny, grid.nz)));
    });
    
    RESULT_BUFFER.with(|buffer| {
        let mut b = buffer.borrow_mut();
        *b = Some(Array3::zeros((grid.nx, grid.ny, grid.nz)));
    });
    
    // Create a dummy field and perform a warm-up transform
    let dummy_field = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
    let dummy_fft = fft_3d(&dummy_field, 0, grid);
    let _dummy_ifft = ifft_3d(&dummy_fft, grid);
    
    debug!("FFT/IFFT cache warm-up complete");
}

/// Report FFT performance statistics
pub fn report_fft_statistics() {
    let fft_hits = FFT_CACHE_HITS.load(Ordering::Relaxed);
    let fft_misses = FFT_CACHE_MISSES.load(Ordering::Relaxed);
    let ifft_hits = IFFT_CACHE_HITS.load(Ordering::Relaxed);
    let ifft_misses = IFFT_CACHE_MISSES.load(Ordering::Relaxed);
    
    let fft_total = fft_hits + fft_misses;
    let ifft_total = ifft_hits + ifft_misses;
    
    let fft_hit_rate = if fft_total > 0 {
        100.0 * fft_hits as f64 / fft_total as f64
    } else {
        0.0
    };
    
    let ifft_hit_rate = if ifft_total > 0 {
        100.0 * ifft_hits as f64 / ifft_total as f64
    } else {
        0.0
    };
    
    let total_fft_time = TOTAL_FFT_TIME.lock().unwrap();
    let total_ifft_time = TOTAL_IFFT_TIME.lock().unwrap();
    
    info!(
        "FFT Performance: Hits: {}, Misses: {}, Hit Rate: {:.2}%, Total Time: {:.2?}",
        fft_hits, fft_misses, fft_hit_rate, *total_fft_time
    );
    
    info!(
        "IFFT Performance: Hits: {}, Misses: {}, Hit Rate: {:.2}%, Total Time: {:.2?}",
        ifft_hits, ifft_misses, ifft_hit_rate, *total_ifft_time
    );
}

/// Optimized 3D FFT for simulation fields
/// 
/// This function performs a 3D FFT on a specific field in the simulation.
/// It uses a cached FFT instance for better performance when called multiple times
/// with the same grid dimensions, and employs thread-local storage to reduce allocations.
///
/// # Arguments
///
/// * `fields` - The 4D array containing all simulation fields
/// * `field_idx` - The index of the field to transform
/// * `grid` - The simulation grid
///
/// # Returns
///
/// A 3D complex array containing the FFT of the specified field
pub fn fft_3d(fields: &Array4<f64>, field_idx: usize, grid: &Grid) -> Array3<Complex<f64>> {
    let start_time = Instant::now();
    trace!("Performing optimized 3D FFT on field {}", field_idx);
    
    // Get field to transform
    let field = fields.index_axis(Axis(0), field_idx);
    
    // Get or create the thread-local buffer
    let field_complex = FFT_BUFFER.with(|buffer| {
        let mut b = buffer.borrow_mut();
        if b.is_none() || b.as_ref().unwrap().dim() != field.dim() {
            // First use or dimensions changed, create new buffer
            *b = Some(Array3::zeros(field.dim()));
        }
        
        // Get a mutable reference to the buffer
        let complex_buffer = b.as_mut().unwrap();
        
        // Convert real field to complex in parallel
        Zip::from(&mut *complex_buffer)
            .and(&field)
            .for_each(|c, &r| {
                *c = Complex::new(r, 0.0);
            });
        
        // Return the filled buffer
        complex_buffer.clone()
    });
    
    // Get or create FFT instance from cache
    let fft_arc = {
        let mut cache = FFT_CACHE.lock().unwrap();
        let key = (grid.nx, grid.ny, grid.nz);
        
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(key) {
            FFT_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            debug!("Cache miss: Creating new FFT3d instance for grid {}x{}x{}", grid.nx, grid.ny, grid.nz);
            let fft = Arc::new(Fft3d::new(grid.nx, grid.ny, grid.nz));
            e.insert(Arc::clone(&fft));
            fft
        } else {
            FFT_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            Arc::clone(cache.get(&key).unwrap())
        }
    };
    
    // Get a mutable clone of the FFT instance
    let mut fft = (*fft_arc).clone();
    
    // Use the buffer for the FFT operation
    let mut result = field_complex.clone();
    
    // Process the FFT
    fft.process(&mut result, grid);
    
    // Update timing statistics
    let elapsed = start_time.elapsed();
    let mut total_time = TOTAL_FFT_TIME.lock().unwrap();
    *total_time += elapsed;
    
    result
}

/// Optimized 3D inverse FFT for simulation fields
/// 
/// This function performs a 3D inverse FFT on a complex field in the simulation.
/// It uses a cached IFFT instance for better performance when called multiple times
/// with the same grid dimensions, and employs thread-local storage to reduce allocations.
///
/// # Arguments
///
/// * `field` - The complex field to transform
/// * `grid` - The simulation grid
///
/// # Returns
///
/// A 3D real array containing the inverse FFT of the input field
pub fn ifft_3d(field: &Array3<Complex<f64>>, grid: &Grid) -> Array3<f64> {
    let start_time = Instant::now();
    trace!("Performing optimized 3D IFFT");
    
    // Get or create the thread-local buffer
    let mut field_complex = IFFT_BUFFER.with(|buffer| {
        let mut b = buffer.borrow_mut();
        if b.is_none() || b.as_ref().unwrap().dim() != field.dim() {
            // First use or dimensions changed, create new buffer
            *b = Some(Array3::zeros(field.dim()));
        }
        
        // Get a mutable reference to the buffer
        let complex_buffer = b.as_mut().unwrap();
        
        // Copy input field to buffer in parallel
        Zip::from(&mut *complex_buffer)
            .and(field)
            .for_each(|dst, &src| {
                *dst = src;
            });
        
        // Return the filled buffer
        complex_buffer.clone()
    });
    
    // Get or create IFFT instance from cache
    let ifft_arc = {
        let mut cache = IFFT_CACHE.lock().unwrap();
        let key = (grid.nx, grid.ny, grid.nz);
        
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(key) {
            IFFT_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            debug!("Cache miss: Creating new IFFT3d instance for grid {}x{}x{}", grid.nx, grid.ny, grid.nz);
            let ifft = Arc::new(Ifft3d::new(grid.nx, grid.ny, grid.nz));
            e.insert(Arc::clone(&ifft));
            ifft
        } else {
            IFFT_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            Arc::clone(cache.get(&key).unwrap())
        }
    };
    
    // Get a mutable clone of the IFFT instance
    let mut ifft = (*ifft_arc).clone();
    
    // Process the IFFT and get the result
    let result = ifft.process(&mut field_complex, grid);
    
    // Update timing statistics
    let elapsed = start_time.elapsed();
    let mut total_time = TOTAL_IFFT_TIME.lock().unwrap();
    *total_time += elapsed;
    
    result
}

/// Utility function to check if a value is a power of two
#[inline]
pub fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Utility function to get the next power of two for a value
#[inline]
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    
    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}

/// Utility function to calculate the log2 ceiling of a value
#[inline]
pub fn log2_ceil(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    
    let mut log = 0;
    let mut value = 1;
    while value < n {
        value *= 2;
        log += 1;
    }
    log
}

pub fn laplacian(fields: &Array4<f64>, field_idx: usize, grid: &Grid) -> Result<Array3<f64>, &'static str> {
    debug!("Computing Laplacian for field {}", field_idx);
    let mut field_fft = fft_3d(fields, field_idx, grid);
    let k2 = grid.k_squared();

    // Multiply by -k² in parallel for better performance
    Zip::from(&mut field_fft)
        .and(&k2)
        .for_each(|f, &k_val| {
            *f *= Complex::new(-k_val, 0.0);
        });

    let result = ifft_3d(&field_fft, grid);
    Ok(result)
}

pub fn derivative(fields: &Array4<f64>, field_idx: usize, grid: &Grid, axis: usize) -> Result<Array3<f64>, &'static str> {
    if axis > 2 { return Err("Axis must be 0 (x), 1 (y), or 2 (z)"); }
    let mut field_fft = fft_3d(fields, field_idx, grid);
    let k = match axis {
        0 => grid.kx(),
        1 => grid.ky(),
        2 => grid.kz(),
        _ => unreachable!(),
    };

    field_fft
        .axis_iter_mut(Axis(axis))
        .enumerate()
        .for_each(|(idx, mut slice)| {
            slice.mapv_inplace(|c| c * Complex::new(0.0, k[idx]));
        });

    Ok(ifft_3d(&field_fft, grid))
}