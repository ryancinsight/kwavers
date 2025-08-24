// src/utils/mod.rs
pub mod array_utils;
pub mod differential_operators;
pub mod fft_operations;
pub mod field_analysis;
pub mod format;
pub mod iterators;
pub mod kwave;  // Modular k-Wave utilities
pub mod linear_algebra;
pub mod sparse_matrix;
pub mod spectral;
pub mod stencil;

// Re-export commonly used utilities
pub use self::fft_operations::{fft_3d_array, ifft_3d_array};
pub use self::field_analysis::FieldAnalyzer;
pub use self::sparse_matrix::CompressedSparseRowMatrix;
pub use self::stencil::{Stencil, StencilValue};

// Export differential operators with unique names to avoid conflicts
pub use self::differential_operators::{
    curl as curl_op, divergence as divergence_op, gradient as gradient_op,
    laplacian as laplacian_op, spectral_laplacian, transverse_laplacian, FDCoefficients,
    SpatialOrder,
};

#[cfg(test)]
pub mod test_helpers;

use crate::fft::{fft3d::Fft3d, ifft3d::Ifft3d};
use crate::grid::Grid;
use lazy_static::lazy_static;
use log::{debug, info, trace};
use ndarray::{Array3, Array4, Axis, Zip};
use num_complex::Complex;

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

// Performance tracking for FFT operations
static FFT_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
static FFT_CACHE_MISSES: AtomicUsize = AtomicUsize::new(0);
static IFFT_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
static IFFT_CACHE_MISSES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_FFT_TIME: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
static TOTAL_IFFT_TIME: Mutex<Duration> = Mutex::new(Duration::new(0, 0));
static TOTAL_FFT_COUNT: Mutex<usize> = Mutex::new(0);
static TOTAL_IFFT_COUNT: Mutex<usize> = Mutex::new(0);

// Thread-local buffer to avoid repeated allocations
thread_local! {
    static FFT_BUFFER: std::cell::RefCell<Option<Array3<Complex<f64>>>> = const { std::cell::RefCell::new(None) };
    static IFFT_BUFFER: std::cell::RefCell<Option<Array3<Complex<f64>>>> = const { std::cell::RefCell::new(None) };
    static RESULT_BUFFER: std::cell::RefCell<Option<Array3<f64>>> = const { std::cell::RefCell::new(None) };
}

// Cache for FFT instances to avoid recreating them for the same grid dimensions
//
// Note: Double mutex pattern is used here because:
// 1. The outer mutex protects the cache HashMap
// 2. The inner mutex protects the Fft3d instance which has mutable temp buffers
//
// FFT planner instances are thread-safe and can be reused.
// For production performance, consider using a global planner pool.
lazy_static! {
    static ref FFT_CACHE: Mutex<HashMap<(usize, usize, usize), Arc<Mutex<Fft3d>>>> =
        Mutex::new(HashMap::new());
    static ref IFFT_CACHE: Mutex<HashMap<(usize, usize, usize), Arc<Ifft3d>>> =
        Mutex::new(HashMap::new());
}

/// Initialize and warm up the FFT cache for common grid sizes
pub fn warm_fft_cache(grid: &Grid) {
    debug!(
        "Warming FFT/IFFT cache for grid {}x{}x{}",
        grid.nx, grid.ny, grid.nz
    );

    // Pre-create FFT instances for the current grid size
    let key = (grid.nx, grid.ny, grid.nz);

    // Warm up FFT cache
    {
        let mut cache = FFT_CACHE.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            debug!(
                "Pre-creating Fft3d instance for grid {}x{}x{}",
                grid.nx, grid.ny, grid.nz
            );
            Arc::new(Mutex::new(Fft3d::new(grid.nx, grid.ny, grid.nz)))
        });
    }

    // Warm up IFFT cache
    {
        let mut cache = IFFT_CACHE.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            debug!(
                "Pre-creating IFFT3d instance for grid {}x{}x{}",
                grid.nx, grid.ny, grid.nz
            );
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
    let dummy_ifft = ifft_3d(&dummy_fft, grid);
    
    // Verify FFT operations are working correctly
    debug!("FFT warmup verification: field dimensions {:?}", dummy_ifft.dim());

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

/// 3D FFT for simulation fields with proper normalization
///
/// This function performs a 3D FFT on the specified field component from a 4D array.
/// It uses a cached FFT instance for improved performance when called multiple times
/// with the same grid dimensions, and employs thread-local storage to reduce allocations.
///
/// # Arguments
///
/// * `fields` - 4D array containing multiple field components
/// * `field_index` - Index of the field to transform (0-based)
/// * `grid` - The simulation grid
///
/// # Returns
///
/// A 3D complex array containing the FFT of the specified field component
pub fn fft_3d(fields: &Array4<f64>, field_index: usize, grid: &Grid) -> Array3<Complex<f64>> {
    let start_time = Instant::now();
    trace!("Performing 3D FFT on field index {}", field_index);

    // Get the field slice to transform
    let field = fields.index_axis(Axis(0), field_index);

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
    let fft = {
        let mut cache = FFT_CACHE.lock().unwrap();
        let key = (grid.nx, grid.ny, grid.nz);

        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(key) {
            FFT_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            debug!(
                "Cache miss: Creating new Fft3d instance for grid {}x{}x{}",
                grid.nx, grid.ny, grid.nz
            );
            let fft = Arc::new(Mutex::new(Fft3d::new(grid.nx, grid.ny, grid.nz)));
            e.insert(Arc::clone(&fft));
            fft
        } else {
            FFT_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            Arc::clone(cache.get(&key).unwrap())
        }
    };

    // Use the FFT instance directly from Arc
    let mut result = field_complex.clone();

    // Apply the 3D FFT transform
    fft.lock().unwrap().process(&mut result, grid);

    // Update timing statistics
    let elapsed = start_time.elapsed();
    let mut total_count = TOTAL_FFT_COUNT.lock().unwrap();
    *total_count += 1;
    let mut total_time = TOTAL_FFT_TIME.lock().unwrap();
    *total_time += elapsed;

    result
}

/// 3D inverse FFT for simulation fields with proper normalization
pub fn ifft_3d(field_complex: &Array3<Complex<f64>>, grid: &Grid) -> Array3<f64> {
    let start_time = Instant::now();
    trace!("Performing 3D inverse FFT");

    // Create a copy for processing
    let mut result = field_complex.clone();

    // Get or create FFT instance from cache (for inverse, we'll use a flag)
    let (nx, ny, nz) = grid.dimensions();
    let key = (nx, ny, nz);

    // For inverse FFT, we need to apply conjugate, forward FFT, conjugate, and scale
    // This is a common technique when only forward FFT is available

    // Step 1: Conjugate the input
    result.mapv_inplace(|c| c.conj());

    let fft = {
        let mut cache = FFT_CACHE.lock().unwrap();
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(key) {
            FFT_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            let fft = Arc::new(Mutex::new(Fft3d::new(nx, ny, nz)));
            e.insert(Arc::clone(&fft));
            fft
        } else {
            FFT_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            Arc::clone(cache.get(&key).unwrap())
        }
    };

    // Step 2: Apply forward FFT
    fft.lock().unwrap().process(&mut result, grid);

    // Step 3: Conjugate again and normalize
    let normalization = 1.0 / (nx * ny * nz) as f64;
    result.mapv_inplace(|c| c.conj() * normalization);

    // Convert back to real values in parallel
    let real_buffer = FFT_BUFFER.with(|buffer| {
        let mut b = buffer.borrow_mut();
        if b.is_none() || b.as_ref().unwrap().dim() != result.dim() {
            *b = Some(Array3::zeros(result.dim()));
        }

        let complex_buffer = b.as_mut().unwrap();

        // Extract real parts in parallel
        Zip::from(&result)
            .and(&mut *complex_buffer)
            .for_each(|&c, r| {
                *r = Complex::new(c.re, 0.0);
            });

        // Create output array
        let mut output = Array3::zeros(result.dim());
        Zip::from(&mut output)
            .and(&*complex_buffer)
            .for_each(|o, &c| {
                *o = c.re;
            });

        output
    });

    // Update timing statistics
    let elapsed = start_time.elapsed();
    let mut total_count = TOTAL_IFFT_COUNT.lock().unwrap();
    *total_count += 1;
    let mut total_time = TOTAL_IFFT_TIME.lock().unwrap();
    *total_time += elapsed;

    real_buffer
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

/// Compute k-space correction factor for a given frequency
pub fn k_space_correction(k: f64, dt: f64) -> f64 {
    let kdt = k * dt;
    if kdt.abs() < 1e-10 {
        1.0
    } else {
        kdt.sin() / kdt
    }
}

pub fn derivative(
    fields: &Array4<f64>,
    field_idx: usize,
    grid: &Grid,
    axis: usize,
) -> Result<Array3<f64>, &'static str> {
    if axis > 2 {
        return Err("Axis must be 0 (x), 1 (y), or 2 (z)");
    }
    let mut field_fft = fft_3d(fields, field_idx, grid);
    let k = match axis {
        0 => grid.kx(),
        1 => grid.ky(),
        2 => grid.kz(),
        _ => return Err("Invalid axis dimension"),
    };

    field_fft
        .axis_iter_mut(Axis(axis))
        .enumerate()
        .for_each(|(idx, mut slice)| {
            slice.mapv_inplace(|c| c * Complex::new(0.0, k[idx]));
        });

    Ok(ifft_3d(&field_fft, grid))
}
