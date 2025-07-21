// src/fft/fft_core.rs
use num_complex::Complex;
use std::f64::consts::PI;
use log::trace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    Forward,
    Inverse,
}

/// Reverses the bits of a number, used for FFT reordering
/// This is an optimized version using bit manipulation tricks
#[inline]
pub fn reverse_bits(mut x: u32, n: u32) -> u32 {
    // Use a lookup table for small values to improve performance
    if n <= 8 {
        const LOOKUP: [u8; 256] = [
            0, 128, 64, 192, 32, 160, 96, 224, 16, 144, 80, 208, 48, 176, 112, 240,
            8, 136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
            4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244,
            12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
            2, 130, 66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242,
            10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
            6, 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246,
            14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
            1, 129, 65, 193, 33, 161, 97, 225, 17, 145, 81, 209, 49, 177, 113, 241,
            9, 137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
            5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245,
            13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
            3, 131, 67, 195, 35, 163, 99, 227, 19, 147, 83, 211, 51, 179, 115, 243,
            11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
            7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247,
            15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255,
        ];
        return (LOOKUP[(x & 0xFF) as usize] as u32) >> (8 - n);
    }

    // For larger values, use the optimized bit manipulation approach
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    x = x.rotate_right(16);
    x >> (32 - n)
}

/// Precomputes twiddle factors for FFT to avoid redundant calculations
/// Uses a more efficient approach with sin/cos tables
pub fn precompute_twiddles(n: usize, direction: FftDirection) -> Vec<Complex<f64>> {
    trace!("Precomputing twiddle factors for size {}", n);
    
    let angle_factor = match direction {
        FftDirection::Forward => -2.0 * PI / n as f64,
        FftDirection::Inverse => 2.0 * PI / n as f64,
    };
    
    // Precompute sin/cos values for better cache locality
    let mut twiddles = Vec::with_capacity(n / 2);
    
    // Use SIMD-friendly iteration pattern
    for i in 0..n / 2 {
        let angle = angle_factor * i as f64;
        // Use sincos when available for better performance
        #[cfg(feature = "nightly")]
        let (sin_val, cos_val) = angle.sin_cos();
        #[cfg(not(feature = "nightly"))]
        let (sin_val, cos_val) = (angle.sin(), angle.cos());
        
        twiddles.push(Complex::new(cos_val, sin_val));
    }
    
    twiddles
}

/// Computes the number of bits needed to represent a value
#[inline]
pub fn log2_ceil(n: usize) -> u32 {
    if n.is_power_of_two() {
        n.trailing_zeros()
    } else {
        n.next_power_of_two().trailing_zeros()
    }
}

/// Checks if a number is a power of two
#[inline]
pub fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Calculates the next power of two for a given number
#[inline]
pub fn next_power_of_two_usize(n: usize) -> usize {
    if is_power_of_two(n) {
        n
    } else {
        n.next_power_of_two()
    }
}