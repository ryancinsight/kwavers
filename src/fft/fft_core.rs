// src/fft/fft_core.rs
use num_complex::Complex;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    Forward,
    Inverse,
}

#[inline]
pub fn reverse_bits(mut x: u32, n: u32) -> u32 {
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    x = (x << 16) | (x >> 16);
    x >> (32 - n)
}

pub fn precompute_twiddles(n: usize, direction: FftDirection) -> Vec<Complex<f64>> {
    let angle_factor = match direction {
        FftDirection::Forward => -2.0 * PI / n as f64,
        FftDirection::Inverse => 2.0 * PI / n as f64,
    };
    (0..n / 2)
        .map(|i| {
            let angle = angle_factor * i as f64;
            Complex::new(angle.cos(), angle.sin())
        })
        .collect()
}