//! Optical properties implementation for heterogeneous media

use crate::medium::{
    heterogeneous::core::HeterogeneousMedium,
    optical::OpticalProperties,
};

impl OpticalProperties for HeterogeneousMedium {
    #[inline]
    fn absorption_coefficient(&self, i: usize, j: usize, k: usize) -> f64 {
        self.mu_a[[i, j, k]]
    }

    #[inline] 
    fn scattering_coefficient(&self, i: usize, j: usize, k: usize) -> f64 {
        self.mu_s_prime[[i, j, k]]
    }
}