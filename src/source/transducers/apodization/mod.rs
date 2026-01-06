// source/apodization/mod.rs

use crate::signal::{window_value, WindowType};
use std::fmt::Debug;

pub trait Apodization: Debug + Sync + Send {
    /// Returns the apodization weight for a given position index and total elements.
    fn weight(&self, position_idx: usize, total_elements: usize) -> f64;
}

#[derive(Debug, Clone)]
pub struct HanningApodization;

impl Apodization for HanningApodization {
    fn weight(&self, position_idx: usize, total_elements: usize) -> f64 {
        if total_elements <= 1 {
            return 1.0;
        }
        window_value(
            WindowType::Hann,
            position_idx as f64 / (total_elements - 1) as f64,
        )
    }
}

#[derive(Debug, Clone)]
pub struct HammingApodization;

impl Apodization for HammingApodization {
    fn weight(&self, position_idx: usize, total_elements: usize) -> f64 {
        if total_elements <= 1 {
            return 1.0;
        }
        window_value(
            WindowType::Hamming,
            position_idx as f64 / (total_elements - 1) as f64,
        )
    }
}

#[derive(Debug, Clone)]
pub struct BlackmanApodization;

impl Apodization for BlackmanApodization {
    fn weight(&self, position_idx: usize, total_elements: usize) -> f64 {
        if total_elements <= 1 {
            return 1.0;
        }
        window_value(
            WindowType::Blackman,
            position_idx as f64 / (total_elements - 1) as f64,
        )
    }
}

#[derive(Debug, Clone)]
pub struct GaussianApodization {
    sigma: f64, // Standard deviation for Gaussian spread (normalized to element spacing)
}

impl GaussianApodization {
    #[must_use]
    pub fn new(sigma: f64) -> Self {
        assert!(sigma > 0.0, "Sigma must be positive");
        Self { sigma }
    }
}

impl Apodization for GaussianApodization {
    fn weight(&self, position_idx: usize, total_elements: usize) -> f64 {
        if total_elements <= 1 {
            return 1.0;
        }
        // Map Gaussian to window_value but handle sigma specifically if needed.
        // For now, consistent with signal::window's Gaussian implementation if we want.
        // But source usually defines sigma in element indices.
        let center = (total_elements - 1) as f64 / 2.0;
        let x = position_idx as f64 - center;
        (-x * x / (2.0 * self.sigma * self.sigma)).exp()
    }
}

#[derive(Debug, Clone)]
pub struct RectangularApodization;

impl Apodization for RectangularApodization {
    fn weight(&self, _position_idx: usize, _total_elements: usize) -> f64 {
        1.0 // Uniform weighting
    }
}
