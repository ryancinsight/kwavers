// source/apodization/mod.rs

use kwavers_signal::{window_value, SignalWindowType};
use std::fmt::Debug;

pub trait Apodization: Debug + Sync + Send {
    /// Returns the apodization weight for a given position index and total elements.
    fn weight(&self, position_idx: usize, total_elements: usize) -> f64;
}

/// Shared symmetric-window weight: maps `position_idx ∈ [0, total_elements)` to the
/// normalized window position and evaluates the canonical `kwavers_signal` window.
/// Single arrays of one element get unit weight. Factored out so the per-window
/// apodization structs carry no duplicated boilerplate.
fn windowed_weight(kind: SignalWindowType, position_idx: usize, total_elements: usize) -> f64 {
    if total_elements <= 1 {
        return 1.0;
    }
    window_value(kind, position_idx as f64 / (total_elements - 1) as f64)
}

#[derive(Debug, Clone)]
pub struct HanningApodization;

impl Apodization for HanningApodization {
    fn weight(&self, position_idx: usize, total_elements: usize) -> f64 {
        windowed_weight(SignalWindowType::Hann, position_idx, total_elements)
    }
}

#[derive(Debug, Clone)]
pub struct HammingApodization;

impl Apodization for HammingApodization {
    fn weight(&self, position_idx: usize, total_elements: usize) -> f64 {
        windowed_weight(SignalWindowType::Hamming, position_idx, total_elements)
    }
}

#[derive(Debug, Clone)]
pub struct BlackmanApodization;

impl Apodization for BlackmanApodization {
    fn weight(&self, position_idx: usize, total_elements: usize) -> f64 {
        windowed_weight(SignalWindowType::Blackman, position_idx, total_elements)
    }
}

#[derive(Debug, Clone)]
pub struct GaussianApodization {
    sigma: f64, // Standard deviation for Gaussian spread (normalized to element spacing)
}

impl GaussianApodization {
    /// New.
    /// # Panics
    /// - Panics if assertion fails: `Sigma must be positive`.
    ///
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
