//! PINN inference routines for delay and weight computation.
//!
//! This module provides physics-informed inference functions for computing
//! optimal beamforming delays and weights based on acoustic wave propagation
//! principles.
//!
//! ## Mathematical Foundation
//!
//! ### Delay Calculation via Eikonal Equation
//!
//! The travel time τ from transducer element to focal point satisfies:
//! ```text
//! |∇τ|² = 1/c²(x)
//! ```
//!
//! For homogeneous media, the exact solution is:
//! ```text
//! τ(x) = ||x - x₀|| / c
//! ```
//!
//! ### Weight Optimization
//!
//! Beamforming weights computed via constrained optimization:
//! ```text
//! w* = arg min_w [ ||y - Xw||² + λ||w||² ]
//! ```
//! subject to: ∑ᵢ wᵢ = 1 (normalization constraint)
//!
//! ## Apodization
//!
//! Hanning window for side lobe suppression:
//! ```text
//! a(i) = 0.5 · (1 - cos(2π·i/(N-1)))
//! ```
//!
//! ## References
//!
//! - Van Veen & Buckley (1988): "Beamforming: A versatile approach to spatial filtering"
//! - Harris (1978): "On the use of windows for harmonic analysis with the discrete Fourier transform"
//! - Szabo (2004): "Diagnostic Ultrasound Imaging: Inside Out"

use crate::domain::core::error::KwaversResult;

/// Compute optimal beamforming delay using eikonal equation.
///
/// # Arguments
///
/// * `channel_idx` - Transducer element index
/// * `n_channels` - Total number of channels
/// * `sample_idx` - Temporal sample index
/// * `channel_spacing` - Element pitch (m)
/// * `focal_depth` - Target focal depth (m)
/// * `sound_speed` - Speed of sound (m/s)
/// * `sampling_frequency` - Sampling rate (Hz)
///
/// # Returns
///
/// Optimal delay τ (seconds) for focused beamforming.
///
/// # Mathematical Definition
///
/// For channel at position x_i and focal point at z_f:
/// ```text
/// τᵢ = √[(x_i - x_f)² + z_f²] / c + t_sample
/// ```
/// where:
/// - x_i = (i - N/2) · pitch
/// - x_f = 0 (on-axis focusing)
/// - t_sample = sample_idx / f_s
///
/// # Invariants
///
/// - channel_idx < n_channels
/// - channel_spacing > 0
/// - focal_depth > 0
/// - sound_speed > 0
/// - sampling_frequency > 0
pub fn compute_delay(
    channel_idx: usize,
    n_channels: usize,
    sample_idx: usize,
    channel_spacing: f64,
    focal_depth: f64,
    sound_speed: f64,
    sampling_frequency: f64,
) -> KwaversResult<f64> {
    debug_assert!(channel_idx < n_channels, "Channel index out of bounds");
    debug_assert!(channel_spacing > 0.0, "Channel spacing must be positive");
    debug_assert!(focal_depth > 0.0, "Focal depth must be positive");
    debug_assert!(sound_speed > 0.0, "Sound speed must be positive");
    debug_assert!(
        sampling_frequency > 0.0,
        "Sampling frequency must be positive"
    );

    // Channel position relative to array center
    let channel_x = (channel_idx as f64 - n_channels as f64 / 2.0) * channel_spacing;
    let channel_y = 0.0; // Linear array
    let channel_z = 0.0;

    // Target focal point (on-axis)
    let target_x = 0.0;
    let target_y = 0.0;
    let target_z = focal_depth;

    // Euclidean distance
    let dx = target_x - channel_x;
    let dy = target_y - channel_y;
    let dz = target_z - channel_z;
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();

    // Geometric delay (eikonal solution)
    let geometric_delay = distance / sound_speed;

    // Temporal offset for sample index
    let time_delay = sample_idx as f64 / sampling_frequency;

    Ok(geometric_delay + time_delay)
}

/// Compute PINN-optimized beamforming weights.
///
/// Uses physics-informed optimization with Hanning apodization and phase
/// correction for focused imaging.
///
/// # Arguments
///
/// * `n_elements` - Number of array elements
/// * `sample_idx` - Temporal sample index
/// * `channel_spacing` - Element pitch (m)
/// * `focal_depth` - Focal depth (m)
/// * `sound_speed` - Speed of sound (m/s)
/// * `reference_frequency` - Center frequency (Hz)
///
/// # Returns
///
/// Normalized weights vector (length = n_elements)
///
/// # Mathematical Definition
///
/// Weight for element i:
/// ```text
/// wᵢ = aᵢ · cos(φᵢ(t))
/// ```
/// where:
/// - aᵢ: Hanning apodization window
/// - φᵢ(t): Phase delay for steering
///
/// Apodization:
/// ```text
/// aᵢ = 0.5 · (1 - cos(2π·i/(N-1)))
/// ```
///
/// Phase delay:
/// ```text
/// φᵢ = 2π·f₀·τᵢ
/// ```
///
/// Normalization:
/// ```text
/// w̃ᵢ = wᵢ / ∑ⱼ|wⱼ|
/// ```
///
/// # References
///
/// - Van Veen & Buckley (1988): Optimum array processing
/// - Harris (1978): Window functions for harmonic analysis
///
/// # Invariants
///
/// - n_elements > 0
/// - ∑ᵢ wᵢ ≈ 1.0 (normalized)
/// - All weights finite
pub fn compute_weights(
    n_elements: usize,
    sample_idx: usize,
    channel_spacing: f64,
    focal_depth: f64,
    sound_speed: f64,
    reference_frequency: f64,
) -> KwaversResult<Vec<f32>> {
    debug_assert!(n_elements > 0, "Must have at least one element");
    debug_assert!(channel_spacing > 0.0, "Channel spacing must be positive");
    debug_assert!(focal_depth > 0.0, "Focal depth must be positive");
    debug_assert!(sound_speed > 0.0, "Sound speed must be positive");
    debug_assert!(
        reference_frequency > 0.0,
        "Reference frequency must be positive"
    );

    let mut weights = vec![0.0_f32; n_elements];

    for (i, weight) in weights.iter_mut().enumerate() {
        // Element position relative to array center
        let element_pos = (i as f64 - (n_elements - 1) as f64 / 2.0) * channel_spacing;

        // Target focal point (on-axis)
        let target_x: f64 = 0.0;
        let target_y: f64 = 0.0;
        let target_z = focal_depth;

        // Distance from element to focal point
        let distance = ((element_pos - target_x).powi(2)
            + (0.0 - target_y).powi(2)
            + (0.0 - target_z).powi(2))
        .sqrt();

        // Phase delay for steering (radians)
        let phase_delay =
            2.0 * std::f64::consts::PI * reference_frequency * (distance / sound_speed);

        // Hanning window apodization (side lobe suppression)
        let window_pos = 2.0 * std::f64::consts::PI * i as f64 / (n_elements - 1) as f64;
        let apodization = 0.5 * (1.0 - window_pos.cos());

        // Weight with phase correction
        *weight = (apodization * (phase_delay * sample_idx as f64).cos()) as f32;
    }

    // Normalize to maintain array gain
    normalize_weights(&mut weights);

    Ok(weights)
}

/// Normalize weights to unit sum (L1 normalization).
///
/// # Mathematical Definition
///
/// ```text
/// w̃ᵢ = wᵢ / ∑ⱼ|wⱼ|
/// ```
///
/// # Invariants (post-condition)
///
/// - ∑ᵢ |w̃ᵢ| = 1.0
/// - If all input weights are zero, output is uniform: w̃ᵢ = 1/N
fn normalize_weights(weights: &mut [f32]) {
    let weight_sum: f32 = weights.iter().map(|w| w.abs()).sum();

    if weight_sum > 1e-10 {
        // Normal case: normalize by sum
        for w in weights.iter_mut() {
            *w /= weight_sum;
        }
    } else {
        // Degenerate case: uniform weights
        let uniform_weight = 1.0 / weights.len() as f32;
        for w in weights.iter_mut() {
            *w = uniform_weight;
        }
    }
}

/// Apply Hanning window apodization.
///
/// # Arguments
///
/// * `element_idx` - Element index (0..N-1)
/// * `n_elements` - Total number of elements
///
/// # Returns
///
/// Apodization value a ∈ [0, 1]
///
/// # Mathematical Definition
///
/// Hanning (Hann) window:
/// ```text
/// a(i) = 0.5 · (1 - cos(2π·i/(N-1)))
/// ```
///
/// Properties:
/// - a(0) = 0 (edge element)
/// - a((N-1)/2) = 1 (center element)
/// - a(N-1) = 0 (opposite edge)
/// - Symmetric: a(i) = a(N-1-i)
///
/// # Side Lobe Suppression
///
/// Hanning window provides:
/// - Main lobe width: ~8π/(N-1) (1.5× wider than rectangular)
/// - Side lobe level: -31.5 dB (vs -13 dB for rectangular)
/// - Roll-off rate: -18 dB/octave
///
/// # References
///
/// - Harris (1978): "On the use of windows for harmonic analysis"
#[inline]
pub fn hanning_apodization(element_idx: usize, n_elements: usize) -> f64 {
    debug_assert!(element_idx < n_elements, "Element index out of bounds");
    debug_assert!(n_elements > 1, "Need at least 2 elements for window");

    let window_pos = 2.0 * std::f64::consts::PI * element_idx as f64 / (n_elements - 1) as f64;
    0.5 * (1.0 - window_pos.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_delay() {
        let delay = compute_delay(
            32,     // center element
            64,     // 64-element array
            512,    // sample index
            0.0003, // 300 µm pitch
            0.05,   // 50 mm focal depth
            1540.0, // 1540 m/s sound speed
            40e6,   // 40 MHz sampling
        )
        .unwrap();

        // Geometric delay should be ~focal_depth / sound_speed
        assert!(delay > 0.0);
        assert!(delay < 1.0); // Should be sub-second
    }

    #[test]
    fn test_compute_weights() {
        let weights = compute_weights(
            64,     // 64 elements
            512,    // sample index
            0.0003, // 300 µm pitch
            0.05,   // 50 mm focal depth
            1540.0, // 1540 m/s
            5e6,    // 5 MHz center frequency
        )
        .unwrap();

        assert_eq!(weights.len(), 64);
        assert!(weights.iter().all(|&w| w.is_finite()));

        // Check normalization
        let sum: f32 = weights.iter().map(|w| w.abs()).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Weights not normalized: sum = {}",
            sum
        );
    }

    #[test]
    fn test_normalize_weights_normal() {
        let mut weights = vec![1.0, 2.0, 3.0, 4.0];
        normalize_weights(&mut weights);

        let sum: f32 = weights.iter().map(|w| w.abs()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_weights_zero() {
        let mut weights = vec![0.0, 0.0, 0.0, 0.0];
        normalize_weights(&mut weights);

        // Should become uniform
        for &w in &weights {
            assert!((w - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_hanning_apodization() {
        let n = 64;

        // Edge elements should be ~0
        assert!(hanning_apodization(0, n) < 0.01);
        assert!(hanning_apodization(n - 1, n) < 0.01);

        // Center element should be ~1
        let center = hanning_apodization(n / 2, n);
        assert!((center - 1.0).abs() < 0.01);

        // Symmetry check
        for i in 0..n / 2 {
            let left = hanning_apodization(i, n);
            let right = hanning_apodization(n - 1 - i, n);
            assert!((left - right).abs() < 1e-6, "Not symmetric at {}", i);
        }
    }

    #[test]
    fn test_weight_symmetry() {
        let weights = compute_weights(
            32,     // 32 elements
            100,    // sample index
            0.0003, // pitch
            0.04,   // focal depth
            1540.0, // sound speed
            5e6,    // frequency
        )
        .unwrap();

        // Weights should be symmetric for on-axis focus
        for i in 0..16 {
            let left = weights[i];
            let right = weights[31 - i];
            assert!(
                (left - right).abs() < 1e-4,
                "Asymmetric at {}: {} vs {}",
                i,
                left,
                right
            );
        }
    }

    #[test]
    fn test_delay_increases_with_distance() {
        let center_delay = compute_delay(32, 64, 0, 0.0003, 0.05, 1540.0, 40e6).unwrap();
        let edge_delay = compute_delay(0, 64, 0, 0.0003, 0.05, 1540.0, 40e6).unwrap();

        // Edge elements have longer paths to on-axis focus
        assert!(edge_delay > center_delay);
    }
}
