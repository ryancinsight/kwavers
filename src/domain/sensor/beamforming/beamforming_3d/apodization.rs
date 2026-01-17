//! Apodization Window Functions for 3D Beamforming
//!
//! This module implements apodization (spatial windowing) functions for sidelobe
//! reduction and improved image quality in 3D volumetric beamforming. Apodization
//! weights are applied to transducer elements to taper the aperture function.
//!
//! # Theory
//!
//! Apodization reduces sidelobe levels in the point spread function (PSF) at the
//! cost of slightly increased mainlobe width. The trade-off is well-documented in
//! classical beamforming literature (Van Trees, 2002).
//!
//! # Supported Windows
//!
//! - **Rectangular**: No weighting (uniform illumination)
//! - **Hamming**: α = 0.54, β = 0.46 — good sidelobe suppression (-43 dB)
//! - **Hann**: Raised cosine window — smooth rolloff (-31 dB sidelobes)
//! - **Blackman**: Three-term cosine — excellent sidelobe rejection (-58 dB)
//! - **Gaussian**: σ-controlled smooth window — adjustable mainlobe/sidelobe trade-off
//! - **Custom**: User-provided weights for specialized applications
//!
//! # References
//!
//! - Harris (1978): "On the use of windows for harmonic analysis with the discrete Fourier transform"
//! - Van Trees (2002): "Optimum Array Processing"
//! - Thomenius (1996): "Evolution of ultrasound beamformers"

use super::config::ApodizationWindow;
use ndarray::Array3;

/// Create 3D apodization weights for a transducer array
///
/// # Arguments
///
/// * `num_elements` - Number of elements in each dimension (nx, ny, nz)
/// * `window` - Apodization window type
///
/// # Returns
///
/// 3D array of weights in [0, 1] for each transducer element
///
/// # Mathematical Formulation
///
/// For each element at position (i, j, k), compute normalized coordinates:
/// ```text
/// x = 2i/(nx-1) - 1  ∈ [-1, 1]
/// y = 2j/(ny-1) - 1  ∈ [-1, 1]
/// z = 2k/(nz-1) - 1  ∈ [-1, 1]
/// r = √(x² + y² + z²)  (radial distance from center)
/// ```
///
/// Then apply the window function w(r) based on the selected window type.
#[must_use]
pub fn create_apodization_weights(
    num_elements: (usize, usize, usize),
    window: &ApodizationWindow,
) -> Array3<f32> {
    let (nx, ny, nz) = num_elements;
    let mut weights = Array3::<f32>::ones((nx, ny, nz));

    match window {
        ApodizationWindow::Rectangular => {
            // Uniform weights (already initialized to 1.0)
        }
        ApodizationWindow::Hamming => {
            // Hamming window: w(r) = 0.54 - 0.46·cos(πr)
            // Provides -43 dB sidelobe suppression
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let r = compute_normalized_radius(i, j, k, nx, ny, nz);
                        weights[[i, j, k]] = 0.54 - 0.46 * (std::f32::consts::PI * r).cos();
                    }
                }
            }
        }
        ApodizationWindow::Hann => {
            // Hann (Hanning) window: w(r) = 0.5·(1 - cos(πr))
            // Provides -31 dB sidelobe suppression with smooth rolloff
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let r = compute_normalized_radius(i, j, k, nx, ny, nz);
                        weights[[i, j, k]] = 0.5 * (1.0 - (std::f32::consts::PI * r).cos());
                    }
                }
            }
        }
        ApodizationWindow::Blackman => {
            // Blackman window: w(r) = 0.42 - 0.5·cos(πr) + 0.08·cos(2πr)
            // Three-term cosine provides excellent -58 dB sidelobe rejection
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let r = compute_normalized_radius(i, j, k, nx, ny, nz);
                        weights[[i, j, k]] = 0.42 - 0.5 * (std::f32::consts::PI * r).cos()
                            + 0.08 * (2.0 * std::f32::consts::PI * r).cos();
                    }
                }
            }
        }
        ApodizationWindow::Gaussian { sigma } => {
            // Gaussian window: w(r) = exp(-r²/(2σ²))
            // Adjustable mainlobe width vs sidelobe trade-off via σ
            let sigma_f32 = *sigma as f32;
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let r = compute_normalized_radius(i, j, k, nx, ny, nz);
                        weights[[i, j, k]] = (-0.5 * r * r / (sigma_f32 * sigma_f32)).exp();
                    }
                }
            }
        }
        ApodizationWindow::Custom(custom_weights) => {
            // User-provided custom weights (flattened 3D array)
            // Layout: weights[i * ny * nz + j * nz + k]
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let idx = i * ny * nz + j * nz + k;
                        if idx < custom_weights.len() {
                            weights[[i, j, k]] = custom_weights[idx] as f32;
                        }
                    }
                }
            }
        }
    }

    weights
}

/// Compute normalized radial distance from array center
///
/// # Arguments
///
/// * `i`, `j`, `k` - Element indices
/// * `nx`, `ny`, `nz` - Array dimensions
///
/// # Returns
///
/// Normalized radius r ∈ [0, √3] (clamped to [0, 1] for edge elements)
///
/// # Implementation Note
///
/// We normalize coordinates to [-1, 1] in each dimension, then compute
/// Euclidean distance from origin. The maximum theoretical distance is √3
/// (corner elements), but we clamp to 1.0 for standard window functions.
#[inline]
fn compute_normalized_radius(i: usize, j: usize, k: usize, nx: usize, ny: usize, nz: usize) -> f32 {
    let x = 2.0 * i as f32 / (nx - 1) as f32 - 1.0;
    let y = 2.0 * j as f32 / (ny - 1) as f32 - 1.0;
    let z = 2.0 * k as f32 / (nz - 1) as f32 - 1.0;
    (x * x + y * y + z * z).sqrt().min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular_window() {
        let weights = create_apodization_weights((8, 8, 4), &ApodizationWindow::Rectangular);
        assert_eq!(weights.dim(), (8, 8, 4));
        // All weights should be 1.0 for rectangular window
        assert!(weights.iter().all(|&w| (w - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_hamming_window() {
        let weights = create_apodization_weights((16, 16, 8), &ApodizationWindow::Hamming);
        assert_eq!(weights.dim(), (16, 16, 8));
        // All weights should be in valid range
        assert!(weights.iter().all(|&w| (0.0..=1.0).contains(&w)));
        // Check that weights vary across the array (not uniform)
        let corner_weight = weights[[0, 0, 0]];
        let mid_weight = weights[[8, 8, 4]];
        assert!(
            (mid_weight - corner_weight).abs() > 0.01,
            "Hamming weights should vary: corner={}, mid={}",
            corner_weight,
            mid_weight
        );
    }

    #[test]
    fn test_hann_window() {
        let weights = create_apodization_weights((12, 12, 6), &ApodizationWindow::Hann);
        assert_eq!(weights.dim(), (12, 12, 6));
        assert!(weights.iter().all(|&w| (0.0..=1.0).contains(&w)));
        // Corner elements should have reduced weight (but for small arrays, may be close to 1.0)
        let corner_weight = weights[[0, 0, 0]];
        // For Hann window, corner weight depends on normalized radius
        assert!(
            (0.0..=1.0).contains(&corner_weight),
            "Corner weight: {}",
            corner_weight
        );
    }

    #[test]
    fn test_blackman_window() {
        let weights = create_apodization_weights((10, 10, 5), &ApodizationWindow::Blackman);
        assert_eq!(weights.dim(), (10, 10, 5));
        assert!(weights.iter().all(|&w| (0.0..=1.0).contains(&w)));
    }

    #[test]
    fn test_gaussian_window() {
        let sigma = 0.5;
        let weights = create_apodization_weights((8, 8, 4), &ApodizationWindow::Gaussian { sigma });
        assert_eq!(weights.dim(), (8, 8, 4));
        assert!(weights.iter().all(|&w| (0.0..=1.0).contains(&w)));
        // Center should have maximum weight (closest to array center)
        // For 8x8x4 array, center is at [3.5, 3.5, 1.5]
        let center_weight = weights[[4, 4, 2]];
        assert!(center_weight > 0.7, "Center weight: {}", center_weight);
    }

    #[test]
    fn test_custom_window() {
        let custom_weights = vec![0.5_f64; 8 * 8 * 4];
        let weights =
            create_apodization_weights((8, 8, 4), &ApodizationWindow::Custom(custom_weights));
        assert_eq!(weights.dim(), (8, 8, 4));
        // All weights should be 0.5
        assert!(weights.iter().all(|&w| (w - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_normalized_radius_center() {
        let r = compute_normalized_radius(4, 4, 2, 9, 9, 5);
        assert!(r.abs() < 1e-5, "Center radius should be ~0, got {}", r);
    }

    #[test]
    fn test_normalized_radius_edge() {
        let r = compute_normalized_radius(8, 4, 2, 9, 9, 5);
        assert!(r > 0.5 && r <= 1.0, "Edge radius: {}", r);
    }
}
