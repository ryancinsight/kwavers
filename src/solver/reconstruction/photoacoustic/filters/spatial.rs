//! Spatial filtering operations for 3D reconstruction
//!
//! This module provides spatial filtering operations including
//! Gaussian and bilateral filters for noise reduction and edge preservation.

use crate::error::KwaversResult;
use ndarray::Array3;
use std::f64::consts::PI;

/// Apply 3D Gaussian filter for noise reduction
///
/// Uses separable implementation for computational efficiency (O(n*k) vs O(n*k³))
///
/// # Arguments
///
/// * `image` - Input 3D image data
/// * `sigma` - Standard deviation in voxels
/// * `radius` - Kernel radius in voxels
///
/// # Returns
///
/// Filtered 3D image
pub fn apply_gaussian_filter(
    image: &Array3<f64>,
    sigma: f64,
    radius: usize,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = image.dim();
    
    // Generate 1D Gaussian kernel
    let kernel = create_gaussian_kernel(radius, sigma);

    // Apply separable filtering in each dimension
    // First pass: filter along X
    let mut temp1 = Array3::zeros((nx, ny, nz));
    for j in 0..ny {
        for k in 0..nz {
            for i in 0..nx {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for (ki, &kernel_val) in kernel.iter().enumerate() {
                    let ii = (i as i32 + ki as i32 - radius as i32) as usize;
                    if ii < nx {
                        sum += image[[ii, j, k]] * kernel_val;
                        weight_sum += kernel_val;
                    }
                }

                if weight_sum > 0.0 {
                    temp1[[i, j, k]] = sum / weight_sum;
                }
            }
        }
    }

    // Second pass: filter along Y
    let mut temp2 = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for k in 0..nz {
            for j in 0..ny {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for (kj, &kernel_val) in kernel.iter().enumerate() {
                    let jj = (j as i32 + kj as i32 - radius as i32) as usize;
                    if jj < ny {
                        sum += temp1[[i, jj, k]] * kernel_val;
                        weight_sum += kernel_val;
                    }
                }

                if weight_sum > 0.0 {
                    temp2[[i, j, k]] = sum / weight_sum;
                }
            }
        }
    }

    // Third pass: filter along Z
    let mut filtered = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for (kk, &kernel_val) in kernel.iter().enumerate() {
                    let zz = (k as i32 + kk as i32 - radius as i32) as usize;
                    if zz < nz {
                        sum += temp2[[i, j, zz]] * kernel_val;
                        weight_sum += kernel_val;
                    }
                }

                if weight_sum > 0.0 {
                    filtered[[i, j, k]] = sum / weight_sum;
                }
            }
        }
    }

    Ok(filtered)
}

/// Apply bilateral filter for edge-preserving noise reduction
///
/// Combines spatial and intensity weighting to preserve edges while smoothing.
/// Literature: Tomasi, C., & Manduchi, R. (1998). "Bilateral filtering for
/// gray and color images." ICCV.
///
/// # Arguments
///
/// * `image` - Input 3D image data
/// * `spatial_sigma` - Spatial Gaussian standard deviation
/// * `window_radius` - Window radius for local filtering
/// * `intensity_sigma` - Intensity Gaussian standard deviation (relative to range)
///
/// # Returns
///
/// Edge-preserved filtered 3D image
pub fn apply_bilateral_filter(
    image: &Array3<f64>,
    spatial_sigma: f64,
    window_radius: usize,
    intensity_sigma_relative: f64,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = image.dim();
    let mut filtered = image.clone();

    // Estimate intensity range for normalization
    let max_val = image.iter().copied().fold(0.0_f64, f64::max);
    let min_val = image.iter().copied().fold(f64::INFINITY, f64::min);
    let range = (max_val - min_val).max(1e-10);
    let intensity_sigma = intensity_sigma_relative * range;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let center_val = image[[i, j, k]];
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                // Apply bilateral filter in local window
                for di in -(window_radius as i32)..=(window_radius as i32) {
                    for dj in -(window_radius as i32)..=(window_radius as i32) {
                        for dk in -(window_radius as i32)..=(window_radius as i32) {
                            let ii = (i as i32 + di) as usize;
                            let jj = (j as i32 + dj) as usize;
                            let kk = (k as i32 + dk) as usize;

                            if ii < nx && jj < ny && kk < nz {
                                let neighbor_val = image[[ii, jj, kk]];

                                // Spatial weight
                                let spatial_dist2 = f64::from(di * di + dj * dj + dk * dk);
                                let spatial_weight =
                                    (-spatial_dist2 / (2.0 * spatial_sigma * spatial_sigma)).exp();

                                // Intensity weight
                                let intensity_diff = neighbor_val - center_val;
                                let intensity_weight = (-(intensity_diff * intensity_diff)
                                    / (2.0 * intensity_sigma * intensity_sigma))
                                    .exp();

                                let weight = spatial_weight * intensity_weight;
                                sum += neighbor_val * weight;
                                weight_sum += weight;
                            }
                        }
                    }
                }

                if weight_sum > 0.0 {
                    filtered[[i, j, k]] = sum / weight_sum;
                }
            }
        }
    }

    Ok(filtered)
}

/// Create 1D Gaussian kernel
///
/// Generates normalized Gaussian kernel for convolution.
///
/// # Arguments
///
/// * `radius` - Kernel radius (total size = 2*radius + 1)
/// * `sigma` - Standard deviation
///
/// # Returns
///
/// Normalized Gaussian kernel
fn create_gaussian_kernel(radius: usize, sigma: f64) -> Vec<f64> {
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0; size];
    let norm = 1.0 / (sigma * (2.0 * PI).sqrt());
    let sigma2 = 2.0 * sigma * sigma;

    for (i, kernel_val) in kernel.iter_mut().enumerate().take(size) {
        let x = f64::from(i as i32 - radius as i32);
        *kernel_val = norm * (-x * x / sigma2).exp();
    }

    // Normalize
    let sum: f64 = kernel.iter().sum();
    for val in &mut kernel {
        *val /= sum;
    }

    kernel
}
