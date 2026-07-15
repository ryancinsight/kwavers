use leto::Array3;

/// Compute local standard deviation (texture feature).
///
/// Applies a 3×3 sliding window to compute standard deviation at each pixel,
/// providing a measure of local texture and variability.
///
/// # Mathematical Definition
///
/// ```text
/// σ(x,y) = √[ E[I²] - E[I]² ]
/// ```
///
/// # Arguments
///
/// * `image` - Input image (frames, height, width)
///
/// # Returns
///
/// Standard deviation map with same dimensions as input.
///
/// # Edge Handling
///
/// Borders (1-pixel margin) are set to zero to avoid edge artifacts.
#[must_use]
pub fn compute_local_std(image: &Array3<f32>) -> Array3<f32> {
    let mut std_map = Array3::zeros(image.shape());
    let [d0, d1, d2] = image.shape();

    for k in 0..d0 {
        for i in 1..d1 - 1 {
            for j in 1..d2 - 1 {
                let mut sum = 0.0;
                let mut sq_sum = 0.0;
                let mut count = 0;

                // 3×3 neighborhood
                for di in -1..=1 {
                    for dj in -1..=1 {
                        let val =
                            image[[k, (i as isize + di) as usize, (j as isize + dj) as usize]];
                        sum += val;
                        sq_sum += val * val;
                        count += 1;
                    }
                }

                let mean = sum / count as f32;
                let variance = (sq_sum / count as f32) - (mean * mean);
                std_map[[k, i, j]] = variance.max(0.0).sqrt();
            }
        }
    }

    std_map
}

/// Compute spatial gradient magnitude using Sobel operator.
///
/// Detects edges and boundaries by computing the magnitude of the
/// image gradient in X and Y directions.
///
/// # Sobel Kernels
///
/// X-direction (vertical edges):
/// ```text
/// [-1  0  1]
/// [-2  0  2]
/// [-1  0  1]
/// ```
///
/// Y-direction (horizontal edges):
/// ```text
/// [-1 -2 -1]
/// [ 0  0  0]
/// [ 1  2  1]
/// ```
///
/// # Arguments
///
/// * `image` - Input image (frames, height, width)
///
/// # Returns
///
/// Gradient magnitude map: |∇I| = √(Gₓ² + Gᵧ²)
#[must_use]
pub fn compute_spatial_gradient(image: &Array3<f32>) -> Array3<f32> {
    let mut grad_map = Array3::zeros(image.shape());
    let [d0, d1, d2] = image.shape();

    for k in 0..d0 {
        for i in 1..d1 - 1 {
            for j in 1..d2 - 1 {
                // Sobel X kernel (vertical edges)
                let gx = 2.0f32.mul_add(
                    image[[k, i + 1, j]],
                    2.0f32.mul_add(
                        -image[[k, i - 1, j]],
                        -image[[k, i - 1, j - 1]] + image[[k, i + 1, j - 1]],
                    ),
                ) - image[[k, i - 1, j + 1]]
                    + image[[k, i + 1, j + 1]];

                // Sobel Y kernel (horizontal edges)
                let gy = 2.0f32.mul_add(
                    image[[k, i, j + 1]],
                    2.0f32.mul_add(-image[[k, i, j - 1]], -image[[k, i - 1, j - 1]])
                        - image[[k, i + 1, j - 1]]
                        + image[[k, i - 1, j + 1]],
                ) + image[[k, i + 1, j + 1]];

                // Gradient magnitude
                grad_map[[k, i, j]] = gx.hypot(gy);
            }
        }
    }

    grad_map
}

/// Compute Laplacian (second derivative) for structural features.
///
/// Detects regions of rapid intensity change, useful for identifying
/// boundaries, ridges, and other structural features.
///
/// # Laplacian Kernel (4-connected)
///
/// ```text
/// [ 0  1  0]
/// [ 1 -4  1]
/// [ 0  1  0]
/// ```
///
/// # Mathematical Definition
///
/// ```text
/// ∇²I(x,y) = I(x+1,y) + I(x-1,y) + I(x,y+1) + I(x,y-1) - 4I(x,y)
/// ```
///
/// # Arguments
///
/// * `image` - Input image (frames, height, width)
///
/// # Returns
///
/// Absolute value of Laplacian: |∇²I|
#[must_use]
pub fn compute_laplacian(image: &Array3<f32>) -> Array3<f32> {
    let mut lap_map = Array3::zeros(image.shape());
    let [d0, d1, d2] = image.shape();

    for k in 0..d0 {
        for i in 1..d1 - 1 {
            for j in 1..d2 - 1 {
                // 4-connected Laplacian kernel
                let lap = 4.0f32.mul_add(
                    -image[[k, i, j]],
                    image[[k, i - 1, j]]
                        + image[[k, i + 1, j]]
                        + image[[k, i, j - 1]]
                        + image[[k, i, j + 1]],
                );

                lap_map[[k, i, j]] = lap.abs();
            }
        }
    }

    lap_map
}

/// Compute local entropy (information content).
///
/// Measures the randomness or information content in a local neighborhood,
/// useful for texture analysis and speckle characterization.
///
/// # Mathematical Definition
///
/// For a 3×3 patch with normalized intensities p(i):
/// ```text
/// H = -∑ᵢ p(i) log₂ p(i)
/// ```
///
/// # Arguments
///
/// * `image` - Input image (frames, height, width)
///
/// # Returns
///
/// Local entropy map (higher values = more random/textured)
///
/// # Implementation Note
///
/// Uses a simplified histogram-free approximation based on normalized
/// patch variance for computational efficiency.
#[must_use]
pub fn compute_local_entropy(image: &Array3<f32>) -> Array3<f32> {
    let mut entropy_map = Array3::zeros(image.shape());
    let [d0, d1, d2] = image.shape();

    const EPSILON: f32 = 1e-10;

    for k in 0..d0 {
        for i in 1..d1 - 1 {
            for j in 1..d2 - 1 {
                let mut sum = 0.0;
                let mut patch = [0.0f32; 9];
                let mut idx = 0;

                // Extract 3×3 patch
                for di in -1..=1 {
                    for dj in -1..=1 {
                        let val = image
                            [[k, (i as isize + di) as usize, (j as isize + dj) as usize]]
                        .abs();
                        patch[idx] = val;
                        sum += val;
                        idx += 1;
                    }
                }

                if sum < EPSILON {
                    continue;
                }

                // Normalize patch to form probability distribution
                let mut entropy = 0.0;
                for val in patch {
                    let p = val / sum;
                    if p > EPSILON {
                        entropy -= p * p.ln();
                    }
                }

                entropy_map[[k, i, j]] = entropy;
            }
        }
    }

    entropy_map
}
