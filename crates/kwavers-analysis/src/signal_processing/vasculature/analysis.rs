//! Image analysis utilities: Otsu thresholding and connected-component labelling.
//!
//! # Threshold boundary
//!
//! Otsu's between-class-variance threshold is computed locally over the Frangi
//! response volume. It is intentionally self-contained: Otsu (1979) is a
//! standardized histogram algorithm, not a domain-specific capability, so
//! owning it here keeps this crate free of a cross-workspace dependency on an
//! external image library's internal module path (a fragile coupling point).
//!
//! # Connected components
//!
//! 6-connectivity (face-adjacent neighbours only) is used for the flood-fill
//! labelling, matching the standard convention for 3-D binary images in
//! medical imaging (Rosenfeld & Pfaltz 1966).
//!
//! # References
//! - Otsu, N. (1979). IEEE Trans. Syst. Man Cybern. 9(1), pp. 62-66.
//! - Rosenfeld & Pfaltz (1966). J. ACM 13(4), pp. 471-494.

use leto::Array3;

/// Number of equally-spaced histogram bins for Otsu thresholding.
const OTSU_BINS: usize = 256;

/// Compute the global Otsu threshold for a 3-D Frangi response volume.
///
/// Otsu's method selects the intensity t* maximising the between-class variance
///   σ²_B(t) = P₁(t)·P₂(t)·(μ₁(t) − μ₂(t))²
/// over a 256-bin histogram of the response, then maps the winning bin back to
/// intensity units `t* = x_min + best_t/(N−1)·(x_max − x_min)`. Complexity
/// O(n + N): one O(n) histogram pass plus an O(N) prefix-sum scan over bins.
/// A constant volume (no separable classes) returns its uniform intensity.
#[must_use]
pub(super) fn otsu_threshold(image: &Array3<f64>) -> f64 {
    let n = image.len();
    if n == 0 {
        return 0.0;
    }

    // Intensity range over the volume.
    let (x_min, x_max) = image
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    let range = x_max - x_min;
    if range.abs() < f64::EPSILON {
        return x_min; // degenerate: constant volume has no separable classes
    }

    let bins_minus_1 = (OTSU_BINS - 1) as f64;

    // Normalised histogram: bin(v) = ⌊(v − x_min)/range · (N − 1)⌋, clamped.
    let mut counts = vec![0u64; OTSU_BINS];
    for &v in image.iter() {
        let bin = (((v - x_min) / range * bins_minus_1).floor() as usize).min(OTSU_BINS - 1);
        counts[bin] += 1;
    }
    let h: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

    // Total weighted mean over bin indices (prefix-sum trick for μ₂).
    let total_mu: f64 = (0..OTSU_BINS).map(|i| i as f64 * h[i]).sum();

    // O(N) prefix-sum scan: class 1 = bins [0, t−1], class 2 = bins [t, N−1].
    let mut best_sigma2 = 0.0_f64;
    let mut best_t = 0_usize;
    let mut w1 = 0.0_f64; // Σ h[0..t−1]
    let mut mu1_partial = 0.0_f64; // Σ i·h[i] for i ∈ [0, t−1]
    for t in 1..OTSU_BINS {
        w1 += h[t - 1];
        mu1_partial += (t - 1) as f64 * h[t - 1];
        let w2 = 1.0 - w1;
        if w1 < 1e-12 || w2 < 1e-12 {
            continue; // skip splits with an empty class
        }
        let mu1 = mu1_partial / w1;
        let mu2 = (total_mu - mu1_partial) / w2;
        let sigma2 = w1 * w2 * (mu1 - mu2) * (mu1 - mu2);
        if sigma2 > best_sigma2 {
            best_sigma2 = sigma2;
            best_t = t;
        }
    }

    x_min + best_t as f64 / bins_minus_1 * range
}

/// Count 6-connected components in a binary mask (values > 0).
///
/// Uses an iterative flood-fill to avoid stack overflow on large volumes.
///
/// Returns `(n_components, total_vessel_voxels)`.
pub(super) fn count_connected_components(mask: &Array3<f64>) -> (usize, usize) {
    let [nx, ny, nz] = mask.shape();
    let mut visited = Array3::<bool>::from_elem((nx, ny, nz), false);
    let mut n_components = 0_usize;
    let mut total_voxels = 0_usize;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if mask[[i, j, k]] > 0.0 && !visited[[i, j, k]] {
                    n_components += 1;
                    let mut stack = vec![(i, j, k)];
                    while let Some((ci, cj, ck)) = stack.pop() {
                        if visited[[ci, cj, ck]] {
                            continue;
                        }
                        visited[[ci, cj, ck]] = true;
                        total_voxels += 1;

                        // 6-connected neighbours
                        if ci > 0 && mask[[ci - 1, cj, ck]] > 0.0 {
                            stack.push((ci - 1, cj, ck));
                        }
                        if ci + 1 < nx && mask[[ci + 1, cj, ck]] > 0.0 {
                            stack.push((ci + 1, cj, ck));
                        }
                        if cj > 0 && mask[[ci, cj - 1, ck]] > 0.0 {
                            stack.push((ci, cj - 1, ck));
                        }
                        if cj + 1 < ny && mask[[ci, cj + 1, ck]] > 0.0 {
                            stack.push((ci, cj + 1, ck));
                        }
                        if ck > 0 && mask[[ci, cj, ck - 1]] > 0.0 {
                            stack.push((ci, cj, ck - 1));
                        }
                        if ck + 1 < nz && mask[[ci, cj, ck + 1]] > 0.0 {
                            stack.push((ci, cj, ck + 1));
                        }
                    }
                }
            }
        }
    }

    (n_components, total_voxels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Otsu on a clearly bimodal volume must place the threshold in the gap
    /// between the two clusters, so `value ≥ t*` separates them exactly.
    #[test]
    fn otsu_separates_bimodal_clusters() {
        // 6 low-intensity + 6 high-intensity voxels, no mass in the gap (1, 9).
        let data = vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 9.0, 9.0, 9.5, 9.5, 10.0, 10.0];
        let volume = Array3::from_shape_vec((2, 2, 3), data).unwrap();

        let t = otsu_threshold(&volume);

        // Strictly above the low cluster's max (1.0) and at/below the high
        // cluster's min (9.0): a correct binary separation of the two classes.
        assert!(
            t > 1.0,
            "threshold {t} should exceed the low cluster max 1.0"
        );
        assert!(
            t <= 9.0,
            "threshold {t} should not exceed the high cluster min 9.0"
        );
    }

    /// A constant volume has no separable classes → returns the uniform value.
    #[test]
    fn otsu_constant_volume_returns_uniform_value() {
        let volume = Array3::from_elem((3, 3, 3), 4.2_f64);
        assert_eq!(otsu_threshold(&volume), 4.2);
    }
}
