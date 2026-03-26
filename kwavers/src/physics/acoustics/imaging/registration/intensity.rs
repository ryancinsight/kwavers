use ndarray::{Array1, Array2, Array3};
use super::spatial::apply_transform_to_volume;

/// Find intensity range (min, max) of a volume for histogram normalization
pub(crate) fn find_intensity_range(volume: &Array3<f64>) -> (f64, f64) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for &val in volume.iter() {
        if val.is_finite() {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    // Add small epsilon to avoid division by zero in histogram binning
    if (max_val - min_val).abs() < 1e-12 {
        max_val = min_val + 1.0;
    }

    (min_val, max_val)
}

pub(crate) fn compute_mutual_information(
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
    transform: &[f64; 16],
) -> f64 {
    /// Mattes Mutual Information Implementation
    ///
    /// Reference: Mattes et al. (2003) "PET-CT Image Registration in the Chest Using Free-form Deformations"
    /// Reference: Viola & Wells (1997) "Alignment by Maximization of Mutual Information"
    ///
    /// MI(A,B) = H(A) + H(B) - H(A,B)
    /// where H is entropy and H(A,B) is joint entropy
    ///
    /// We use a histogram-based estimator with linear interpolation for subpixel accuracy.
    const N_BINS: usize = 64; // Number of histogram bins (Mattes recommends 50-64)

    // Transform moving image (simplified - assumes small rotations/translations)
    // Full implementation would use proper 3D interpolation
    let transformed_moving = apply_transform_to_volume(moving, transform);

    // Find intensity ranges for histogram binning
    let (fixed_min, fixed_max) = find_intensity_range(fixed);
    let (moving_min, moving_max) = find_intensity_range(&transformed_moving);

    // Build joint histogram
    let mut joint_histogram = Array2::<f64>::zeros((N_BINS, N_BINS));
    let mut marginal_fixed = Array1::<f64>::zeros(N_BINS);
    let mut marginal_moving = Array1::<f64>::zeros(N_BINS);

    let fixed_bin_width = (fixed_max - fixed_min) / (N_BINS as f64);
    let moving_bin_width = (moving_max - moving_min) / (N_BINS as f64);

    let mut n_samples = 0;

    // Populate histograms
    for ((i, j, k), &fixed_val) in fixed.indexed_iter() {
        if i < transformed_moving.dim().0
            && j < transformed_moving.dim().1
            && k < transformed_moving.dim().2
        {
            let moving_val = transformed_moving[[i, j, k]];

            // Compute bin indices with linear interpolation (Parzen windowing)
            let fixed_bin_f =
                ((fixed_val - fixed_min) / fixed_bin_width).clamp(0.0, (N_BINS - 1) as f64);
            let moving_bin_f =
                ((moving_val - moving_min) / moving_bin_width).clamp(0.0, (N_BINS - 1) as f64);

            let fixed_bin = fixed_bin_f.floor() as usize;
            let moving_bin = moving_bin_f.floor() as usize;

            // Bilinear interpolation weights
            let alpha_fixed = fixed_bin_f - fixed_bin as f64;
            let alpha_moving = moving_bin_f - moving_bin as f64;

            // Add contributions to all 4 neighboring bins
            if fixed_bin < N_BINS && moving_bin < N_BINS {
                joint_histogram[[fixed_bin, moving_bin]] +=
                    (1.0 - alpha_fixed) * (1.0 - alpha_moving);
                marginal_fixed[fixed_bin] += (1.0 - alpha_fixed) * (1.0 - alpha_moving);
                marginal_moving[moving_bin] += (1.0 - alpha_fixed) * (1.0 - alpha_moving);
            }

            if fixed_bin + 1 < N_BINS && moving_bin < N_BINS {
                joint_histogram[[fixed_bin + 1, moving_bin]] +=
                    alpha_fixed * (1.0 - alpha_moving);
                marginal_fixed[fixed_bin + 1] += alpha_fixed * (1.0 - alpha_moving);
                marginal_moving[moving_bin] += alpha_fixed * (1.0 - alpha_moving);
            }

            if fixed_bin < N_BINS && moving_bin + 1 < N_BINS {
                joint_histogram[[fixed_bin, moving_bin + 1]] +=
                    (1.0 - alpha_fixed) * alpha_moving;
                marginal_fixed[fixed_bin] += (1.0 - alpha_fixed) * alpha_moving;
                marginal_moving[moving_bin + 1] += (1.0 - alpha_fixed) * alpha_moving;
            }

            if fixed_bin + 1 < N_BINS && moving_bin + 1 < N_BINS {
                joint_histogram[[fixed_bin + 1, moving_bin + 1]] += alpha_fixed * alpha_moving;
                marginal_fixed[fixed_bin + 1] += alpha_fixed * alpha_moving;
                marginal_moving[moving_bin + 1] += alpha_fixed * alpha_moving;
            }

            n_samples += 1;
        }
    }

    if n_samples == 0 {
        return 0.0; // No overlap
    }

    // Normalize histograms to probabilities
    let normalization = 1.0 / n_samples as f64;
    joint_histogram *= normalization;
    marginal_fixed *= normalization;
    marginal_moving *= normalization;

    // Compute entropies
    let mut h_fixed = 0.0;
    let mut h_moving = 0.0;
    let mut h_joint = 0.0;

    for &p in marginal_fixed.iter() {
        if p > 1e-12 {
            h_fixed -= p * p.ln();
        }
    }

    for &p in marginal_moving.iter() {
        if p > 1e-12 {
            h_moving -= p * p.ln();
        }
    }

    for &p in joint_histogram.iter() {
        if p > 1e-12 {
            h_joint -= p * p.ln();
        }
    }

    // Mutual information: MI = H(A) + H(B) - H(A,B)
    let mi = h_fixed + h_moving - h_joint;

    mi.max(0.0) // MI is always non-negative
}

pub(crate) fn compute_correlation(
    fixed: &Array3<f64>,
    moving: &Array3<f64>,
    transform: &[f64; 16],
) -> f64 {
    // Pearson Correlation Coefficient Implementation
    //
    // Measures the linear relationship between two images:
    // r = Cov(A,B) / (σ_A · σ_B)
    //
    // Range: [-1, 1]
    // - r = 1: Perfect positive correlation
    // - r = 0: No correlation
    // - r = -1: Perfect negative correlation

    let transformed_moving = apply_transform_to_volume(moving, transform);

    let mut sum_fixed = 0.0;
    let mut sum_moving = 0.0;
    let mut sum_fixed_sq = 0.0;
    let mut sum_moving_sq = 0.0;
    let mut sum_product = 0.0;
    let mut n_samples = 0;

    // Compute sums for Pearson correlation
    for ((i, j, k), &fixed_val) in fixed.indexed_iter() {
        if i < transformed_moving.dim().0
            && j < transformed_moving.dim().1
            && k < transformed_moving.dim().2
        {
            let moving_val = transformed_moving[[i, j, k]];

            sum_fixed += fixed_val;
            sum_moving += moving_val;
            sum_fixed_sq += fixed_val * fixed_val;
            sum_moving_sq += moving_val * moving_val;
            sum_product += fixed_val * moving_val;
            n_samples += 1;
        }
    }

    if n_samples == 0 {
        return 0.0; // No overlap
    }

    let n = n_samples as f64;

    // Pearson correlation formula
    let numerator = n * sum_product - sum_fixed * sum_moving;
    let denominator_a = (n * sum_fixed_sq - sum_fixed * sum_fixed).sqrt();
    let denominator_b = (n * sum_moving_sq - sum_moving * sum_moving).sqrt();

    if denominator_a.abs() < 1e-12 || denominator_b.abs() < 1e-12 {
        return 0.0; // Avoid division by zero
    }

    let correlation = numerator / (denominator_a * denominator_b);

    correlation.clamp(-1.0, 1.0)
}

pub(crate) fn compute_ncc(fixed: &Array3<f64>, moving: &Array3<f64>, transform: &[f64; 16]) -> f64 {
    // Normalized Cross-Correlation (NCC) Implementation
    //
    // Reference: Avants et al. (2008) "Symmetric diffeomorphic image registration with cross-correlation"
    // Reference: Lewis (1995) "Fast normalized cross-correlation"
    //
    // NCC = Σ[(A - mean(A)) · (B - mean(B))] / sqrt(Σ(A - mean(A))² · Σ(B - mean(B))²)
    //
    // Range: [-1, 1], with 1 being perfect alignment

    let transformed_moving = apply_transform_to_volume(moving, transform);

    // Compute means
    let mut sum_fixed = 0.0;
    let mut sum_moving = 0.0;
    let mut n_samples = 0;

    for ((i, j, k), &fixed_val) in fixed.indexed_iter() {
        if i < transformed_moving.dim().0
            && j < transformed_moving.dim().1
            && k < transformed_moving.dim().2
        {
            let moving_val = transformed_moving[[i, j, k]];
            sum_fixed += fixed_val;
            sum_moving += moving_val;
            n_samples += 1;
        }
    }

    if n_samples == 0 {
        return 0.0; // No overlap
    }

    let mean_fixed = sum_fixed / n_samples as f64;
    let mean_moving = sum_moving / n_samples as f64;

    // Compute NCC components
    let mut numerator = 0.0;
    let mut sum_fixed_centered_sq = 0.0;
    let mut sum_moving_centered_sq = 0.0;

    for ((i, j, k), &fixed_val) in fixed.indexed_iter() {
        if i < transformed_moving.dim().0
            && j < transformed_moving.dim().1
            && k < transformed_moving.dim().2
        {
            let moving_val = transformed_moving[[i, j, k]];

            let fixed_centered = fixed_val - mean_fixed;
            let moving_centered = moving_val - mean_moving;

            numerator += fixed_centered * moving_centered;
            sum_fixed_centered_sq += fixed_centered * fixed_centered;
            sum_moving_centered_sq += moving_centered * moving_centered;
        }
    }

    // Compute NCC
    let denominator = (sum_fixed_centered_sq * sum_moving_centered_sq).sqrt();

    if denominator.abs() < 1e-12 {
        return 0.0; // Avoid division by zero
    }

    let ncc = numerator / denominator;

    ncc.clamp(-1.0, 1.0)
}
