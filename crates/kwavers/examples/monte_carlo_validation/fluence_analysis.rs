//! Fluence comparison, correlation, and depth-profile analysis for the Monte Carlo validation.

use kwavers_grid::GridDimensions;
use std::io::Write;

/// Compare two fluence distributions
pub(crate) fn compare_fluence(
    mc_fluence: &[f64],
    diff_fluence: &[f64],
    _dims: GridDimensions,
) -> (f64, f64, f64) {
    assert_eq!(mc_fluence.len(), diff_fluence.len());

    let n = mc_fluence.len();
    let mut sum_rel_error = 0.0;
    let mut max_rel_error: f64 = 0.0;
    let mut count = 0;

    // Relative error calculation
    for i in 0..n {
        let mc = mc_fluence[i];
        let diff = diff_fluence[i];

        // Only compare where fluence is significant
        let threshold = mc_fluence.iter().cloned().fold(0.0, f64::max) * 1e-3;
        if mc > threshold && diff > threshold {
            let rel_error = ((mc - diff) / mc.max(diff)).abs();
            sum_rel_error += rel_error;
            max_rel_error = max_rel_error.max(rel_error);
            count += 1;
        }
    }

    let mean_rel_error = if count > 0 {
        sum_rel_error / count as f64
    } else {
        0.0
    };

    // Compute correlation coefficient
    let correlation = compute_correlation(mc_fluence, diff_fluence);

    (mean_rel_error, max_rel_error, correlation)
}

/// Compute Pearson correlation coefficient
fn compute_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-12 || var_y < 1e-12 {
        return 0.0;
    }

    cov / (var_x * var_y).sqrt()
}

/// Analyze depth profile (central axis)
pub(crate) fn analyze_depth_profile(
    mc_fluence: &[f64],
    diff_fluence: &[f64],
    dims: GridDimensions,
    label: &str,
) {
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Depth Profile Analysis ({}):",
        label
    );

    let cx = dims.nx / 2;
    let cy = dims.ny / 2;

    eprintln!("    z (mm) | MC Fluence | Diff Fluence | Rel. Error");
    let _ = writeln!(
        std::io::stdout().lock(),
        "    -------|------------|--------------|------------"
    );

    for k in (0..dims.nz).step_by(5) {
        let idx = k * (dims.nx * dims.ny) + cy * dims.nx + cx;
        let mc = mc_fluence[idx];
        let diff = diff_fluence[idx];
        let rel_err = if mc > 1e-12 {
            ((mc - diff) / mc).abs() * 100.0
        } else {
            0.0
        };

        let z_mm = (k as f64 + 0.5) * dims.dz * 1000.0;
        let _ = writeln!(
            std::io::stdout().lock(),
            "    {:.1}    | {:.3e}   | {:.3e}   | {:.1}%",
            z_mm,
            mc,
            diff,
            rel_err
        );
    }
}
