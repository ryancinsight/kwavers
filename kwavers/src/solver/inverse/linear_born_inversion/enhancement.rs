//! Generic post-processing image enhancement for linear Born reconstructions.
//!
//! Anatomy-neutral high-pass enhancement: lifts edge contrast within a binary
//! support mask without modifying values outside it, and bounds the result to
//! a physiologically admissible range relative to a reference value. Clinical
//! adapters supply the mask, the reference (e.g. brain reference speed,
//! soft-tissue reference speed), and the gain.

use ndarray::Array3;

/// 3-D high-pass image enhancement.
///
/// For every grid cell inside `support_mask`, computes a 3×3×3 local mean over
/// neighbouring masked cells, subtracts it from the centre to obtain a
/// high-pass component, and adds `gain × high_pass` back. The result is clamped
/// to `[reference_value · 0.92, reference_value · 1.08]` to stay inside an
/// 8 % band around `reference_value`.
///
/// Cells outside the support mask pass through unchanged. `gain = 0.0`
/// short-circuits to a cloned input.
#[must_use]
pub fn high_pass_enhance_volume(
    reconstruction: &Array3<f64>,
    support_mask: &Array3<bool>,
    gain: f64,
    reference_value: f64,
) -> Array3<f64> {
    if gain == 0.0 {
        return reconstruction.clone();
    }
    let (nx, ny, nz) = reconstruction.dim();
    let mut enhanced = reconstruction.clone();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                if !support_mask[[ix, iy, iz]] {
                    continue;
                }
                let mut sum = 0.0;
                let mut count = 0.0;
                for ax in ix.saturating_sub(1)..=(ix + 1).min(nx - 1) {
                    for ay in iy.saturating_sub(1)..=(iy + 1).min(ny - 1) {
                        for az in iz.saturating_sub(1)..=(iz + 1).min(nz - 1) {
                            if support_mask[[ax, ay, az]] {
                                sum += reconstruction[[ax, ay, az]];
                                count += 1.0;
                            }
                        }
                    }
                }
                if count > 0.0 {
                    let blur = sum / count;
                    let high_pass = reconstruction[[ix, iy, iz]] - blur;
                    enhanced[[ix, iy, iz]] = (reconstruction[[ix, iy, iz]] + gain * high_pass)
                        .clamp(reference_value * 0.92, reference_value * 1.08);
                }
            }
        }
    }
    enhanced
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_zero_passes_through_input() {
        let reconstruction = Array3::from_elem((2, 2, 2), 1500.0);
        let mask = Array3::from_elem((2, 2, 2), true);
        let enhanced = high_pass_enhance_volume(&reconstruction, &mask, 0.0, 1500.0);
        for (a, b) in enhanced.iter().zip(reconstruction.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn cells_outside_support_mask_are_unchanged() {
        let reconstruction = Array3::from_elem((3, 3, 3), 1700.0);
        let mut mask = Array3::from_elem((3, 3, 3), false);
        mask[[1, 1, 1]] = true;
        let enhanced = high_pass_enhance_volume(&reconstruction, &mask, 1.0, 1500.0);
        // Cell (0,0,0) is outside the mask — pass-through.
        assert_eq!(enhanced[[0, 0, 0]], 1700.0);
    }

    #[test]
    fn enhanced_values_stay_in_eight_percent_band() {
        let mut reconstruction = Array3::from_elem((3, 3, 3), 1500.0);
        reconstruction[[1, 1, 1]] = 2500.0;
        let mask = Array3::from_elem((3, 3, 3), true);
        let enhanced = high_pass_enhance_volume(&reconstruction, &mask, 5.0, 1500.0);
        let lower = 1500.0 * 0.92;
        let upper = 1500.0 * 1.08;
        for value in enhanced.iter() {
            assert!(
                *value >= lower - 1.0e-9 && *value <= upper + 1.0e-9,
                "value {value} outside [{lower}, {upper}]"
            );
        }
    }
}
