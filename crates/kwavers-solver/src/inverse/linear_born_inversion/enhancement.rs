//! Generic post-processing image enhancement for linear Born reconstructions.
//!
//! Anatomy-neutral high-pass enhancement: lifts edge contrast within a binary
//! support mask without modifying values outside it, and bounds the result to
//! a physiologically admissible range relative to a reference value. Clinical
//! adapters supply the mask, the reference (e.g. brain reference speed,
//! soft-tissue reference speed), and the gain.

use leto::{
    Array2,
    Array3,
};

/// 2-D high-pass image enhancement (slice analogue of [`high_pass_enhance_volume`]).
///
/// For every grid cell inside `support_mask`, computes a 3×3 local mean over
/// neighbouring masked cells, subtracts it from the centre to obtain a high-pass
/// component, and adds `gain × high_pass` back. Result clamped to
/// `[reference_value · 0.92, reference_value · 1.08]`. Cells outside the mask
/// pass through unchanged; `gain = 0.0` short-circuits to a cloned input.
#[must_use]
pub fn high_pass_enhance_slice(
    reconstruction: &Array2<f64>,
    support_mask: &Array2<bool>,
    gain: f64,
    reference_value: f64,
) -> Array2<f64> {
    if gain == 0.0 {
        return reconstruction.clone();
    }
    let (nx, ny) = reconstruction.dim();
    let mut enhanced = reconstruction.clone();
    for ix in 0..nx {
        for iy in 0..ny {
            if !support_mask[[ix, iy]] {
                continue;
            }
            let x0 = ix.saturating_sub(1);
            let x1 = (ix + 1).min(nx - 1);
            let y0 = iy.saturating_sub(1);
            let y1 = (iy + 1).min(ny - 1);
            let mut sum = 0.0;
            let mut count = 0.0;
            for ax in x0..=x1 {
                for ay in y0..=y1 {
                    if support_mask[[ax, ay]] {
                        sum += reconstruction[[ax, ay]];
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                let blur = sum / count;
                let high_pass = reconstruction[[ix, iy]] - blur;
                enhanced[[ix, iy]] = (reconstruction[[ix, iy]] + gain * high_pass)
                    .clamp(reference_value * 0.92, reference_value * 1.08);
            }
        }
    }
    enhanced
}

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
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

    #[test]
    fn gain_zero_passes_through_input() {
        let reconstruction = Array3::from_elem((2, 2, 2), SOUND_SPEED_WATER_SIM);
        let mask = Array3::from_elem((2, 2, 2), true);
        let enhanced = high_pass_enhance_volume(&reconstruction, &mask, 0.0, SOUND_SPEED_WATER_SIM);
        for (a, b) in enhanced.iter().zip(reconstruction.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn cells_outside_support_mask_are_unchanged() {
        let reconstruction = Array3::from_elem((3, 3, 3), 1700.0);
        let mut mask = Array3::from_elem((3, 3, 3), false);
        mask[[1, 1, 1]] = true;
        let enhanced = high_pass_enhance_volume(&reconstruction, &mask, 1.0, SOUND_SPEED_WATER_SIM);
        // Cell (0,0,0) is outside the mask — pass-through.
        assert_eq!(enhanced[[0, 0, 0]], 1700.0);
    }

    #[test]
    fn enhanced_values_stay_in_eight_percent_band() {
        let mut reconstruction = Array3::from_elem((3, 3, 3), SOUND_SPEED_WATER_SIM);
        reconstruction[[1, 1, 1]] = 2500.0;
        let mask = Array3::from_elem((3, 3, 3), true);
        let enhanced = high_pass_enhance_volume(&reconstruction, &mask, 5.0, SOUND_SPEED_WATER_SIM);
        let lower = SOUND_SPEED_WATER_SIM * 0.92;
        let upper = SOUND_SPEED_WATER_SIM * 1.08;
        for value in enhanced.iter() {
            assert!(
                *value >= lower - 1.0e-9 && *value <= upper + 1.0e-9,
                "value {value} outside [{lower}, {upper}]"
            );
        }
    }
}
