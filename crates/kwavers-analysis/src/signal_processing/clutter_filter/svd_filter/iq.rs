//! Complex I/Q spatiotemporal clutter filtering.
//!
//! The fUS input is a complex beamformed I/Q cube `X[T, A, R]`. Flattening the
//! spatial axes gives `X ∈ ℂ^(T×P)`, where `P = A·R`. This module removes the
//! leading complex singular modes and computes unnormalized power Doppler in
//! one provider call, matching the uncentred rank-truncation contract used for
//! beamformed fUS ensembles.
//!
//! Leto currently supplies the canonical real rank-revealing SVD. The exact
//! real representation of `X` is
//! `R(X) = [[Re(X), -Im(X)], [Im(X), Re(X)]]`. Every singular value of `X`
//! occurs twice in `R(X)`, so removing `k` complex modes is exactly removing
//! its first `2k` real singular modes. The reconstruction reads `Re(X)` from
//! the upper-left block and `Im(X)` from the lower-left block.
//!
//! Demené et al., *IEEE Transactions on Medical Imaging* 34(11), 2271-2285
//! (2015), DOI: 10.1109/TMI.2015.2428634, establishes spatiotemporal SVD as a
//! clutter-separation method. The complex-realification identity is verified
//! here by paired-rank value regressions and at the LeoNeuro CPython boundary.

use eunomia::Complex64;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array2, Array3};
use leto_ops::svd_rank_revealing;

/// Result of complex I/Q SVD clutter filtering.
#[derive(Debug, Clone)]
pub struct IqSvdClutterResult {
    /// Filtered I/Q ensemble with the input shape `[slow_time, angle, range]`.
    pub filtered_iq: Array3<Complex64>,
    /// Unnormalized power Doppler `Σ_t |filtered_iq[t, angle, range]|²`.
    pub power: Array2<f64>,
}

/// Fixed-rank complex I/Q SVD clutter filter for fUS ensembles.
#[derive(Debug, Clone, Copy)]
pub struct IqSvdClutterFilter {
    clutter_rank: usize,
}

impl IqSvdClutterFilter {
    /// Construct a filter that removes `clutter_rank` dominant complex modes.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::InvalidInput` when `clutter_rank` is zero.
    pub fn new(clutter_rank: usize) -> KwaversResult<Self> {
        if clutter_rank == 0 {
            return Err(KwaversError::InvalidInput(
                "I/Q clutter rank must be at least 1".to_owned(),
            ));
        }
        Ok(Self { clutter_rank })
    }

    /// Filter an I/Q cube and discard the accompanying power-Doppler map.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::InvalidInput` when the cube is empty, contains
    /// non-finite I/Q, its flattened spatial dimension overflows, or the rank
    /// would remove every available complex singular mode.
    pub fn filter(&self, iq_ensemble: &Array3<Complex64>) -> KwaversResult<Array3<Complex64>> {
        self.filter_with_power(iq_ensemble)
            .map(|result| result.filtered_iq)
    }

    /// Filter an I/Q cube and compute its unnormalized power-Doppler map.
    ///
    /// The input shape is `[slow_time, angle, range]`. No temporal centering or
    /// mean restoration occurs: a coherent stationary rank-one tissue component
    /// is a dominant complex mode and is removed exactly by the stated rank.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::InvalidInput` when the cube is empty, contains
    /// non-finite I/Q, its flattened spatial dimension overflows, or the rank
    /// would remove every available complex singular mode.
    pub fn filter_with_power(
        &self,
        iq_ensemble: &Array3<Complex64>,
    ) -> KwaversResult<IqSvdClutterResult> {
        let [slow_time, angles, ranges] = iq_ensemble.shape();
        let pixels = angles.checked_mul(ranges).ok_or_else(|| {
            KwaversError::InvalidInput(
                "I/Q angle and range dimensions overflow the flattened spatial size".to_owned(),
            )
        })?;
        let available_modes = slow_time.min(pixels);
        if slow_time == 0 || pixels == 0 {
            return Err(KwaversError::InvalidInput(
                "I/Q clutter filtering requires non-empty slow-time and spatial dimensions"
                    .to_owned(),
            ));
        }
        if self.clutter_rank >= available_modes {
            return Err(KwaversError::InvalidInput(format!(
                "I/Q clutter rank ({}) must be smaller than available complex modes ({available_modes})",
                self.clutter_rank
            )));
        }
        if !iq_ensemble.iter().copied().all(Complex64::is_finite) {
            return Err(KwaversError::InvalidInput(
                "I/Q clutter filtering requires finite real and imaginary samples".to_owned(),
            ));
        }

        let doubled_slow_time = slow_time.checked_mul(2).ok_or_else(|| {
            KwaversError::InvalidInput("I/Q slow-time dimension overflows realification".to_owned())
        })?;
        let doubled_pixels = pixels.checked_mul(2).ok_or_else(|| {
            KwaversError::InvalidInput("I/Q spatial dimension overflows realification".to_owned())
        })?;
        let realified_len = doubled_slow_time
            .checked_mul(doubled_pixels)
            .ok_or_else(|| {
                KwaversError::InvalidInput("I/Q realified matrix size overflows".to_owned())
            })?;
        let mut realified = vec![0.0; realified_len];
        for time in 0..slow_time {
            for angle in 0..angles {
                for range in 0..ranges {
                    let pixel = angle * ranges + range;
                    let sample = iq_ensemble[[time, angle, range]];
                    realified[time * doubled_pixels + pixel] = sample.re;
                    realified[time * doubled_pixels + pixels + pixel] = -sample.im;
                    realified[(slow_time + time) * doubled_pixels + pixel] = sample.im;
                    realified[(slow_time + time) * doubled_pixels + pixels + pixel] = sample.re;
                }
            }
        }

        let realified = Array2::from_shape_vec([doubled_slow_time, doubled_pixels], realified)?;
        let svd = svd_rank_revealing(&realified.view())?;
        let removed_modes = self.clutter_rank.checked_mul(2).ok_or_else(|| {
            KwaversError::InvalidInput("I/Q clutter rank overflows paired real modes".to_owned())
        })?;
        let mut singular_values = svd.singular_values;
        for singular_value in singular_values.iter_mut().take(removed_modes) {
            *singular_value = 0.0;
        }

        let left = svd.left_singular_vectors;
        let right = svd.right_singular_vectors;
        let mut filtered = Vec::with_capacity(slow_time * pixels);
        let mut power = vec![0.0; pixels];
        for time in 0..slow_time {
            for angle in 0..angles {
                for range in 0..ranges {
                    let pixel = angle * ranges + range;
                    let mut real = 0.0;
                    let mut imaginary = 0.0;
                    for (mode, singular_value) in singular_values.iter().enumerate() {
                        let right_component = right[[pixel, mode]];
                        real = left[[time, mode]].mul_add(*singular_value * right_component, real);
                        imaginary = left[[slow_time + time, mode]]
                            .mul_add(*singular_value * right_component, imaginary);
                    }
                    let sample = Complex64::new(real, imaginary);
                    power[pixel] += sample.norm_sqr();
                    filtered.push(sample);
                }
            }
        }

        Ok(IqSvdClutterResult {
            filtered_iq: Array3::from_shape_vec([slow_time, angles, ranges], filtered)?,
            power: Array2::from_shape_vec([angles, ranges], power)?,
        })
    }

    /// Number of dominant complex modes removed by this filter.
    #[must_use]
    pub const fn clutter_rank(self) -> usize {
        self.clutter_rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gamma(operations: usize) -> f64 {
        let scaled_epsilon = operations as f64 * f64::EPSILON;
        scaled_epsilon / (1.0 - scaled_epsilon)
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, bound: f64) {
        let error = (actual.re - expected.re).hypot(actual.im - expected.im);
        assert!(
            error <= bound * expected.norm_sqr().sqrt().max(1.0),
            "actual={actual:?}, expected={expected:?}, error={error}, bound={bound}"
        );
    }

    #[test]
    fn paired_realification_removes_one_complex_mode_and_preserves_phase() {
        let clutter = Complex64::new(3.0, 4.0);
        let flow = Complex64::new(1.5, -2.0);
        let iq = Array3::from_shape_vec(
            [2, 1, 2],
            vec![
                clutter,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                flow,
            ],
        )
        .expect("the fixed 2×1×2 I/Q fixture matches its storage");

        let result = IqSvdClutterFilter::new(1)
            .expect("rank one is valid")
            .filter_with_power(&iq)
            .expect("two complex singular modes admit rank-one removal");

        // A 4×4 one-sided-Jacobi factorization has at most six pair rotations
        // per sweep for sixty sweeps, then four reconstruction terms. gamma_1024
        // bounds those rounded rotation/reconstruction operations conservatively.
        let bound = gamma(1024);
        assert_complex_close(
            result.filtered_iq[[0, 0, 0]],
            Complex64::new(0.0, 0.0),
            bound,
        );
        assert_complex_close(
            result.filtered_iq[[0, 0, 1]],
            Complex64::new(0.0, 0.0),
            bound,
        );
        assert_complex_close(
            result.filtered_iq[[1, 0, 0]],
            Complex64::new(0.0, 0.0),
            bound,
        );
        assert_complex_close(result.filtered_iq[[1, 0, 1]], flow, bound);
        assert!((result.power[[0, 0]] - 0.0).abs() <= bound);
        assert!((result.power[[0, 1]] - flow.norm_sqr()).abs() <= bound * flow.norm_sqr());
    }

    #[test]
    fn rank_one_stationary_complex_clutter_is_not_mean_restored() {
        let clutter = [Complex64::new(3.0, 4.0), Complex64::new(-1.0, 2.0)];
        let iq = Array3::from_shape_vec(
            [2, 1, 2],
            vec![clutter[0], clutter[1], clutter[0], clutter[1]],
        )
        .expect("the fixed rank-one I/Q fixture matches its storage");

        let result = IqSvdClutterFilter::new(1)
            .expect("rank one is valid")
            .filter_with_power(&iq)
            .expect("the rank-one clutter component is removable");
        let bound = gamma(1024);
        for sample in result.filtered_iq.iter().copied() {
            assert_complex_close(sample, Complex64::new(0.0, 0.0), bound);
        }
        for value in result.power.iter().copied() {
            assert!(value.abs() <= bound);
        }
    }

    #[test]
    fn rejects_invalid_rank_and_non_finite_iq() {
        assert!(IqSvdClutterFilter::new(0).is_err());
        let iq = Array3::from_shape_vec([2, 1, 2], vec![Complex64::new(1.0, 0.0); 4])
            .expect("the fixed I/Q fixture matches its storage");
        assert!(IqSvdClutterFilter::new(2)
            .expect("the construction validates only nonzero rank")
            .filter(&iq)
            .is_err());

        let non_finite = Array3::from_shape_vec(
            [2, 1, 2],
            vec![
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("the fixed I/Q fixture matches its storage");
        assert!(IqSvdClutterFilter::new(1)
            .expect("rank one is valid")
            .filter(&non_finite)
            .is_err());
    }
}
