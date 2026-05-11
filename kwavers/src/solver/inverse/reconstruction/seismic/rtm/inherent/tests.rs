//! Unit tests for RTM imaging conditions and Laplacian operator.
//!
//! All expected values are derived from the closed-form definitions of each
//! imaging condition; no empirical tolerances or magic numbers.

#[cfg(test)]
mod tests {
    use ndarray::{s, Array3, Array4};

    use crate::solver::inverse::reconstruction::seismic::config::{
        RtmImagingCondition, SeismicImagingConfig,
    };
    use crate::solver::inverse::reconstruction::seismic::rtm::ReverseTimeMigration;

    fn rtm_with_condition(condition: RtmImagingCondition) -> ReverseTimeMigration {
        let mut config = SeismicImagingConfig::default();
        config.rtm_imaging_condition = condition;
        ReverseTimeMigration::new(config, Array3::from_elem((3, 3, 3), 1500.0))
    }

    /// `EnergyNormalized`: I = (Σ_t S·R) / (Σ_t S²)
    ///
    /// S = 2, R = 3, 2 time steps:
    ///   numerator   = 2·3·2 = 12
    ///   denominator = 4·2   =  8
    ///   result      = 1.5
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn energy_normalized_condition_matches_cross_correlation_over_source_energy() {
        let mut rtm = rtm_with_condition(RtmImagingCondition::EnergyNormalized);
        let source = Array4::from_elem((2, 3, 3, 3), 2.0);
        let receiver = Array4::from_elem((2, 3, 3, 3), 3.0);

        rtm.apply_imaging_condition(&source, &receiver).unwrap();

        assert!(
            rtm.image.iter().all(|v| (*v - 1.5).abs() < 1e-12),
            "expected all image values = 1.5, centre = {}",
            rtm.image[[1, 1, 1]]
        );
    }

    /// `SourceNormalized`: I = Σ_t (∂S/∂t)·R   (centred differences).
    ///
    /// S[t,…] = t; ∂S/∂t = 1 at all interior t.
    /// R = 3; 3 time steps → image = 1·3·3 = 9.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn source_normalized_condition_uses_temporal_source_derivative() {
        let mut rtm = rtm_with_condition(RtmImagingCondition::SourceNormalized);
        let mut source = Array4::zeros((3, 3, 3, 3));
        for t in 0..3_usize {
            source.slice_mut(s![t, .., .., ..]).fill(t as f64);
        }
        let receiver = Array4::from_elem((3, 3, 3, 3), 3.0);

        rtm.apply_imaging_condition(&source, &receiver).unwrap();

        assert!(
            rtm.image.iter().all(|v| (*v - 9.0).abs() < 1e-12),
            "expected all image values = 9.0, centre = {}",
            rtm.image[[1, 1, 1]]
        );
    }

    /// `Poynting`: I = Σ_t ∇S·∇R at interior point (1,1,1).
    ///
    /// S[t,i,j,k] = i+j+k, R = 2·S, 2 time steps.
    /// Centred gradient at (1,1,1):
    ///   ∂S/∂x = 0.5·(S[2,1,1]−S[0,1,1]) = 0.5·(4−2) = 1
    ///   ∂R/∂x = 2·∂S/∂x = 2
    ///   x-contribution = 0.25·(4−2)·(8−4) = 2; same for y and z.
    /// Total per step = 6; 2 steps → image[[1,1,1]] = 12.
    /// Boundary point (0,1,1) must be zero (not in interior).
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn poynting_condition_accumulates_spatial_gradient_dot_product() {
        let mut rtm = rtm_with_condition(RtmImagingCondition::Poynting);
        let mut source = Array4::zeros((2, 3, 3, 3));
        let mut receiver = Array4::zeros((2, 3, 3, 3));
        for t in 0..2 {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let v = (i + j + k) as f64;
                        source[[t, i, j, k]] = v;
                        receiver[[t, i, j, k]] = 2.0 * v;
                    }
                }
            }
        }

        rtm.apply_imaging_condition(&source, &receiver).unwrap();

        assert!(
            (rtm.image[[1, 1, 1]] - 12.0).abs() < 1e-12,
            "expected image[[1,1,1]] = 12.0, got {}",
            rtm.image[[1, 1, 1]]
        );
        assert_eq!(rtm.image[[0, 1, 1]], 0.0, "boundary must be zero");
    }
}
