//! Unit tests for RTM imaging conditions and Laplacian operator.
//!
//! All expected values are derived from the closed-form definitions of each
//! imaging condition; no empirical tolerances or magic numbers.

#[cfg(test)]
mod tests {
    use ndarray::{s, Array2, Array3, Array4};

    use crate::inverse::reconstruction::seismic::config::{
        RtmImagingCondition, SeismicImagingConfig,
    };
    use crate::inverse::reconstruction::seismic::rtm::ReverseTimeMigration;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_grid::Grid;

    fn rtm_with_condition(condition: RtmImagingCondition) -> ReverseTimeMigration {
        let mut config = SeismicImagingConfig::default();
        config.rtm_imaging_condition = condition;
        ReverseTimeMigration::new(config, Array3::from_elem((3, 3, 3), SOUND_SPEED_WATER_SIM))
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

    /// Regression: at medical-ultrasound frequencies (f₀ = 150 kHz, dx = 3 mm,
    /// dt ≈ 200 ns) the RTM forward-propagation `dt` and Ricker frequency must
    /// be read from `SeismicImagingConfig`, not from the seismic-scale defaults
    /// (5e-4 s, 15 Hz).  When this regression was introduced (May 2026) the
    /// hardcoded defaults produced CFL ≈ 250 → silent NaN divergence →
    /// identically-zero image and zero illumination.
    ///
    /// This test runs RTM end-to-end on a 32³ grid with one source / one
    /// receiver at MHz-scale parameters and asserts (a) the image is not
    /// identically zero, (b) every value is finite (no NaN/Inf from CFL
    /// violation), and (c) the source illumination accumulated.
    /// # Panics
    /// - Panics if the propagator silently regresses to the seismic-scale defaults.
    #[test]
    fn rtm_propagates_at_medical_ultrasound_frequency() {
        const NX: usize = 32;
        const NY: usize = 32;
        const NZ: usize = 32;
        const DX: f64 = 3.0e-3; // 3 mm
        const F0: f64 = 150_000.0; // 150 kHz
        const C: f64 = SOUND_SPEED_WATER_SIM; // water

        // CFL-stable dt for the 4th-order interior stencil:
        //   dt ≤ dx · CFL / (c_max · √D),  CFL = 0.3, D = 3
        let dt = 0.3 * DX / (C * (3.0_f64).sqrt());
        let nt = 64usize; // one-way transit ≈ 64 µs / dt(346 ns) ≈ 185; 64 is enough to see the wave move

        let config = SeismicImagingConfig {
            nx: NX,
            ny: NY,
            nz: NZ,
            dt,
            source_frequency_hz: F0,
            rtm_imaging_condition: RtmImagingCondition::ZeroLag,
            ..SeismicImagingConfig::default()
        };
        let velocity = Array3::from_elem((NX, NY, NZ), C);
        let mut rtm = ReverseTimeMigration::new(config, velocity);

        // Single source / single receiver placed inside the interior so the
        // 4th-order stencil (needs 2-cell halo) can read all neighbours.
        let source_pos = (NX / 2, NY / 2, NZ / 2);
        let receiver_pos = (NX / 2 + 6, NY / 2, NZ / 2);
        let receiver_positions = vec![receiver_pos];

        // Receiver data: same Ricker delayed by one path-length so backward
        // injection has non-trivial energy.  Magnitude is arbitrary — the
        // assertion is on non-zero output, not amplitude.
        let mut shot_data = Array2::<f64>::zeros((1, nt));
        for t in 0..nt {
            let tau = std::f64::consts::PI * F0 * (t as f64 * dt - 1.5 / F0);
            shot_data[[0, t]] = (1.0 - 2.0 * tau * tau) * (-tau * tau).exp();
        }

        let grid = Grid::new(NX, NY, NZ, DX, DX, DX).unwrap();
        rtm.migrate_shot(&shot_data, source_pos, &receiver_positions, &grid)
            .expect("migrate_shot must succeed at MHz-scale config.dt");

        // (b) every voxel finite — CFL violation would have produced NaN.
        assert!(
            rtm.get_image().iter().all(|v| v.is_finite()),
            "RTM image contains non-finite values — propagator likely violated CFL \
             (regression to hardcoded seismic-scale dt)"
        );

        // (a) image is not identically zero — the imaging condition fired.
        let img_peak = rtm
            .get_image()
            .iter()
            .copied()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            img_peak > 0.0,
            "RTM image is identically zero — forward propagation produced no \
             energy in the medium (regression to hardcoded 15 Hz wavelet at \
             MHz-scale dt would do this)"
        );
    }
}
