//! Absorption validation tests for KZK equation implementation.

#[cfg(test)]
mod tests {
    use super::super::super::*;
    use kwavers_core::constants::acoustic_parameters::DB_TO_NP;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use leto::Array2;

    /// Test power-law absorption: spectral field amplitude decay.
    ///
    /// ## Setup
    ///
    /// Source: uniform plane wave p₀ = 1 Pa at f₀ = 1 MHz.
    /// Medium: water-like, α₀ = 0.5 dB/(cm·MHz), y = 1.0.
    /// Propagation: 10 cm (100 steps × 1 mm).
    ///
    /// ## Expected result
    ///
    /// After propagating distance d = 0.1 m at frequency f₀:
    ///   α(f₀) = 0.5 dB/cm = 5 dB/m = 5/8.686 Np/m ≈ 0.5756 Np/m
    ///   Amplitude decay: exp(-α·d) = exp(-0.05756) ≈ 0.944
    ///   Intensity ratio = [exp(-α·d)]² = exp(-2α·d) ≈ 0.891
    ///
    /// Wait — at 10 cm: α·d = 0.5756 × 0.1 = 0.05756 Np.
    /// In dB: 0.05756 × 8.686 = 0.5 dB.  Total: 5 dB over 10 cm.
    /// Intensity ratio = 10^(-5/10) = 10^(-0.5) ≈ 0.316.  ✓
    ///
    /// ## Grid requirements for spectral accuracy
    ///
    /// The time grid must resolve the fundamental frequency as an exact FFT bin:
    ///   N·Δτ = M · T₀  (M complete periods; T₀ = 1/f₀)
    ///
    /// With f₀ = 1 MHz and Δτ = 1/(8f₀) = 125 ns: each period contains 8
    /// samples.  N = 256 gives 32 complete periods.  Fundamental is at bin
    ///   k₀ = f₀·N·Δτ = 1e6 × 256 × 125e-9 = 32   (exact integer ✓)
    ///
    /// ## Theorem (spectral absorption exactness for pure tones)
    ///
    /// A single-frequency signal p(τ) = p₀·sin(ω₀τ) has all energy in bins
    /// k₀ and N−k₀.  The spectral absorption operator multiplies bin k₀ by
    ///   H[k₀] = exp(−α(f₀)·Δz)
    /// exactly.  After d/Δz steps the amplitude at f₀ decays by exp(−α(f₀)·d).
    ///
    /// ## Reference
    ///
    /// Szabo TL (1994). J. Acoust. Soc. Am. 96(1), 491–500.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_absorption() {
        use apollo::fft_1d_leto;
        use leto::Array1;

        let frequency = MHZ_TO_HZ;
        // dt chosen so fundamental falls on exact FFT bin: dt = 1/(8·f₀)
        let dt = 1.0 / (8.0 * frequency); // 125 ns
        let nt = 256_usize; // 32 complete periods
        let dz = 1.0e-3_f64;

        let config = KZKConfig {
            nx: 4,
            ny: 4,
            nz: 100,
            nt,
            dx: 1e-3,
            dz,
            dt,
            include_nonlinearity: false,
            include_absorption: true,
            include_diffraction: false,
            alpha0: 0.5,
            alpha_power: 1.0,
            frequency,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        let source = Array2::from_elem((config.nx, config.ny), 1.0_f64);
        solver.set_source(source, frequency);

        let initial_signal = solver.get_time_signal(config.nx / 2, config.ny / 2);
        let initial_signal =
            Array1::from_shape_vec([nt], initial_signal).expect("time signal length matches nt");
        let initial_spectrum = fft_1d_leto(initial_signal.view());
        let fundamental_bin = (frequency * nt as f64 * dt).round() as usize; // = 32
        let initial_amp = initial_spectrum[fundamental_bin].norm() * 2.0 / nt as f64;

        let steps = (0.1 / dz) as usize;
        for _ in 0..steps {
            solver.step();
        }

        let final_signal = solver.get_time_signal(config.nx / 2, config.ny / 2);
        let final_signal =
            Array1::from_shape_vec([nt], final_signal).expect("time signal length matches nt");
        let final_spectrum = fft_1d_leto(final_signal.view());
        let final_amp = final_spectrum[fundamental_bin].norm() * 2.0 / nt as f64;

        // Expected amplitude decay: exp(−α·d)
        // α(f₀) in Np/m = α₀_dB_cm_MHz × 100 × DB_TO_NP
        let alpha_np_per_m = config.alpha0 * 100.0 * DB_TO_NP;
        let expected_amp_ratio = (-alpha_np_per_m * 0.1).exp();

        let actual_amp_ratio = if initial_amp > 1e-12 {
            final_amp / initial_amp
        } else {
            1.0
        };

        assert!(
            (actual_amp_ratio - expected_amp_ratio).abs() / expected_amp_ratio < 0.02,
            "Spectral absorption amplitude error: expected {:.5}, got {:.5} \
             (α = {:.4} Np/m, distance = 10 cm)",
            expected_amp_ratio,
            actual_amp_ratio,
            alpha_np_per_m
        );
    }
}
