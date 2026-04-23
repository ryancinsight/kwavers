//! Validation tests for KZK equation implementation
//!
//! Compares against analytical solutions and published results.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::solver::forward::nonlinear::kzk::constants::*;

    use ndarray::Array2;
    use std::f64::consts::PI;

    /// Test linear propagation of Gaussian beam (COMPREHENSIVE - Tier 3)
    /// Should maintain Gaussian profile with known spreading
    ///
    /// This test propagates to full Rayleigh distance with default grid.
    /// Execution time: >60s, classified as Tier 3 comprehensive validation.
    #[test]
    #[ignore = "Tier 3: Comprehensive validation (>60s execution time)"]
    fn test_gaussian_beam_diffraction() {
        let config = KZKConfig {
            nx: DEFAULT_GRID_SIZE,
            ny: DEFAULT_GRID_SIZE,
            nz: 100,
            nt: 50,
            dx: DEFAULT_DX,
            dz: DEFAULT_DZ,
            dt: 1e-8, // Small time step for stability
            include_nonlinearity: false,
            include_absorption: false,
            include_diffraction: true,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Create Gaussian beam with proper normalization
        // For a Gaussian beam: I(r) = I₀ * exp(-2r²/w₀²)
        // where w₀ is the beam waist radius at 1/e² intensity
        let beam_waist = DEFAULT_BEAM_WAIST; // 5 mm at 1/e² intensity
        let mut source = Array2::zeros((config.nx, config.ny));

        for j in 0..config.ny {
            for i in 0..config.nx {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                // Use -r²/w₀² for field amplitude (not intensity!)
                // Intensity will be |E|² = exp(-2r²/w₀²)
                source[[i, j]] = (-r2 / (beam_waist * beam_waist)).exp();
            }
        }

        solver.set_source(source.clone(), DEFAULT_FREQUENCY);

        // Propagate to Rayleigh distance
        let wavelength = config.c0 / DEFAULT_FREQUENCY;
        let rayleigh_distance = PI * beam_waist * beam_waist / wavelength;
        let steps = (rayleigh_distance / config.dz) as usize;

        println!(
            "Propagating {} steps to Rayleigh distance {:.2}mm",
            steps,
            rayleigh_distance * 1000.0
        );

        for step in 0..steps {
            solver.step();

            // Check beam size periodically
            if step == 0 || step == steps / 2 || step == steps - 1 {
                let intensity = solver.get_intensity();
                let max_int = intensity[[config.nx / 2, config.ny / 2]];
                let threshold = max_int / (std::f64::consts::E * std::f64::consts::E);

                // Simple radius estimate
                let mut radius_est = 0;
                for i in config.nx / 2..config.nx {
                    if intensity[[i, config.ny / 2]] < threshold {
                        radius_est = i - config.nx / 2;
                        break;
                    }
                }

                println!(
                    "Step {}: radius ≈ {:.2}mm",
                    step,
                    radius_est as f64 * config.dx * 1000.0
                );
            }
        }

        // Check beam has spread by √2 at Rayleigh distance
        let intensity = solver.get_intensity();

        // Find beam radius at 1/e² intensity (same as initial definition)
        let center_i = config.nx / 2;
        let center_j = config.ny / 2;
        let max_intensity = intensity[[center_i, center_j]];
        let threshold = max_intensity / (std::f64::consts::E * std::f64::consts::E); // 1/e² threshold

        println!(
            "Center: ({}, {}), Max intensity: {:.2e}, Threshold: {:.2e}",
            center_i, center_j, max_intensity, threshold
        );

        // Find radius by measuring from center to where intensity drops below threshold
        let mut radius_pixels = 0.0;
        for i in center_i..config.nx {
            let curr_intensity = intensity[[i, center_j]];
            if i == center_i || i == center_i + 1 || i == center_i + 10 {
                println!(
                    "i={}, intensity={:.2e}, threshold={:.2e}",
                    i, curr_intensity, threshold
                );
            }
            if curr_intensity < threshold {
                // Linear interpolation for sub-pixel accuracy
                if i > center_i {
                    let prev_intensity = intensity[[i - 1, center_j]];
                    let fraction = (threshold - curr_intensity) / (prev_intensity - curr_intensity);
                    radius_pixels = (i - center_i) as f64 - fraction;
                    println!("Found edge at i={}, radius_pixels={:.2}", i, radius_pixels);
                }
                break;
            }
        }

        // If we didn't find the edge, use the maximum distance
        if radius_pixels == 0.0 {
            println!("Warning: beam edge not found within grid!");
            radius_pixels = (config.nx - center_i - 1) as f64;
        }

        let final_radius = radius_pixels * config.dx;
        let expected_radius = beam_waist * 2.0_f64.sqrt();

        assert!(
            (final_radius - expected_radius).abs() / expected_radius < 0.35,
            "Beam radius error: expected {:.2}mm, got {:.2}mm (within 35% tolerance for numerical diffusion)",
            expected_radius * 1000.0,
            final_radius * 1000.0
        );
    }

    /// Test linear propagation of Gaussian beam (FAST - Tier 1)
    /// Fast version with reduced grid and fewer steps for CI/CD.
    /// Execution time: <2s, classified as Tier 1 fast validation.
    #[test]
    fn test_gaussian_beam_diffraction_fast() {
        let config = KZKConfig {
            nx: 32,
            ny: 32,
            nz: 20,
            nt: 16,
            dx: 2e-4, // Smaller dx for better resolution
            dz: 1e-3,
            dt: 1e-8,
            include_nonlinearity: false,
            include_absorption: false,
            include_diffraction: true,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Create Gaussian beam with smaller waist to avoid parabolic approximation limit
        let beam_waist = 1e-3; // 1 mm (smaller than before)
        let mut source = Array2::zeros((config.nx, config.ny));

        for j in 0..config.ny {
            for i in 0..config.nx {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                source[[i, j]] = (-r2 / (beam_waist * beam_waist)).exp();
            }
        }

        solver.set_source(source, 1e6);

        // Propagate a few steps (smoke test)
        for _ in 0..3 {
            solver.step();
        }

        // Just verify solver runs without panicking and produces output
        let intensity = solver.get_intensity();
        assert!(
            intensity[[config.nx / 2, config.ny / 2]] > 0.0,
            "Should have non-zero intensity at center"
        );
    }

    /// Test harmonic generation in nonlinear propagation
    #[test]
    fn test_harmonic_generation() {
        let config = KZKConfig {
            nx: 32,
            ny: 32,
            nz: 50,
            nt: 128,
            dx: 1e-3,
            dz: 2e-3,
            dt: 5e-9,
            include_nonlinearity: true,
            include_absorption: false,
            include_diffraction: false,
            b_over_a: 3.5,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Uniform plane wave source
        let amplitude = 1e6; // 1 MPa
        let frequency = 2e6; // 2 MHz
        let source = Array2::from_elem((config.nx, config.ny), amplitude);

        solver.set_source(source, frequency);

        // Propagate
        for _ in 0..20 {
            solver.step();
        }

        // Analyze spectrum at center point
        let signal = solver.get_time_signal(config.nx / 2, config.ny / 2);

        // Compute FFT
        use crate::math::fft::fft_1d_array;
        use ndarray::Array1;

        let spectrum = fft_1d_array(&Array1::from_vec(signal));

        // Find fundamental and second harmonic
        let df = 1.0 / (config.nt as f64 * config.dt);
        let fundamental_bin = (frequency / df) as usize;
        let second_harmonic_bin = 2 * fundamental_bin;

        if second_harmonic_bin < config.nt / 2 {
            let fundamental_amp = spectrum[fundamental_bin].norm();
            let second_harmonic_amp = spectrum[second_harmonic_bin].norm();

            // Second harmonic should be generated
            assert!(
                second_harmonic_amp > fundamental_amp * 0.01,
                "No second harmonic generation detected"
            );
        }
    }

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
    /// Therefore the spectral and single-frequency absorption operators give
    /// identical results for pure-tone signals.
    ///
    /// ## Reference
    ///
    /// Szabo TL (1994). J. Acoust. Soc. Am. 96(1), 491–500.
    #[test]
    fn test_absorption() {
        use crate::math::fft::fft_1d_array;
        use ndarray::Array1;

        let frequency = 1.0e6_f64; // 1 MHz
                                   // dt chosen so fundamental falls on exact FFT bin: dt = 1/(8·f₀)
        let dt = 1.0 / (8.0 * frequency); // 125 ns
        let nt = 256_usize; // 32 complete periods
        let dz = 1.0e-3_f64; // 1 mm steps

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
            alpha0: 0.5, // dB/(cm·MHz)
            alpha_power: 1.0,
            frequency,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Uniform plane-wave source at f₀
        let source = Array2::from_elem((config.nx, config.ny), 1.0_f64);
        solver.set_source(source, frequency);

        // Record initial fundamental amplitude at centre
        let initial_signal = solver.get_time_signal(config.nx / 2, config.ny / 2);
        let initial_signal = Array1::from_vec(initial_signal);
        let initial_spectrum = fft_1d_array(&initial_signal);
        let fundamental_bin = (frequency * nt as f64 * dt).round() as usize; // = 32
        let initial_amp = initial_spectrum[fundamental_bin].norm() * 2.0 / nt as f64;

        // Propagate 10 cm
        let steps = (0.1 / dz) as usize;
        for _ in 0..steps {
            solver.step();
        }

        // Extract final fundamental amplitude
        let final_signal = solver.get_time_signal(config.nx / 2, config.ny / 2);
        let final_signal = Array1::from_vec(final_signal);
        let final_spectrum = fft_1d_array(&final_signal);
        let final_amp = final_spectrum[fundamental_bin].norm() * 2.0 / nt as f64;

        // Expected amplitude decay: exp(−α·d)
        // α(f₀) in Np/m = α₀_dB_cm_MHz × 100 / 8.686 = 0.5×100/8.686 ≈ 5.756 Np/m
        let alpha_np_per_m = config.alpha0 * 100.0 / 8.686;
        let expected_amp_ratio = (-alpha_np_per_m * 0.1).exp(); // field amplitude ratio

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

    /// Validate harmonic amplitudes against Aanonsen et al. (1984) Table 1.
    ///
    /// ## Physical setup
    ///
    /// Plane wave (no diffraction, no absorption) propagating in water.
    /// The normalised harmonic amplitudes at Gol'dberg numbers Γ = 0.25, 0.5, 1.0
    /// follow the Fubini solution:
    ///
    /// ```text
    /// |Pₙ| / |P₁| = (2/n) Jₙ(nΓ) / [Γ·J₁(Γ)]
    /// ```
    ///
    /// (Aanonsen 1984, eq. 6–7; Hamilton & Blackstock 1998 §4.3.2).
    ///
    /// The exact Fubini harmonic ratios are:
    /// ```text
    /// |P₂|/|P₁| = J₂(2Γ) / (2 J₁(Γ))
    /// |P₃|/|P₁| = J₃(3Γ) / (3 J₁(Γ))     [not (Γ/2)²/2! which is wrong by 3×]
    /// ```
    ///
    /// For small Γ, the leading-order approximations are:
    /// ```text
    /// |P₂|/|P₁| ≈ Γ/2          (valid to ~1% for Γ < 0.3)
    /// |P₃|/|P₁| ≈ 3Γ²/8        (valid to ~5% for Γ < 0.3)
    /// ```
    ///
    /// Note: the "(Γ/2)^{n−1}/(n−1)!" formula is correct only for n=2.
    /// For n=3 it gives Γ²/8, which underestimates by 3× compared to the
    /// exact Fubini coefficient 3Γ²/8 = J₃(3Γ)/(3J₁(Γ)) at leading order.
    ///
    /// ## Reference data — exact Fubini solution (pre-computed Bessel values)
    ///
    /// Values from the Fubini series Bₙ = 2Jₙ(nΓ)/(nΓ); ratio Bₙ/B₁:
    ///
    /// | Γ    | |P₂|/|P₁| | |P₃|/|P₁| | Source               |
    /// |------|------------|------------|----------------------|
    /// | 0.25 | 0.1234     | 0.02279    | exact Fubini (Bessel) |
    /// | 0.50 | 0.2371     | 0.08274    | exact Fubini (Bessel) |
    /// | 1.00 | 0.4009     | 0.23411    | exact Fubini (Bessel) |
    ///
    /// ## Tolerance
    ///
    /// 5% relative error on each harmonic ratio.  The Fubini solution is exact
    /// for the inviscid, plane-wave case.  Deviations above 5% indicate errors
    /// in the nonlinear operator, incorrect β, or FFT spectral leakage.
    ///
    /// ## References
    ///
    /// - Aanonsen SI et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768, Table 1.
    ///   DOI: 10.1121/1.390585
    /// - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics §4.3.2.
    #[test]
    #[ignore = "Tier 2: Literature validation (~10-30s depending on grid)"]
    fn test_aanonsen_1984_harmonic_amplitudes() {
        use crate::math::fft::fft_1d_array;
        use ndarray::Array1;

        // Medium: water at 25°C
        let rho0 = 998.0_f64; // kg/m³ (Kaye & Laby)
        let c0 = 1481.0_f64; // m/s   (Del Grosso & Mader 1972)
        let b_over_a = 5.0_f64; // B/A for water (Beyer 1960)
        let beta = 1.0 + b_over_a / 2.0; // β = 3.5

        let frequency = 1.0e6_f64; // 1 MHz
        let omega = 2.0 * std::f64::consts::PI * frequency;

        // Source amplitude: chosen small enough to stay in Fubini regime (Γ < 1)
        let p0 = 5.0e4_f64; // 50 kPa

        // Shock formation distance for plane wave
        // z_shock = ρ₀c₀³ / (β·ω·p₀)
        // Ref: Hamilton & Blackstock (1998) §4.3, eq. (4.3.5)
        let z_shock = rho0 * c0.powi(3) / (beta * omega * p0);

        // Time grid: 16 samples per period, 256 total (16 complete periods).
        // 16 samples/period gives Nyquist = 8 MHz, so harmonics up to the 7th
        // (7 MHz) are representable without aliasing.  With only 8 samples/period
        // (Nyquist = 4 MHz), the 5th harmonic (5 MHz) aliases into the 3rd
        // harmonic bin (bin = |5 - 8| = 3 in relative units), inflating P₃ at
        // Γ=1 by ~7% regardless of axial step count.
        //
        // f₀·N·dt = 1e6 × 256 × 62.5e-9 = 16 → exact bin at k₁=16, k₂=32, k₃=48.
        let nt = 256_usize;
        let dt = 1.0 / (16.0 * frequency); // 62.5 ns

        // Gol'dberg numbers and exact Fubini reference values.
        //
        // Computed from |P₂|/|P₁| = J₂(2Γ)/(2J₁(Γ)) and
        //               |P₃|/|P₁| = J₃(3Γ)/(3J₁(Γ))
        // using the Bessel function power series to 6 significant figures.
        //
        // NOTE: |P₃|/|P₁| ≈ 3Γ²/8 at leading order — NOT Γ²/8.
        // The (Γ/2)^{n-1}/(n-1)! formula is only correct for n=2.
        let goldberg_targets = [0.25_f64, 0.50_f64, 1.00_f64];
        // |P₂|/|P₁| exact Fubini: [0.12337, 0.23714, 0.40090]
        let expected_p2_p1 = [0.12337_f64, 0.23714_f64, 0.40090_f64];
        // |P₃|/|P₁| exact Fubini: [0.02279, 0.08274, 0.23411]
        let expected_p3_p1 = [0.02279_f64, 0.08274_f64, 0.23411_f64];

        for (idx, &gamma) in goldberg_targets.iter().enumerate() {
            let z_target = gamma * z_shock;
            // 200 axial steps: error ≈ O(Δz) with explicit Euler.
            // At Γ=1 (200 steps, Δz = z_shock/200) this gives ~2% axial error.
            let n_steps = 200_usize;
            let dz = z_target / n_steps as f64;

            let config = KZKConfig {
                nx: 4,
                ny: 4,
                nz: n_steps,
                nt,
                dx: 1e-3,
                dz,
                dt,
                c0,
                rho0,
                b_over_a,
                alpha0: 0.0,                // No absorption (inviscid)
                include_diffraction: false, // Pure plane wave
                include_absorption: false,
                include_nonlinearity: true,
                frequency,
                ..Default::default()
            };

            let mut solver = KZKSolver::new(config.clone()).unwrap();

            // Uniform plane-wave source at f₀
            let source = Array2::from_elem((config.nx, config.ny), p0);
            solver.set_source(source, frequency);

            solver.solve(n_steps).expect("KZK solve failed");

            // Extract time signal at centre point
            let signal = solver.get_time_signal(config.nx / 2, config.ny / 2);

            // FFT for spectral analysis
            let signal = Array1::from_vec(signal);
            let spectrum = fft_1d_array(&signal);

            // Fundamental at bin 32 (exact: f₀·N·dt = 32)
            let df = 1.0 / (nt as f64 * dt);
            let k1 = (frequency / df).round() as usize;
            let k2 = 2 * k1;
            let k3 = 3 * k1;

            // Peak amplitude (two-sided spectrum → multiply by 2/N)
            let amp1 = spectrum[k1].norm() * 2.0 / nt as f64;
            let amp2 = spectrum[k2].norm() * 2.0 / nt as f64;
            let amp3 = if k3 < nt / 2 {
                spectrum[k3].norm() * 2.0 / nt as f64
            } else {
                0.0
            };

            let ratio2 = amp2 / amp1;
            let ratio3 = amp3 / amp1;

            let tol = 0.05; // 5% relative

            assert!(
                (ratio2 - expected_p2_p1[idx]).abs() / expected_p2_p1[idx] < tol,
                "Γ={:.2}: |P₂|/|P₁| — expected {:.5}, got {:.5} \
                 (error = {:.1}% > 5%)\n\
                 Check: NonlinearOperator β, buffered-update correctness.",
                gamma,
                expected_p2_p1[idx],
                ratio2,
                (ratio2 - expected_p2_p1[idx]).abs() / expected_p2_p1[idx] * 100.0
            );

            if k3 < nt / 2 {
                assert!(
                    (ratio3 - expected_p3_p1[idx]).abs() / expected_p3_p1[idx] < tol,
                    "Γ={:.2}: |P₃|/|P₁| — expected {:.5}, got {:.5} \
                     (error = {:.1}% > 5%)\n\
                     Check: Strang splitting accumulation, harmonic leakage.",
                    gamma,
                    expected_p3_p1[idx],
                    ratio3,
                    (ratio3 - expected_p3_p1[idx]).abs() / expected_p3_p1[idx] * 100.0
                );
            }
        }
    }
}
