//! Nonlinear propagation validation tests for KZK equation implementation.

#[cfg(test)]
mod tests {
    use super::super::super::*;
    use ndarray::Array2;

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

        let amplitude = 1e6;
        let frequency = 2e6;
        let source = Array2::from_elem((config.nx, config.ny), amplitude);

        solver.set_source(source, frequency);

        for _ in 0..20 {
            solver.step();
        }

        let signal = solver.get_time_signal(config.nx / 2, config.ny / 2);

        use crate::math::fft::fft_1d_array;
        use ndarray::Array1;

        let spectrum = fft_1d_array(&Array1::from_vec(signal));

        let df = 1.0 / (config.nt as f64 * config.dt);
        let fundamental_bin = (frequency / df) as usize;
        let second_harmonic_bin = 2 * fundamental_bin;

        if second_harmonic_bin < config.nt / 2 {
            let fundamental_amp = spectrum[fundamental_bin].norm();
            let second_harmonic_amp = spectrum[second_harmonic_bin].norm();

            assert!(
                second_harmonic_amp > fundamental_amp * 0.01,
                "No second harmonic generation detected"
            );
        }
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
    /// ## Reference data — exact Fubini solution (pre-computed Bessel values)
    ///
    /// | Γ    | |P₂|/|P₁| | |P₃|/|P₁| | Source               |
    /// |------|------------|------------|----------------------|
    /// | 0.25 | 0.1234     | 0.02279    | exact Fubini (Bessel) |
    /// | 0.50 | 0.2371     | 0.08274    | exact Fubini (Bessel) |
    /// | 1.00 | 0.4009     | 0.23411    | exact Fubini (Bessel) |
    ///
    /// ## Tolerance
    ///
    /// 5% relative error on each harmonic ratio.
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

        let rho0 = 998.0_f64;
        let c0 = 1481.0_f64;
        let b_over_a = 5.0_f64;
        let beta = 1.0 + b_over_a / 2.0;

        let frequency = 1.0e6_f64;
        let omega = 2.0 * std::f64::consts::PI * frequency;

        let p0 = 5.0e4_f64;

        let z_shock = rho0 * c0.powi(3) / (beta * omega * p0);

        let nt = 256_usize;
        let dt = 1.0 / (16.0 * frequency);

        // Exact Fubini reference values (Bessel function evaluations)
        let goldberg_targets = [0.25_f64, 0.50_f64, 1.00_f64];
        let expected_p2_p1 = [0.12337_f64, 0.23714_f64, 0.40090_f64];
        let expected_p3_p1 = [0.02279_f64, 0.08274_f64, 0.23411_f64];

        for (idx, &gamma) in goldberg_targets.iter().enumerate() {
            let z_target = gamma * z_shock;
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
                alpha0: 0.0,
                include_diffraction: false,
                include_absorption: false,
                include_nonlinearity: true,
                frequency,
                ..Default::default()
            };

            let mut solver = KZKSolver::new(config.clone()).unwrap();

            let source = Array2::from_elem((config.nx, config.ny), p0);
            solver.set_source(source, frequency);

            solver.solve(n_steps).expect("KZK solve failed");

            let signal = solver.get_time_signal(config.nx / 2, config.ny / 2);
            let signal = Array1::from_vec(signal);
            let spectrum = fft_1d_array(&signal);

            let df = 1.0 / (nt as f64 * dt);
            let k1 = (frequency / df).round() as usize;
            let k2 = 2 * k1;
            let k3 = 3 * k1;

            let amp1 = spectrum[k1].norm() * 2.0 / nt as f64;
            let amp2 = spectrum[k2].norm() * 2.0 / nt as f64;
            let amp3 = if k3 < nt / 2 {
                spectrum[k3].norm() * 2.0 / nt as f64
            } else {
                0.0
            };

            let ratio2 = amp2 / amp1;
            let ratio3 = amp3 / amp1;

            let tol = 0.05;

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
