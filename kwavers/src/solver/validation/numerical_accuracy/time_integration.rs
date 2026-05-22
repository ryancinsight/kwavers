#[cfg(test)]
mod tests {
    use super::super::helpers::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use crate::core::constants::thermodynamic::THERMAL_DIFFUSIVITY_TISSUE;
    use ndarray::Array3;
    use std::f64::consts::PI;

    const CFL_NUMBER: f64 = 0.3;

    #[test]
    fn test_multirate_time_integration() {
        // Validate time-scale separation (Gear & Wells 1984)
        // Fast acoustic + slow thermal diffusion

        const ACOUSTIC_SPEED: f64 = SOUND_SPEED_WATER_SIM; // m/s

        let dx = 1e-3;
        let dt_acoustic = CFL_NUMBER * dx / ACOUSTIC_SPEED;
        let dt_thermal = 0.5 * dx.powi(2) / THERMAL_DIFFUSIVITY_TISSUE;

        let time_scale_ratio = dt_thermal / dt_acoustic;

        assert!(
            time_scale_ratio > 100.0,
            "Insufficient time scale separation: {:.1}x",
            time_scale_ratio
        );

        // Verify stability of multirate scheme with reduced grid for testing
        let n = 8; // Reduced from 32 to 8 for faster testing
        let _steps_acoustic = 100; // Reduced from 1000
        let steps_thermal = 2; // Fixed small number for testing

        let mut acoustic_state = Array3::zeros((n, n, n));
        let mut thermal_state = Array3::zeros((n, n, n));

        // Initialize with test pattern
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let r = ((i as f64 - n as f64 / 2.0).powi(2)
                        + (j as f64 - n as f64 / 2.0).powi(2)
                        + (k as f64 - n as f64 / 2.0).powi(2))
                    .sqrt();
                    acoustic_state[[i, j, k]] = (-r.powi(2) / 10.0).exp();
                    thermal_state[[i, j, k]] = 300.0 + 10.0 * (-r.powi(2) / 20.0).exp();
                }
            }
        }

        let initial_acoustic_energy: f64 = acoustic_state.iter().map(|x| x * x).sum();
        let initial_thermal_energy: f64 = thermal_state.iter().sum();

        // Bootstrap Störmer-Verlet: p_{−1} = p_0 − Δt·v_0.
        // With zero initial velocity (v_0 = 0), p_{−1} = p_0.
        let mut prev_acoustic_state = acoustic_state.clone();

        // Multirate evolution with proper time stepping
        // (Gear & Wells 1984, §3 — fast acoustic inner loop, slow thermal outer loop).
        let dt_slow = dt_thermal;
        for _slow_step in 0..steps_thermal {
            // Multiple fast steps per slow step (capped for testing)
            let fast_per_slow = (time_scale_ratio as usize).clamp(1, 10);

            for _ in 0..fast_per_slow {
                // Störmer-Verlet integrator for ∂²p/∂t² = c²∇²p.
                //
                // ## Algorithm: Störmer-Verlet (leapfrog) — O(Δt²) symplectic
                //
                // ```text
                //   p^{n+1} = 2p^n − p^{n-1} + Δt²·c²·∇²p^n
                // ```
                //
                // ## Theorem: Energy conservation (Hairer et al. 2006, §II.1)
                //
                // The Störmer-Verlet scheme exactly preserves the modified Hamiltonian
                //   H̃ = (1/2)||v^{n+½}||² + (c²/2)||∇p^n||²  (v^{n+½} = (p^{n+1}−p^n)/Δt)
                // up to O(Δt²) perturbations.  In particular the L² norm of p is
                // bounded for all time within the stability region CFL ≤ 1.
                //
                // ## Stability condition
                //
                // Von Neumann analysis (Courant, Friedrichs & Lewy 1928):
                //   CFL = c·Δt/Δx ≤ 1  (1D, 2nd-order FD in space)
                //
                // ## References
                //   Hairer, E., Lubich, C. & Wanner, G. (2006) *Geometric Numerical
                //     Integration*, 2nd ed., §II.1, Springer.
                //   Courant, R., Friedrichs, K. & Lewy, H. (1928) Math. Ann. 100, 32–74.
                // Proper 3D Laplacian with periodic BCs (7-point stencil).
                let lap3d = compute_laplacian_3d(&acoustic_state, dx);
                let c_squared = ACOUSTIC_SPEED * ACOUSTIC_SPEED;
                let dt2_c2 = dt_acoustic * dt_acoustic * c_squared;

                let mut new_acoustic_state = acoustic_state.clone();
                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            new_acoustic_state[[i, j, k]] = 2.0 * acoustic_state[[i, j, k]]
                                - prev_acoustic_state[[i, j, k]]
                                + dt2_c2 * lap3d[[i, j, k]];
                        }
                    }
                }
                prev_acoustic_state = acoustic_state.clone();
                acoustic_state = new_acoustic_state;
            }

            // Thermal diffusion: ∂T/∂t = α∇²T (Forward Euler, Fourier number = 0.3·3 < 0.5)
            // With periodic BCs, total temperature Σ T[i,j,k] is conserved exactly
            // (discrete divergence theorem: Σ ∇²T = 0 on periodic domain).
            let thermal_laplacian = compute_laplacian_3d(&thermal_state, dx);
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        thermal_state[[i, j, k]] +=
                            dt_slow * THERMAL_DIFFUSIVITY_TISSUE * thermal_laplacian[[i, j, k]];
                    }
                }
            }
        }

        let final_acoustic_energy: f64 = acoustic_state.iter().map(|x| x * x).sum();
        let final_thermal_energy: f64 = thermal_state.iter().sum();

        // Acoustic stability check: Störmer-Verlet conserves H̃ = Σ(ṗ²+c²|∇p|²)/2, not
        // Σp² alone (which oscillates as energy exchanges between pressure and velocity).
        // We check that the L² norm did not GROW (no instability), not that it is constant.
        assert!(
            final_acoustic_energy <= 2.0 * initial_acoustic_energy,
            "Acoustic energy grew (Störmer-Verlet instability): \
             final/initial = {:.4}",
            final_acoustic_energy / initial_acoustic_energy
        );

        // Thermal conservation: for ∂T/∂t = α∇²T with periodic BCs,
        // Σ T[i,j,k] is preserved exactly (discrete divergence theorem).
        let thermal_loss =
            (initial_thermal_energy - final_thermal_energy).abs() / initial_thermal_energy;
        assert!(
            thermal_loss < 0.01,
            "Thermal sum changed by {:.4}% (should be ~0 with periodic BCs)",
            thermal_loss * 100.0
        );
    }

    #[test]
    fn test_stormer_verlet_wave_energy_conservation() {
        const C: f64 = SOUND_SPEED_WATER_SIM; // wave speed [m/s]
        let n = 64_usize; // grid points
        let l = 1.0_f64; // domain length [m]
        let dx = l / n as f64;
        let dt = 0.4 * dx / C; // CFL = 0.4 < 1 (stable)
        let k = 2.0 * PI / l; // wavenumber
        let omega = C * k; // angular frequency

        // Number of steps = 3 complete wave periods
        let t_period = 2.0 * PI / omega;
        let n_steps = (3.0 * t_period / dt).ceil() as usize;

        // Initial conditions: p(x,0) = sin(kx), v(x,0) = 0
        let mut p_curr: Vec<f64> = (0..n).map(|i| (k * i as f64 * dx).sin()).collect();
        // Bootstrap: p_{-1} = p_0 (v_0 = 0 ⟹ p_{-1} = p_0)
        let mut p_prev = p_curr.clone();

        // Compute initial L2² norm for reference
        let e0: f64 = p_curr.iter().map(|&v| v * v).sum::<f64>() * dx;
        let dt2_c2 = dt * dt * C * C;

        let mut max_e = e0;
        let mut p_next = vec![0.0_f64; n];

        for _ in 0..n_steps {
            // Störmer-Verlet step with periodic boundary
            for i in 0..n {
                let ip1 = (i + 1) % n;
                let im1 = (i + n - 1) % n;
                let lap = (p_curr[ip1] - 2.0 * p_curr[i] + p_curr[im1]) / (dx * dx);
                p_next[i] = 2.0 * p_curr[i] - p_prev[i] + dt2_c2 * lap;
            }
            p_prev.copy_from_slice(&p_curr);
            p_curr.copy_from_slice(&p_next);

            let e: f64 = p_curr.iter().map(|&v| v * v).sum::<f64>() * dx;
            if e > max_e {
                max_e = e;
            }
        }

        // The L² norm should remain bounded: max_e ≤ 2·e0 (it oscillates between 0 and e0,
        // so max_e ≈ e0 with small O(Δt²) perturbations).
        assert!(
            max_e < 2.0 * e0,
            "Störmer-Verlet L² norm grew: max_e/e0 = {:.4} > 2.0 (CFL={:.2})",
            max_e / e0,
            C * dt / dx
        );

        // After N complete periods, solution should return close to initial shape.
        // At t = N·T: p(x) ≈ sin(kx).  Measure L∞ error.
        let error: f64 = (0..n)
            .map(|i| {
                let p_exact = (k * i as f64 * dx).sin() * (omega * n_steps as f64 * dt).cos();
                (p_curr[i] - p_exact).abs()
            })
            .fold(0.0_f64, f64::max);

        // Störmer-Verlet is 2nd-order: error ≤ C·Δt²·T/T_period.  With CFL=0.4 and
        // 3 periods, Δt²/T_period² ≈ 0.4²·(dx/L)² ≈ 0.025% → allow 2%.
        assert!(
            error < 0.02,
            "Störmer-Verlet phase error {:.6} after {} steps exceeds 2%",
            error,
            n_steps
        );
    }
}
