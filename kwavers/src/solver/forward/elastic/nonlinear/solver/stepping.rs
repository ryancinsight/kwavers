use super::super::wave_field::NonlinearElasticWaveField;
use super::NonlinearElasticWaveSolver;
use ndarray::Zip;

impl NonlinearElasticWaveSolver {
    /// Update fundamental frequency displacement.
    ///
    /// ## Algorithm
    ///
    /// Solves the nonlinear Burgers-like equation per x-aligned line (j, k fixed):
    ///
    /// ```text
    /// ∂u/∂t + (c + β·u/u_ref) · ∂u/∂x = ν · ∂²u/∂x²
    /// ```
    ///
    /// using Heun's method (TVD-RK2) with a **minmod** flux limiter for shock
    /// capturing.  The scheme is TVD (total variation diminishing) and
    /// monotonicity-preserving.
    ///
    /// ## Theorem (Heun TVD-RK2)
    ///
    /// **Stage 1.** Compute interface fluxes F_{i+½} from piecewise-linear
    ///   reconstruction with minmod slopes; advance `u* = u⁰ + Δt·L(u⁰)`.
    ///
    /// **Stage 2.** Recompute fluxes from `u*`; combine:
    ///   `u¹ = ½(u⁰ + u* + Δt·L(u*))`.
    ///
    /// The minmod limiter ensures |TV(u¹)| ≤ |TV(u⁰)| (Harten 1983), preventing
    /// spurious oscillations near shocks.
    ///
    /// ## Parallelization
    ///
    /// Each k-slice `u[:, :, k]` is disjoint.  `Zip::par_for_each` over
    /// `Axis(2)` partitions work across Rayon threads with race-free writes.
    /// Per-thread scratch vectors eliminate shared mutable state.
    ///
    /// ## Reference
    ///
    /// LeVeque RJ (2002). Finite Volume Methods for Hyperbolic Problems.
    /// Cambridge University Press. Ch. 6.
    pub(super) fn update_fundamental_frequency(
        &self,
        field: &mut NonlinearElasticWaveField,
        dt: f64,
    ) {
        let (nx, ny, _nz) = self.grid.dimensions();
        let c = self.config.sound_speed();
        let beta = self.config.nonlinearity_parameter;
        let dissipation = self.config.dissipation_coeff.max(0.0);
        let u_ref = 1e-3;
        let inv_dx = 1.0 / self.grid.dx;
        let inv_dx2 = inv_dx * inv_dx;

        // Minmod flux limiter: picks the smallest-magnitude slope among a, b, c
        // when all three share the same sign; returns 0 otherwise.
        let minmod3 = |a: f64, b: f64, c_: f64| -> f64 {
            if a > 0.0 && b > 0.0 && c_ > 0.0 {
                a.min(b).min(c_)
            } else if a < 0.0 && b < 0.0 && c_ < 0.0 {
                a.max(b).max(c_)
            } else {
                0.0
            }
        };

        // Nonlinear flux: F(u) = c·u + ½·c·β·u²/u_ref
        let flux = |u: f64| -> f64 { c.mul_add(u, 0.5 * c * beta * (u * u) / u_ref) };
        // Local wave speed: a(u) = c + c·β·u/u_ref
        let wave_speed = |u: f64| -> f64 { c + c * beta * u / u_ref };

        // Save current state as previous before overwriting.
        let prev = field.u_fundamental.clone();
        field.u_fundamental_prev.assign(&prev);

        // Parallel over k-slices: each Axis(2) slice is disjoint.
        // Theorem: Zip::par_for_each partitions along Axis(2); slices for
        // distinct k are disjoint by ndarray's striding, so parallel writes
        // to `u_k[[i, j]]` are race-free.
        Zip::from(field.u_fundamental.axis_iter_mut(ndarray::Axis(2)))
            .and(prev.axis_iter(ndarray::Axis(2)))
            .par_for_each(|mut u_k, prev_k| {
                // Per-thread scratch: nx × 6 × 8 bytes — no shared mutable state.
                let mut u_line = vec![0.0f64; nx];
                let mut rhs0 = vec![0.0f64; nx];
                let mut rhs1 = vec![0.0f64; nx];
                let mut u_stage = vec![0.0f64; nx];
                let mut slopes = vec![0.0f64; nx];
                let mut f_iface = vec![0.0f64; nx];

                for j in 0..ny {
                    // Load x-line from previous state.
                    for i in 0..nx {
                        u_line[i] = prev_k[[i, j]];
                    }

                    // ── Stage 1 ──────────────────────────────────────────────
                    // Piecewise-linear reconstruction + minmod slopes.
                    for i in 0..nx {
                        let im1 = (i + nx - 1) % nx;
                        let ip1 = (i + 1) % nx;
                        let du_l = u_line[i] - u_line[im1];
                        let du_r = u_line[ip1] - u_line[i];
                        let du_c = 0.5 * (u_line[ip1] - u_line[im1]);
                        slopes[i] = minmod3(du_c, 2.0 * du_l, 2.0 * du_r);
                    }

                    // Upwind Godunov interface flux.
                    for i in 0..nx {
                        let ip1 = (i + 1) % nx;
                        let u_l = 0.5f64.mul_add(slopes[i], u_line[i]);
                        let u_r = 0.5f64.mul_add(-slopes[ip1], u_line[ip1]);
                        let a = wave_speed(0.5 * (u_l + u_r));
                        f_iface[i] = if a >= 0.0 { flux(u_l) } else { flux(u_r) };
                    }

                    for i in 0..nx {
                        let im1 = (i + nx - 1) % nx;
                        rhs0[i] = -(f_iface[i] - f_iface[im1]) * inv_dx;
                    }

                    for i in 0..nx {
                        u_stage[i] = dt.mul_add(rhs0[i], u_line[i]);
                    }

                    // ── Stage 2 ──────────────────────────────────────────────
                    for i in 0..nx {
                        let im1 = (i + nx - 1) % nx;
                        let ip1 = (i + 1) % nx;
                        let du_l = u_stage[i] - u_stage[im1];
                        let du_r = u_stage[ip1] - u_stage[i];
                        let du_c = 0.5 * (u_stage[ip1] - u_stage[im1]);
                        slopes[i] = minmod3(du_c, 2.0 * du_l, 2.0 * du_r);
                    }

                    for i in 0..nx {
                        let ip1 = (i + 1) % nx;
                        let u_l = 0.5f64.mul_add(slopes[i], u_stage[i]);
                        let u_r = 0.5f64.mul_add(-slopes[ip1], u_stage[ip1]);
                        let a = wave_speed(0.5 * (u_l + u_r));
                        f_iface[i] = if a >= 0.0 { flux(u_l) } else { flux(u_r) };
                    }

                    for i in 0..nx {
                        let im1 = (i + nx - 1) % nx;
                        rhs1[i] = -(f_iface[i] - f_iface[im1]) * inv_dx;
                    }

                    // Heun combination: u¹ = ½(u⁰ + u* + Δt·L(u*))
                    for i in 0..nx {
                        u_line[i] =
                            0.5f64.mul_add(u_line[i], 0.5 * dt.mul_add(rhs1[i], u_stage[i]));
                    }

                    // ── Artificial dissipation ────────────────────────────────
                    if dissipation > 0.0 {
                        let nu = dissipation * c;
                        for i in 0..nx {
                            let ip1 = (i + 1) % nx;
                            let im1 = (i + nx - 1) % nx;
                            let lap =
                                (2.0f64.mul_add(-u_line[i], u_line[ip1]) + u_line[im1]) * inv_dx2;
                            u_line[i] += nu * dt * lap;
                        }
                    }

                    // Write result back into the k-slice.
                    for (i, &u) in u_line.iter().enumerate().take(nx) {
                        u_k[[i, j]] = u;
                    }
                }
            });
    }
}
