use super::super::wave_field::NonlinearElasticWaveField;
use super::NonlinearElasticWaveSolver;

impl NonlinearElasticWaveSolver {
    /// Update fundamental frequency displacement.
    ///
    /// ## Algorithm
    /// Solves the nonlinear Burgers-like equation:
    /// `∂u/∂t + (c + β u/u_ref) ∂u/∂x = ν ∂²u/∂x²`
    ///
    /// using second-order Runge-Kutta (Heun's method) with a **minmod** flux limiter
    /// for shock capturing. The scheme is TVD (total variation diminishing) and
    /// monotonicity-preserving.
    ///
    /// Reference: LeVeque (2002), "Finite Volume Methods", Ch. 6.
    pub(super) fn update_fundamental_frequency(
        &self,
        field: &mut NonlinearElasticWaveField,
        dt: f64,
    ) {
        let (nx, ny, nz) = self.grid.dimensions();
        let c = self.config.sound_speed();
        let beta = self.config.nonlinearity_parameter;
        let dissipation = self.config.dissipation_coeff.max(0.0);
        let u_ref = 1e-3;

        let minmod3 = |a: f64, b: f64, c: f64| -> f64 {
            if a > 0.0 && b > 0.0 && c > 0.0 {
                a.min(b).min(c)
            } else if a < 0.0 && b < 0.0 && c < 0.0 {
                a.max(b).max(c)
            } else {
                0.0
            }
        };

        let flux = |u: f64| -> f64 { c * u + 0.5 * c * beta * (u * u) / u_ref };
        let wave_speed = |u: f64| -> f64 { c + c * beta * u / u_ref };

        let prev = field.u_fundamental.clone();
        field.u_fundamental_prev.assign(&prev);

        let mut u_line = vec![0.0f64; nx];
        let mut rhs0 = vec![0.0f64; nx];
        let mut rhs1 = vec![0.0f64; nx];
        let mut u_stage = vec![0.0f64; nx];
        let mut slopes = vec![0.0f64; nx];
        let mut f_iface = vec![0.0f64; nx];

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    u_line[i] = prev[[i, j, k]];
                }

                // Stage 1: slopes with minmod limiter
                for i in 0..nx {
                    let im1 = (i + nx - 1) % nx;
                    let ip1 = (i + 1) % nx;
                    let du_l = u_line[i] - u_line[im1];
                    let du_r = u_line[ip1] - u_line[i];
                    let du_c = 0.5 * (u_line[ip1] - u_line[im1]);
                    slopes[i] = minmod3(du_c, 2.0 * du_l, 2.0 * du_r);
                }

                for i in 0..nx {
                    let ip1 = (i + 1) % nx;
                    let u_l = u_line[i] + 0.5 * slopes[i];
                    let u_r = u_line[ip1] - 0.5 * slopes[ip1];
                    let a = wave_speed(0.5 * (u_l + u_r));
                    f_iface[i] = if a >= 0.0 { flux(u_l) } else { flux(u_r) };
                }

                let inv_dx = 1.0 / self.grid.dx;
                for i in 0..nx {
                    let im1 = (i + nx - 1) % nx;
                    rhs0[i] = -(f_iface[i] - f_iface[im1]) * inv_dx;
                }

                for i in 0..nx {
                    u_stage[i] = u_line[i] + dt * rhs0[i];
                }

                // Stage 2: second RK stage
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
                    let u_l = u_stage[i] + 0.5 * slopes[i];
                    let u_r = u_stage[ip1] - 0.5 * slopes[ip1];
                    let a = wave_speed(0.5 * (u_l + u_r));
                    f_iface[i] = if a >= 0.0 { flux(u_l) } else { flux(u_r) };
                }

                let inv_dx = 1.0 / self.grid.dx;
                for i in 0..nx {
                    let im1 = (i + nx - 1) % nx;
                    rhs1[i] = -(f_iface[i] - f_iface[im1]) * inv_dx;
                }

                for i in 0..nx {
                    u_line[i] = 0.5 * u_line[i] + 0.5 * (u_stage[i] + dt * rhs1[i]);
                }

                // Artificial dissipation
                if dissipation > 0.0 {
                    let nu = dissipation * c;
                    let inv_dx2 = 1.0 / (self.grid.dx * self.grid.dx);
                    for i in 0..nx {
                        let ip1 = (i + 1) % nx;
                        let im1 = (i + nx - 1) % nx;
                        let lap = (u_line[ip1] - 2.0 * u_line[i] + u_line[im1]) * inv_dx2;
                        u_line[i] += nu * dt * lap;
                    }
                }

                for (i, &u) in u_line.iter().enumerate().take(nx) {
                    field.u_fundamental[[i, j, k]] = u;
                }
            }
        }
    }
}
