use super::super::wave_field::NonlinearElasticWaveField;
use super::NonlinearElasticWaveSolver;
use core::cell::RefCell;
use moirai_parallel::{map_collect_index_with, Adaptive};

/// Per-worker scratch buffers for one x-line update.
///
/// The global executor's workers persist across calls, so a thread-local
/// amortizes these allocations to once per worker instead of five `nx`-length
/// vectors per x-line per timestep. Each buffer grows monotonically to the
/// largest `nx` seen on that worker and is fully rewritten from the previous
/// state on every use, so reuse carries no state between lines.
struct LineScratch {
    rhs0: Vec<f64>,
    rhs1: Vec<f64>,
    u_stage: Vec<f64>,
    slopes: Vec<f64>,
    f_iface: Vec<f64>,
}

thread_local! {
    static LINE_SCRATCH: RefCell<LineScratch> = const {
        RefCell::new(LineScratch {
            rhs0: Vec::new(),
            rhs1: Vec::new(),
            u_stage: Vec::new(),
            slopes: Vec::new(),
            f_iface: Vec::new(),
        })
    };
}

/// Grow `buf` to at least `n` zeroed entries (monotonic per worker) and view it.
fn scratch_buf(buf: &mut Vec<f64>, n: usize) -> &mut [f64] {
    if buf.len() < n {
        buf.clear();
        buf.resize(n, 0.0);
    }
    &mut buf[..n]
}

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
    /// Each `(j, k)` x-line is independent. Moirai computes line updates from
    /// the immutable previous field, then a separate write-back pass updates
    /// the non-contiguous x-lines without unsafe strided mutable aliasing.
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
        let (nx, ny, nz) = self.grid.dimensions();
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

        // Rotate the pre-step state into the history buffer without copying:
        // after the swap, `u_fundamental_prev` holds the state the update reads
        // and `u_fundamental` holds stale data that the write-back pass below
        // overwrites for every (i, j, k). The former clone-then-assign paid two
        // full-field copies per timestep for the same result.
        core::mem::swap(&mut field.u_fundamental, &mut field.u_fundamental_prev);

        let line_updates = map_collect_index_with::<Adaptive, _, _>(ny * nz, |line_index| {
            let j = line_index / nz;
            let k = line_index % nz;

            // The result line is the one buffer this design still allocates: it
            // is returned for the write-back pass, which deliberately keeps the
            // strided field writes out of the parallel closure.
            let mut u_line = vec![0.0f64; nx];

            LINE_SCRATCH.with(|cell| {
                let scratch = &mut *cell.borrow_mut();
                let rhs0 = scratch_buf(&mut scratch.rhs0, nx);
                let rhs1 = scratch_buf(&mut scratch.rhs1, nx);
                let u_stage = scratch_buf(&mut scratch.u_stage, nx);
                let slopes = scratch_buf(&mut scratch.slopes, nx);
                let f_iface = scratch_buf(&mut scratch.f_iface, nx);

                // Load x-line from previous state.
                for i in 0..nx {
                    u_line[i] = field.u_fundamental_prev[[i, j, k]];
                }

                // Stage 1: piecewise-linear reconstruction + minmod slopes.
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

                // Stage 2.
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
                    u_line[i] = 0.5f64.mul_add(u_line[i], 0.5 * dt.mul_add(rhs1[i], u_stage[i]));
                }

                // Artificial dissipation.
                if dissipation > 0.0 {
                    let nu = dissipation * c;
                    for i in 0..nx {
                        let ip1 = (i + 1) % nx;
                        let im1 = (i + nx - 1) % nx;
                        let lap = (2.0f64.mul_add(-u_line[i], u_line[ip1]) + u_line[im1]) * inv_dx2;
                        u_line[i] += nu * dt * lap;
                    }
                }

                (j, k, u_line)
            })
        });

        for (j, k, u_line) in line_updates {
            for (i, u) in u_line.into_iter().enumerate().take(nx) {
                field.u_fundamental[[i, j, k]] = u;
            }
        }
    }
}
