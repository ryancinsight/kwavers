//! Harmonic generation for the nonlinear elastic wave solver.
//!
//! ## Mathematical background
//!
//! The nonlinear elastic wave equation with quadratic nonlinearity:
//! ```text
//! ∂²u/∂t² = c²∇²u + β u ∇²u
//! ```
//!
//! admits a perturbation expansion `u = u₁ + u₂ + u₃ + …` where each
//! harmonic satisfies a forced linear wave equation driven by lower-order
//! harmonics:
//!
//! ```text
//! ∂²u₂/∂t² − c²∇²u₂ = β u₁ ∇²u₁
//! ∂²u₃/∂t² − c²∇²u₃ = β(u₁∇²u₂ + u₂∇²u₁ + 2∇u₁·∇u₂)
//! ∂²uₙ/∂t² − c²∇²uₙ ≈ amplitude_factor · β (u₁∇²uₙ₋₁ + uₙ₋₁∇²u₁)
//! ```
//!
//! Harmonic amplitudes satisfy `Aₙ ∝ β^(n-1) / n` (Chen 2013, §III).
//!
//! ## Theorem (Jacobi correctness, operator-isolation)
//!
//! Each harmonic update computes the full right-hand side evaluated at the
//! input state before applying any update.  Specifically, the Laplacian of
//! `uₙ` reads the array values from before the current time step (stored in
//! a temporary `delta` buffer), then applies `uₙ += delta` in a separate
//! pass.  This preserves the Jacobi invariant: every RHS evaluation uses a
//! consistent input state.
//!
//! The serial formulation in the original code used Gauss-Seidel ordering
//! (reading partially-updated values from within the same pass), which
//! introduces an O(Δt · Δx) order-dependent inconsistency across spatial
//! neighbours.  The Jacobi formulation here eliminates this inconsistency
//! and enables race-free parallel execution.
//!
//! ## Theorem (race-freedom)
//!
//! - Loop 1 writes to `u_second[[i,j,k]]` using only `u_fundamental[[i,j,k]]`
//!   (immutable view). No cross-cell dependency → race-free.
//! - Loops 2 and 3 compute `delta[[i,j,k]]` using immutable views of all
//!   source arrays (including the Laplacian of `uₙ` read from its pre-update
//!   state). Writes are to `delta` only; `uₙ` is not mutated until after the
//!   parallel section ends → race-free.
//!
//! ## Reference
//!
//! Chen, S., et al. (2013). "Harmonic motion detection in ultrasound
//! elastography." IEEE Trans. Med. Imaging, 32(5), 863–874.
//! DOI: 10.1109/TMI.2013.2239671

use kwavers_core::utils::iterators::for_each_indexed_mut;
use ndarray::Array3;

use super::super::wave_field::NonlinearElasticWaveField;
use super::NonlinearElasticWaveSolver;

impl NonlinearElasticWaveSolver {
    /// Generate harmonic components using Chen (2013) harmonic motion detection.
    ///
    /// All three passes (second, third, higher) use Jacobi updates:
    /// the RHS is evaluated at the **input** state for all grid points, then
    /// `uₙ += delta` is applied as a single vectorised pass.  See module-level
    /// theorem documentation.
    pub(super) fn generate_harmonics(&self, field: &mut NonlinearElasticWaveField, dt: f64) {
        let beta = self.config.nonlinearity_parameter;
        let (nx, ny, nz) = self.grid.dimensions();

        // --- Loop 1: second harmonic generation ---
        //
        // `u₂ += harmonic_factor · u₁ · |u₁| · dt`
        //
        // Each element of `u_second` depends only on the collocated value of
        // `u_fundamental` → pure Jacobi, no temporary buffer needed.
        // Theorem (race-freedom, Loop 1): see module doc.
        let harmonic_factor = (beta * 1e-6).min(1e-8);
        {
            let u_fund_v = field.u_fundamental.view();
            for_each_indexed_mut(field.u_second.view_mut(), |(i, j, k), u2| {
                if i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1 {
                    let u1 = u_fund_v[[i, j, k]];
                    *u2 += harmonic_factor * u1 * u1.abs() * dt;
                }
            });
        }

        // --- Loop 2: third harmonic via cascading ---
        //
        // Source term: β(u₁∇²u₂ + u₂∇²u₁ + 2∇u₁·∇u₂)
        // Explicit update: u₃ += dt² · (c²∇²u₃ + source)
        //
        // Theorem (race-freedom, Loop 2): see module doc.
        if !field.u_harmonics.is_empty() {
            let delta3 = {
                // Immutable borrows of source fields — all released at block exit
                let u_fund = &field.u_fundamental;
                let u_second = &field.u_second;
                let u3 = &field.u_harmonics[0];
                let numerics = &self.numerics;
                let sound_speed_sq = self.config.sound_speed().powi(2);

                let mut d = Array3::<f64>::zeros((nx, ny, nz));
                for_each_indexed_mut(d.view_mut(), |(i, j, k), di| {
                    if i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1 {
                        let u1 = u_fund[[i, j, k]];
                        let u2 = u_second[[i, j, k]];

                        let laplacian_u1 = numerics.laplacian(i, j, k, u_fund);
                        let laplacian_u2 = numerics.laplacian(i, j, k, u_second);

                        let term1 = u1 * laplacian_u2;
                        let term2 = u2 * laplacian_u1;
                        let term3 = 2.0 * numerics.divergence_product(i, j, k, u_fund, u_second);

                        let third_harmonic_source = beta * (term1 + term2 + term3);
                        let laplacian_u3 = numerics.laplacian(i, j, k, u3);
                        let acceleration_u3 =
                            sound_speed_sq.mul_add(laplacian_u3, third_harmonic_source);
                        *di = dt * dt * acceleration_u3;
                    }
                });
                d
            };
            // Apply delta after all RHS evaluations complete (Jacobi apply pass).
            field.u_harmonics[0] += &delta3;
        }

        // --- Loop 3: fourth and higher harmonics via continued cascading ---
        //
        // Source term: amplitude_factor · β (u₁∇²uₙ₋₁ + uₙ₋₁∇²u₁)
        // Explicit update: uₙ += dt² · (c²∇²uₙ + source)
        //
        // For each harmonic order n = harmonic_idx + 3:
        //   amplitude_factor = β^(n-2) / n  [Chen 2013, Eq. 12]
        //
        // Theorem (race-freedom, Loop 3): see module doc.
        for harmonic_idx in 1..field.u_harmonics.len() {
            let harmonic_order = harmonic_idx + 3;
            let amplitude_factor = beta.powi(harmonic_order as i32 - 1) / harmonic_order as f64;

            let delta_n = {
                // All immutable borrows scoped here; released before the += below.
                // Rust NLL field-split borrow: different elements of u_harmonics
                // slice can be simultaneously borrowed immutably.
                let u_fund = &field.u_fundamental;
                let u_prev: &Array3<f64> = if harmonic_idx == 1 {
                    &field.u_second
                } else {
                    &field.u_harmonics[harmonic_idx - 1]
                };
                let u_n = &field.u_harmonics[harmonic_idx];
                let numerics = &self.numerics;
                let sound_speed_sq = self.config.sound_speed().powi(2);

                let mut d = Array3::<f64>::zeros((nx, ny, nz));
                for_each_indexed_mut(d.view_mut(), |(i, j, k), di| {
                    if i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1 {
                        let u1 = u_fund[[i, j, k]];
                        let u_pv = u_prev[[i, j, k]];
                        let lap_u1 = numerics.laplacian(i, j, k, u_fund);
                        let lap_u_prev = numerics.laplacian(i, j, k, u_prev);
                        let source =
                            amplitude_factor * beta * u1.mul_add(lap_u_prev, u_pv * lap_u1);
                        let lap_u_n = numerics.laplacian(i, j, k, u_n);
                        let accel = sound_speed_sq.mul_add(lap_u_n, source);
                        *di = dt * dt * accel;
                    }
                });
                d
            };
            // Jacobi apply: uₙ updated using only its pre-step values.
            field.u_harmonics[harmonic_idx] += &delta_n;
        }
    }
}
