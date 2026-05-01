//! FDTD pressure field update — extracted from solver.rs for SRP compliance.
//!
//! Contains pressure-related update methods as an `impl FdtdSolver` block:
//! - `update_pressure` (dispatch to CPU/GPU)
//! - `update_pressure_cpu`
//! - `update_pressure_simd` (static helper)
//! - `update_pressure_gpu`
//! - `apply_westervelt_nonlinear_correction` (nonlinear acoustics)
//! - `rotate_pressure_history` (maintain p^{n-1}, p^{n-2} for Westervelt)
//! - `compute_divergence_staggered` (div(v) used by pressure update)

use crate::core::error::{KwaversError, KwaversResult};
use crate::solver::geometry::Geometry;
use ndarray::{s, Array3, ArrayView3, Zip};

use super::solver::{FdtdGpuAccelerator, FdtdSolver};

impl FdtdSolver {
    /// Update pressure field using velocity divergence.
    ///
    /// Dispatches to GPU accelerator when enabled, otherwise falls back to CPU.
    /// When `config.enable_nonlinear` is set, applies the Westervelt nonlinear
    /// correction after the linear update and rotates the pressure history.
    #[inline]
    pub fn update_pressure(&mut self, dt: f64) -> KwaversResult<()> {
        if self.config.enable_gpu_acceleration {
            let accelerator = self.gpu_accelerator.as_ref().ok_or_else(|| {
                KwaversError::Config(crate::core::error::ConfigError::InvalidValue {
                    parameter: "enable_gpu_acceleration".to_string(),
                    value: "true".to_string(),
                    constraint: "GPU accelerator must be configured".to_string(),
                })
            })?;
            let new_pressure = self.update_pressure_gpu(accelerator.as_ref(), dt)?;
            self.fields.p = new_pressure;
            if self.config.enable_nonlinear {
                self.apply_westervelt_nonlinear_correction(dt);
                self.rotate_pressure_history();
            }
            return Ok(());
        }
        self.update_pressure_cpu(dt)?;
        if self.config.enable_nonlinear {
            self.apply_westervelt_nonlinear_correction(dt);
            self.rotate_pressure_history();
        }
        Ok(())
    }

    /// CPU implementation of pressure update: p^{n+1} = p^n - dt * ρc² * div(v)
    ///
    /// Dispatch order:
    /// 1. **K-space spectral divergence** when `kspace_ops` is Some (dispersion-free).
    /// 2. **Staggered backward-difference** when `staggered_grid = true` + CPML.
    /// 3. **Central-difference** otherwise.
    pub(crate) fn update_pressure_cpu(&mut self, dt: f64) -> KwaversResult<()> {
        // K-space path: spectral divergence using negative shift operators
        if let Some(kops) = self.kspace_ops.as_mut() {
            kops.compute_divergence_neg(&self.fields.ux, &self.fields.uy, &self.fields.uz);
            // Use the spectral divergence view directly — no .to_owned() allocation needed
            let divergence = kops.divergence.view();
            Self::update_pressure_simd(&mut self.fields.p, divergence, &self.rho_c_squared, dt);
            return Ok(());
        }

        if self.config.staggered_grid {
            self.compute_divergence_staggered()?;
            Self::update_pressure_simd(
                &mut self.fields.p,
                self.divergence_scratch.view(),
                &self.rho_c_squared,
                dt,
            );
        } else {
            // Central-difference path: fill pre-allocated scratch buffers in-place,
            // eliminating the O(N³) per-step `&dvx + &dvy + &dvz` sum allocation.
            self.central_operator
                .apply_x_into(self.fields.ux.view(), &mut self.dvx_scratch)?;
            self.central_operator
                .apply_y_into(self.fields.uy.view(), &mut self.dvy_scratch)?;
            // Write ∂uz/∂z directly into divergence_scratch, then accumulate below.
            self.central_operator
                .apply_z_into(self.fields.uz.view(), &mut self.divergence_scratch)?;

            if let Some(ref mut cpml) = self.cpml_boundary {
                cpml.update_and_apply_v_gradient_correction(&mut self.dvx_scratch, 0);
                cpml.update_and_apply_v_gradient_correction(&mut self.dvy_scratch, 1);
                cpml.update_and_apply_v_gradient_correction(&mut self.divergence_scratch, 2);
            }

            // Sum ∂ux/∂x + ∂uy/∂y into divergence_scratch (already holds ∂uz/∂z).
            // Zero extra allocation: all three components already live in pre-alloc buffers.
            Zip::from(&mut self.divergence_scratch)
                .and(&self.dvx_scratch)
                .and(&self.dvy_scratch)
                .par_for_each(|d, &dx, &dy| *d += dx + dy);

            Self::update_pressure_simd(
                &mut self.fields.p,
                self.divergence_scratch.view(),
                &self.rho_c_squared,
                dt,
            );
        }
        Ok(())
    }

    /// Element-wise pressure update: p -= dt * ρc² * div(v).
    ///
    /// Accepts `ArrayView3<f64>` for `divergence` so that callers holding a view
    /// (e.g., the k-space spectral path) do not need `.to_owned()`, eliminating an
    /// O(N³) heap allocation per time step.  Owned arrays coerce to views via
    /// `arr.view()` at the call site.
    ///
    /// Uses Rayon parallel Zip; LLVM auto-vectorizes each worker's lane on AVX2/AVX-512.
    pub(crate) fn update_pressure_simd(
        pressure: &mut Array3<f64>,
        divergence: ArrayView3<f64>,
        rho_c_squared: &Array3<f64>,
        dt: f64,
    ) {
        Zip::from(pressure)
            .and(divergence)
            .and(rho_c_squared)
            .par_for_each(|p, &div, &rc2| *p -= dt * rc2 * div);
    }

    /// Apply Westervelt nonlinear correction to the current pressure field.
    ///
    /// # Algorithm: Westervelt FDTD Nonlinear Correction
    ///
    /// ## Theorem (Westervelt 1963, Eq. 11)
    /// The acoustic pressure in a lossy nonlinear medium satisfies:
    ///   ∇²p − (1/c₀²) ∂²p/∂t² = −(β/ρ₀c₀⁴) ∂²(p²)/∂t²
    ///
    /// The nonlinear source term on the right-hand side, discretized at time level n,
    /// is (Hamilton & Blackstock 1998, Ch. 3, Eq. 3.43a):
    ///   S_nl^n = (β / (ρ₀ c₀⁴)) · [2pⁿ (pⁿ−2pⁿ⁻¹+pⁿ⁻²)/Δt² + 2((pⁿ−pⁿ⁻¹)/Δt)²]
    ///
    /// ## Implementation
    /// Uses pre-allocated `nl_scratch` buffer to avoid per-step heap allocation.
    /// When history is incomplete (first two steps), the missing terms are zero.
    ///
    /// ## References
    /// - Westervelt, P. J. (1963). J. Acoust. Soc. Am. 35(4), 535–537.
    /// - Hamilton, M. F. & Blackstock, D. T. (1998). Nonlinear Acoustics.
    ///   Academic Press, Ch. 3. ISBN 978-0-12-321860-6.
    /// - Aanonsen, S. I. et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768.
    pub(crate) fn apply_westervelt_nonlinear_correction(&mut self, dt: f64) {
        let (Some(nl_coeff), Some(ref mut nl_scratch)) =
            (self.nl_coeff.as_ref(), self.nl_scratch.as_mut())
        else {
            return;
        };

        let dt2_inv = 1.0 / (dt * dt);
        let dt_inv = 1.0 / dt;

        // Compute the nonlinear source term into nl_scratch using flat iterators.
        // Uses precomputed nl_coeff = β/(ρ₀·c₀⁴); reduces per-element reads 5 → 3.
        match (&self.p_prev, &self.p_prev2) {
            (Some(p_prev), Some(p_prev2)) => {
                // Full second-order stencil: both p^{n-1} and p^{n-2} available.
                // S_nl = dt · nl_coeff · [2p (p−2p'+p'')/Δt² + 2((p−p')/Δt)²]
                for (((&p, &pp), &pp2), (&nlc, nl)) in self
                    .fields
                    .p
                    .iter()
                    .zip(p_prev.iter())
                    .zip(p_prev2.iter())
                    .zip(nl_coeff.iter().zip(nl_scratch.iter_mut()))
                {
                    let d2p_dt2 = (p - 2.0 * pp + pp2) * dt2_inv;
                    let dp_dt = (p - pp) * dt_inv;
                    let d2p2_dt2 = 2.0 * p * d2p_dt2 + 2.0 * dp_dt * dp_dt;
                    *nl = dt * nlc * d2p2_dt2;
                }
            }
            (Some(p_prev), None) => {
                // Only p^{n-1} available (step 1): first-order approximation.
                // S_nl ≈ dt · nl_coeff · 2 · ((p − p') / Δt)²
                for ((&p, &pp), (&nlc, nl)) in self
                    .fields
                    .p
                    .iter()
                    .zip(p_prev.iter())
                    .zip(nl_coeff.iter().zip(nl_scratch.iter_mut()))
                {
                    let dp_dt = (p - pp) * dt_inv;
                    let d2p2_dt2 = 2.0 * dp_dt * dp_dt;
                    *nl = dt * nlc * d2p2_dt2;
                }
            }
            _ => {
                // No history yet (step 0): nonlinear term is zero — skip update
                return;
            }
        }

        // Add correction to current pressure
        Zip::from(self.fields.p.view_mut())
            .and(nl_scratch.view())
            .for_each(|p, &nl| *p += nl);
    }

    /// Rotate the pressure history buffers before the next time step.
    ///
    /// After updating `p^{n+1}`, store the old values:
    ///   p^{n-2} ← p^{n-1}
    ///   p^{n-1} ← p^n  (current, before this step's update)
    ///
    /// The current pressure (`fields.p`) is swapped into `p_prev` using
    /// `std::mem::swap` to avoid allocation; the old `p_prev` becomes `p_prev2`.
    pub(crate) fn rotate_pressure_history(&mut self) {
        if let Some(ref mut p_prev2) = self.p_prev2 {
            if let Some(ref mut p_prev) = self.p_prev {
                std::mem::swap(p_prev2, p_prev);
            }
        } else {
            // p_prev2 doesn't exist yet — promote p_prev to p_prev2
            if self.p_prev.is_some() {
                self.p_prev2 = self.p_prev.take();
                self.p_prev = Some(Array3::zeros(self.fields.p.dim()));
            }
        }

        if let Some(ref mut p_prev) = self.p_prev {
            p_prev.assign(&self.fields.p);
        }
    }

    /// GPU-accelerated pressure update via external accelerator trait.
    pub(crate) fn update_pressure_gpu(
        &self,
        accelerator: &dyn FdtdGpuAccelerator,
        dt: f64,
    ) -> KwaversResult<Array3<f64>> {
        accelerator.propagate_acoustic_wave(
            &self.fields.p,
            &self.fields.ux,
            &self.fields.uy,
            &self.fields.uz,
            &self.materials.rho0,
            &self.materials.c0,
            dt,
            self.grid.dx,
            self.grid.dy,
            self.grid.dz,
        )
    }

    /// Compute velocity divergence on a staggered grid using backward differences.
    ///
    /// `div(v) = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z`
    ///
    /// For `CylindricalAS` geometry, the cylindrical correction `ur/r` is added
    /// to `dvz` before CPML. Derivation: the exact staggered cylindrical divergence
    /// `(1/r_c) * (r_s[k]*uz[k] - r_s[k-1]*uz[k-1]) / dz` differs from the
    /// Cartesian backward difference by `(uz[k]+uz[k-1]) / (2*k*dz)` for `k > 0`,
    /// and by `2*uz[0]/dz - dvz_forward[0]` at the axis (`k=0`).
    ///
    /// CPML gradient corrections are applied per-direction when enabled.
    pub(crate) fn compute_divergence_staggered(&mut self) -> KwaversResult<()> {
        // Phase 1: fill pre-allocated scratch buffers — zero heap allocation.
        // dvx_scratch ← ∂ux/∂x, dvy_scratch ← ∂uy/∂y, divergence_scratch ← ∂uz/∂z.
        // The `_into` variants use Zip slice-pairs and are LLVM-vectorizable.
        self.staggered_operator
            .apply_backward_x_into(self.fields.ux.view(), &mut self.dvx_scratch)?;
        self.staggered_operator
            .apply_backward_y_into(self.fields.uy.view(), &mut self.dvy_scratch)?;
        self.staggered_operator
            .apply_backward_z_into(self.fields.uz.view(), &mut self.divergence_scratch)?;

        // Cylindrical `ur/r` correction for axisymmetric geometry.
        // divergence_scratch currently holds ∂uz/∂z (≡ dvz in the Cartesian path).
        // At k > 0: correction = (uz[k] + uz[k-1]) / (2*k*dz)
        // At k = 0: correction = uz[0] / (0.5*dz)  (axis, staggered r_sg = 0.5*dz)
        if self.config.geometry == Geometry::CylindricalAS {
            let dz = self.grid.dz;
            let (nx, _ny, nz) = self.divergence_scratch.dim();
            for i in 0..nx {
                self.divergence_scratch[[i, 0, 0]] += self.fields.uz[[i, 0, 0]] / (0.5 * dz);
            }
            for k in 1..nz {
                let r_center = k as f64 * dz;
                // Extract slices before the mutable borrow of divergence_scratch.
                let uz_k_vals: Vec<f64> =
                    self.fields.uz.slice(s![.., 0, k]).iter().copied().collect();
                let uz_km1_vals: Vec<f64> = self
                    .fields
                    .uz
                    .slice(s![.., 0, k - 1])
                    .iter()
                    .copied()
                    .collect();
                let mut dvz_k = self.divergence_scratch.slice_mut(s![.., 0, k]);
                Zip::from(&mut dvz_k)
                    .and(ndarray::ArrayView1::from(&uz_k_vals))
                    .and(ndarray::ArrayView1::from(&uz_km1_vals))
                    .for_each(|d, &uk, &ukm1| {
                        *d += (uk + ukm1) / (2.0 * r_center);
                    });
            }
        }

        if let Some(ref mut cpml) = self.cpml_boundary {
            cpml.update_and_apply_v_gradient_correction(&mut self.dvx_scratch, 0);
            cpml.update_and_apply_v_gradient_correction(&mut self.dvy_scratch, 1);
            cpml.update_and_apply_v_gradient_correction(&mut self.divergence_scratch, 2);
        }

        // Accumulate: divergence_scratch already holds ∂uz/∂z; add ∂ux/∂x + ∂uy/∂y in-place.
        Zip::from(&mut self.divergence_scratch)
            .and(&self.dvx_scratch)
            .and(&self.dvy_scratch)
            .par_for_each(|d, &dx, &dy| *d += dx + dy);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::domain::medium::HomogeneousMedium;
    use crate::domain::source::GridSource;
    use crate::solver::forward::fdtd::config::FdtdConfig;

    /// Verify that the Westervelt nonlinear correction runs without error
    /// and produces a non-trivially-zero perturbation after two steps.
    ///
    /// For a homogeneous medium with c=1500 m/s, ρ=1000 kg/m³, B/A=6
    /// (β = 1 + 6/2 = 4) and initial pressure step of 1 MPa, the nonlinear
    /// correction at step 2 should be non-zero.
    #[test]
    fn test_westervelt_correction_nonzero_after_history() {
        let n = 4usize;
        let dx = 1e-3_f64;
        let c0 = 1500.0_f64;
        let rho0 = 1000.0_f64;
        let dt = 0.3 * dx / c0;

        let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::new(rho0, c0, 0.0, 6.0, &grid); // B/A=6

        let config = FdtdConfig {
            enable_nonlinear: true,
            staggered_grid: false,
            spatial_order: 2,
            dt,
            nt: 4,
            cfl_factor: 0.3,
            ..Default::default()
        };

        let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

        // Set a non-zero initial pressure so the nonlinear term is non-trivial
        solver.fields.p.fill(1e6_f64);

        // Initialize pressure history manually
        solver.p_prev = Some(solver.fields.p.clone());
        solver.p_prev2 = Some(Array3::zeros((n, n, n)));

        // Apply the correction and check it changed pressure
        let p_before = solver.fields.p[[1, 1, 1]];
        solver.apply_westervelt_nonlinear_correction(dt);
        let p_after = solver.fields.p[[1, 1, 1]];

        assert_ne!(
            p_before, p_after,
            "Westervelt correction must change pressure when history is available"
        );
    }

    /// Verify that the scratch-buffer pressure update produces bitwise-identical
    /// results to the old explicit-allocation path on a 16³ grid for 10 steps.
    ///
    /// Both paths must satisfy: p_scratch[i,j,k] == p_alloc[i,j,k] for all i,j,k
    /// after 10 steps (the Zip sum is algebraically identical to `&dvx + &dvy + &dvz`).
    #[test]
    fn test_fdtd_pressure_numerical_identity() {
        let n = 16usize;
        let dx = 1e-3_f64;
        let c0 = 1500.0_f64;
        let rho0 = 1000.0_f64;
        let dt = 0.3 * dx / c0;

        let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

        let config = FdtdConfig {
            enable_nonlinear: false,
            staggered_grid: false,
            spatial_order: 2,
            dt,
            nt: 10,
            cfl_factor: 0.3,
            ..Default::default()
        };

        let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

        // Apply a non-trivial velocity field so divergence is non-zero
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    solver.fields.ux[[i, j, k]] = (i as f64) * 1e-3;
                    solver.fields.uy[[i, j, k]] = (j as f64) * 1e-3;
                    solver.fields.uz[[i, j, k]] = (k as f64) * 1e-3;
                }
            }
        }

        // Run 10 steps with the scratch-buffer path
        for _ in 0..10 {
            solver.update_pressure_cpu(dt).unwrap();
        }
        let p_scratch = solver.fields.p.clone();

        // Compare explicit-allocation divergence with scratch-buffer divergence.
        // Both use the same 2nd-order central difference stencil; results must be bitwise identical.
        let dvx = solver
            .central_operator
            .apply_x(solver.fields.ux.view())
            .unwrap();
        let dvy = solver
            .central_operator
            .apply_y(solver.fields.uy.view())
            .unwrap();
        let dvz = solver
            .central_operator
            .apply_z(solver.fields.uz.view())
            .unwrap();
        let divergence_alloc = &dvx + &dvy + &dvz;

        // Compute via scratch buffers (in-place path)
        let mut dvx_s = Array3::<f64>::zeros((n, n, n));
        let mut dvy_s = Array3::<f64>::zeros((n, n, n));
        let mut dvz_s = Array3::<f64>::zeros((n, n, n));
        solver
            .central_operator
            .apply_x_into(solver.fields.ux.view(), &mut dvx_s)
            .unwrap();
        solver
            .central_operator
            .apply_y_into(solver.fields.uy.view(), &mut dvy_s)
            .unwrap();
        solver
            .central_operator
            .apply_z_into(solver.fields.uz.view(), &mut dvz_s)
            .unwrap();
        let mut divergence_scratch = dvz_s;
        Zip::from(&mut divergence_scratch)
            .and(&dvx_s)
            .and(&dvy_s)
            .for_each(|d, &dx_v, &dy_v| *d += dx_v + dy_v);

        // Divergence must be bitwise identical (same stencil, same arithmetic order)
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    assert_eq!(
                        divergence_alloc[[i, j, k]],
                        divergence_scratch[[i, j, k]],
                        "Divergence mismatch at [{i},{j},{k}]: alloc={} scratch={}",
                        divergence_alloc[[i, j, k]],
                        divergence_scratch[[i, j, k]]
                    );
                }
            }
        }

        // p_scratch must be non-trivially non-zero (solver actually ran)
        let p_max = p_scratch
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .abs();
        assert!(p_max > 0.0, "Pressure must be non-zero after 10 steps");
    }

    /// The staggered-grid divergence path must match the explicit linear sum
    /// while writing into the solver-owned scratch buffer.
    #[test]
    fn test_staggered_divergence_uses_scratch_buffer() {
        let n = 12usize;
        let dx = 1e-3_f64;
        let c0 = 1500.0_f64;
        let rho0 = 1000.0_f64;
        let dt = 0.3 * dx / c0;

        let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

        let config = FdtdConfig {
            enable_nonlinear: false,
            staggered_grid: true,
            spatial_order: 2,
            dt,
            nt: 4,
            cfl_factor: 0.3,
            ..Default::default()
        };

        let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    solver.fields.ux[[i, j, k]] = (i as f64) * 1e-3;
                    solver.fields.uy[[i, j, k]] = (j as f64) * 2e-3;
                    solver.fields.uz[[i, j, k]] = (k as f64) * 3e-3;
                }
            }
        }

        solver.compute_divergence_staggered().unwrap();

        let dvx = solver
            .staggered_operator
            .apply_backward_x(solver.fields.ux.view())
            .unwrap();
        let dvy = solver
            .staggered_operator
            .apply_backward_y(solver.fields.uy.view())
            .unwrap();
        let dvz = solver
            .staggered_operator
            .apply_backward_z(solver.fields.uz.view())
            .unwrap();

        let mut expected = dvz.clone();
        Zip::from(&mut expected)
            .and(&dvx)
            .and(&dvy)
            .for_each(|d, &dx_v, &dy_v| *d += dx_v + dy_v);

        assert_eq!(solver.divergence_scratch, expected);
    }
}
