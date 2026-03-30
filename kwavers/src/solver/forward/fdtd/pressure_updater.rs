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
use ndarray::{Array3, ArrayView3, Zip};

use super::solver::{FdtdGpuAccelerator, FdtdSolver};

impl FdtdSolver {
    /// Update pressure field using velocity divergence.
    ///
    /// Dispatches to GPU accelerator when enabled, otherwise falls back to CPU.
    /// When `config.enable_nonlinear` is set, applies the Westervelt nonlinear
    /// correction after the linear update and rotates the pressure history.
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
        if self.kspace_ops.is_some() {
            {
                let kops = self.kspace_ops.as_mut().unwrap();
                kops.compute_divergence_neg(
                    &self.fields.ux,
                    &self.fields.uy,
                    &self.fields.uz,
                );
            }
            // Use the spectral divergence view directly — no .to_owned() allocation needed
            let divergence = self.kspace_ops.as_ref().unwrap().divergence.view();
            Self::update_pressure_simd(&mut self.fields.p, divergence, &self.rho_c_squared, dt);
            return Ok(());
        }

        let divergence = if self.config.staggered_grid {
            self.compute_divergence_staggered()?
        } else {
            let mut dvx = self.central_operator.apply_x(self.fields.ux.view())?;
            let mut dvy = self.central_operator.apply_y(self.fields.uy.view())?;
            let mut dvz = self.central_operator.apply_z(self.fields.uz.view())?;

            if let Some(ref mut cpml) = self.cpml_boundary {
                cpml.update_and_apply_v_gradient_correction(&mut dvx, 0);
                cpml.update_and_apply_v_gradient_correction(&mut dvy, 1);
                cpml.update_and_apply_v_gradient_correction(&mut dvz, 2);
            }
            &dvx + &dvy + &dvz
        };

        Self::update_pressure_simd(&mut self.fields.p, divergence.view(), &self.rho_c_squared, dt);
        Ok(())
    }

    /// Element-wise pressure update: p -= dt * ρc² * div(v).
    ///
    /// Accepts `ArrayView3<f64>` for `divergence` so that callers holding a view
    /// (e.g., the k-space spectral path) do not need `.to_owned()`, eliminating an
    /// O(N³) heap allocation per time step.  Owned arrays coerce to views via
    /// `arr.view()` at the call site.
    ///
    /// Uses plain iterator loop; LLVM will auto-vectorize on AVX2/AVX-512 targets.
    pub(crate) fn update_pressure_simd(
        pressure: &mut Array3<f64>,
        divergence: ArrayView3<f64>,
        rho_c_squared: &Array3<f64>,
        dt: f64,
    ) {
        for ((p, &div), &rc2) in pressure
            .iter_mut()
            .zip(divergence.iter())
            .zip(rho_c_squared.iter())
        {
            *p -= dt * rc2 * div;
        }
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
        let (Some(ref nl_coeff), Some(ref mut nl_scratch)) = (
            self.nl_coeff.as_ref(),
            self.nl_scratch.as_mut(),
        ) else {
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
    /// CPML gradient corrections are applied per-direction when enabled.
    pub(crate) fn compute_divergence_staggered(&mut self) -> KwaversResult<Array3<f64>> {
        let mut dvx = self
            .staggered_operator
            .apply_backward_x(self.fields.ux.view())?;
        let mut dvy = self
            .staggered_operator
            .apply_backward_y(self.fields.uy.view())?;
        let mut dvz = self
            .staggered_operator
            .apply_backward_z(self.fields.uz.view())?;

        if let Some(ref mut cpml) = self.cpml_boundary {
            cpml.update_and_apply_v_gradient_correction(&mut dvx, 0);
            cpml.update_and_apply_v_gradient_correction(&mut dvy, 1);
            cpml.update_and_apply_v_gradient_correction(&mut dvz, 2);
        }

        Ok(&dvx + &dvy + &dvz)
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
}
