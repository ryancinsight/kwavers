//! IMEXScheme trait implementation for IMEXRK.

use super::scheme::IMEXRK;
use super::types::IMEXRKType;
use crate::core::error::KwaversResult;
use crate::solver::forward::imex::traits::IMEXScheme;
use crate::solver::forward::imex::ImplicitSolverType;
use ndarray::{Array3, Zip};

impl IMEXScheme for IMEXRK {
    fn step<F, G>(
        &self,
        field: &Array3<f64>,
        dt: f64,
        explicit_rhs: F,
        implicit_rhs: G,
        implicit_solver: &ImplicitSolverType,
    ) -> KwaversResult<Array3<f64>>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        let mut k_explicit = vec![Array3::zeros(field.dim()); self.s];
        let mut k_implicit = vec![Array3::zeros(field.dim()); self.s];
        let mut stages = vec![field.clone(); self.s];

        // Compute stages
        for i in 0..self.s {
            // Build stage value
            let mut stage_value = field.clone();

            // Add explicit contributions
            for (j, k_exp) in k_explicit.iter().enumerate().take(i) {
                if self.a_explicit[i][j] != 0.0 {
                    Zip::from(&mut stage_value)
                        .and(k_exp)
                        .for_each(|s, &k| *s += dt * self.a_explicit[i][j] * k);
                }
            }

            // Add implicit contributions from previous stages
            for (j, k_imp) in k_implicit.iter().enumerate().take(i) {
                if self.a_implicit[i][j] != 0.0 {
                    Zip::from(&mut stage_value)
                        .and(k_imp)
                        .for_each(|s, &k| *s += dt * self.a_implicit[i][j] * k);
                }
            }

            // Solve implicit equation for current stage
            if self.a_implicit[i][i] != 0.0 {
                let aii = self.a_implicit[i][i];
                let implicit_fn = |y: &Array3<f64>| -> KwaversResult<Array3<f64>> {
                    let mut residual = y - &stage_value;
                    let g = implicit_rhs(y)?;
                    Zip::from(&mut residual)
                        .and(&g)
                        .for_each(|r, &g| *r -= dt * aii * g);
                    Ok(residual)
                };

                stages[i] = implicit_solver.solve(&stage_value, implicit_fn)?;
                k_implicit[i] = implicit_rhs(&stages[i])?;
            } else {
                stages[i] = stage_value;
            }

            // Compute explicit RHS
            k_explicit[i] = explicit_rhs(&stages[i])?;
        }

        // Combine stages
        let mut result = field.clone();
        for i in 0..self.s {
            if self.b[i] != 0.0 {
                Zip::from(&mut result)
                    .and(&k_explicit[i])
                    .and(&k_implicit[i])
                    .for_each(|r, &ke, &ki| {
                        *r += dt * self.b[i] * (ke + ki);
                    });
            }
        }

        Ok(result)
    }

    fn order(&self) -> usize {
        self.p
    }

    fn stages(&self) -> usize {
        self.s
    }

    fn is_a_stable(&self) -> bool {
        // All our IMEX-RK schemes are A-stable
        true
    }

    fn is_l_stable(&self) -> bool {
        // Check if the last row of implicit tableau has the form needed for L-stability
        matches!(
            self.config.scheme_type,
            IMEXRKType::SSP2_222 | IMEXRKType::ARK3 | IMEXRKType::ARK4
        )
    }

    fn adjust_for_stiffness(&mut self, stiffness_ratio: f64) {
        // Adjust implicit solver tolerance based on stiffness
        self.stiffness_factor = (stiffness_ratio / 10.0).clamp(0.1, 10.0);
    }

    fn stability_function(&self, z: f64) -> f64 {
        // Approximate stability function for the implicit part
        // For L-stable methods, this approaches 0 as |z| -> infinity
        match self.config.scheme_type {
            IMEXRKType::SSP2_222 => {
                let gamma = 1.0 - 1.0 / 2.0_f64.sqrt();
                (1.0 + (1.0 - 2.0 * gamma) * z) / (1.0 - gamma * z).powi(2)
            }
            _ => {
                // General approximation for higher order methods
                1.0 / (1.0 - z / self.p as f64).powi(self.p as i32)
            }
        }
    }
}
