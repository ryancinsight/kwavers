//! Matrix-free stiffness and acceleration kernels for `SemSolver`.
//!
//! ## Stiffness Application — Sum Factorisation (Komatitsch & Tromp 1999 §3)
//!
//! For each element e with local DOF array u_e[a,b,c]:
//!
//! ```text
//! Step 1 — reference-space gradients via D[p,a] = ld[[a,p]]:
//!   d_ξ[p,q,r] = Σ_a D[p,a] u_e[a,q,r]
//!   d_η[p,q,r] = Σ_b D[q,b] u_e[p,b,r]
//!   d_ζ[p,q,r] = Σ_c D[r,c] u_e[p,q,c]
//!
//! Step 2 — physical metric (G·d)[α] = Σ_m J⁻¹[α,m] h_m,  h_m = Σ_β J⁻¹[β,m] d_β
//!
//! Step 3 — weighted flux:
//!   q_ξ[p,q,r] = ρc² |J| w_p (G·d)[0]
//!   q_η[p,q,r] = ρc² |J| w_q (G·d)[1]
//!   q_ζ[p,q,r] = ρc² |J| w_r (G·d)[2]
//!
//! Step 4 — transpose derivative (scatter):
//!   (Ku)_{abc} = w_b w_c Σ_p ld[[a,p]] q_ξ[p,b,c]
//!              + w_a w_c Σ_q ld[[b,q]] q_η[a,q,c]
//!              + w_a w_b Σ_r ld[[c,r]] q_ζ[a,b,r]
//! ```

use super::sem_solver::SemSolver;
use crate::core::error::KwaversResult;
use ndarray::Array1;

impl SemSolver {
    /// Apply the stiffness operator K·u matrix-free using 3-D sum-factorisation.
    pub(super) fn apply_stiffness(&self, u: &Array1<f64>) -> Array1<f64> {
        let mut ku = Array1::<f64>::zeros(u.len());
        let n = self.mesh.basis.n_points();
        let ld = &self.mesh.basis.lagrange_derivatives;
        let w = &self.mesh.basis.gll_weights;
        let rho_c2 = self.config.density * self.config.sound_speed * self.config.sound_speed;

        for elem_idx in 0..self.mesh.elements.len() {
            let element = &self.mesh.elements[elem_idx];
            let eid = element.id;

            // Step 1: Gather u_e[a,b,c] from global DOFs
            let mut u_e = ndarray::Array3::<f64>::zeros((n, n, n));
            for a in 0..n {
                for b in 0..n {
                    for c in 0..n {
                        let g = self.element_local_to_global_dof(eid, a, b, c, n);
                        u_e[[a, b, c]] = u[g];
                    }
                }
            }

            // Step 2: Reference-space gradients
            let mut d_xi = ndarray::Array3::<f64>::zeros((n, n, n));
            let mut d_eta = ndarray::Array3::<f64>::zeros((n, n, n));
            let mut d_zeta = ndarray::Array3::<f64>::zeros((n, n, n));

            for p in 0..n {
                for q in 0..n {
                    for r in 0..n {
                        let mut dxi = 0.0;
                        let mut deta = 0.0;
                        let mut dzeta = 0.0;
                        for a in 0..n {
                            dxi += ld[[a, p]] * u_e[[a, q, r]];
                            deta += ld[[a, q]] * u_e[[p, a, r]];
                            dzeta += ld[[a, r]] * u_e[[p, q, a]];
                        }
                        d_xi[[p, q, r]] = dxi;
                        d_eta[[p, q, r]] = deta;
                        d_zeta[[p, q, r]] = dzeta;
                    }
                }
            }

            // Steps 3–4: Metric tensor and weighted fluxes
            let mut q_xi = ndarray::Array3::<f64>::zeros((n, n, n));
            let mut q_eta = ndarray::Array3::<f64>::zeros((n, n, n));
            let mut q_zeta = ndarray::Array3::<f64>::zeros((n, n, n));

            for p in 0..n {
                for q in 0..n {
                    for r in 0..n {
                        let jdet = element.jacobian_det[[p, q, r]];
                        let jinv = element.jacobian_inv.slice(ndarray::s![p, q, r, .., ..]);

                        let d = [d_xi[[p, q, r]], d_eta[[p, q, r]], d_zeta[[p, q, r]]];

                        let mut h = [0.0f64; 3];
                        for beta in 0..3usize {
                            for m in 0..3usize {
                                h[m] += jinv[[beta, m]] * d[beta];
                            }
                        }
                        let mut gd = [0.0f64; 3];
                        for alpha in 0..3usize {
                            for m in 0..3usize {
                                gd[alpha] += jinv[[alpha, m]] * h[m];
                            }
                        }

                        let scale = rho_c2 * jdet;
                        q_xi[[p, q, r]] = scale * w[p] * gd[0];
                        q_eta[[p, q, r]] = scale * w[q] * gd[1];
                        q_zeta[[p, q, r]] = scale * w[r] * gd[2];
                    }
                }
            }

            // Step 5: Transpose derivative — scatter weighted fluxes to DOFs
            for a in 0..n {
                for b in 0..n {
                    for c in 0..n {
                        let mut xi_sum = 0.0;
                        for p in 0..n {
                            xi_sum += ld[[a, p]] * q_xi[[p, b, c]];
                        }
                        let mut eta_sum = 0.0;
                        for q in 0..n {
                            eta_sum += ld[[b, q]] * q_eta[[a, q, c]];
                        }
                        let mut zeta_sum = 0.0;
                        for r in 0..n {
                            zeta_sum += ld[[c, r]] * q_zeta[[a, b, r]];
                        }

                        let val =
                            w[b] * w[c] * xi_sum + w[a] * w[c] * eta_sum + w[a] * w[b] * zeta_sum;

                        let g = self.element_local_to_global_dof(eid, a, b, c, n);
                        ku[g] += val;
                    }
                }
            }
        }

        ku
    }

    /// Compute nodal acceleration ü_i = −(K·u)_i / M_ii
    pub(super) fn compute_acceleration(&self) -> KwaversResult<Array1<f64>> {
        let ku = self.apply_stiffness(&self.solution);
        let n_dofs = self.solution.len();
        let mut acceleration = Array1::<f64>::zeros(n_dofs);

        for i in 0..n_dofs {
            let m = self.mass_matrix[i];
            if !m.is_finite() || m <= 0.0 {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "Non-positive mass matrix entry at dof {i}: {m}"
                )));
            }
            acceleration[i] = -ku[i] / m;
        }

        Ok(acceleration)
    }
}
