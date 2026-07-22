//! Adjoint run and the `K_μ` shear-strain cross-correlation imaging condition
//! (ADR 033 increment 2).

use kwavers_core::error::{KwaversResult, NumericalError};
use leto::Array3;

use super::{l2_misfit, sample_receivers, ElasticFwi, ReceiverTraces};
use crate::forward::elastic::swe::{ElasticPointForce, ElasticWaveField, ElasticWaveSolver};

impl ElasticFwi {
    /// One forward + one adjoint elastic simulation at `mu`, returning the data
    /// misfit, the **raw** sensitivity kernel `K_μ`, and the forward-illumination
    /// (pseudo-Hessian diagonal) map. No muting/preconditioning/regularization.
    fn forward_adjoint(
        &mut self,
        mu: &Array3<f64>,
    ) -> KwaversResult<(f64, Array3<f64>, Array3<f64>)> {
        let n = self.config.n_steps;
        let dt = self.config.dt;

        self.solver.set_mu(mu)?;
        let fwd = self
            .solver
            .propagate_point_forces(n, dt, &self.config.source)?;
        let syn = sample_receivers(&fwd, &self.config.receivers);
        let j = l2_misfit(&syn, &self.observed, dt);

        let residual = residual(&syn, &self.observed);
        let adj_forces = build_adjoint_forces(&residual, &self.config.receivers, n);
        let (grad, illum) =
            stream_k_mu_kernel(&self.solver, &fwd, &adj_forces, dt, self.grid_spacing)?;
        Ok((j, grad, illum))
    }

    /// Data misfit and its **raw** gradient `∂J/∂μ = K_μ` at `mu` (muted at the
    /// acquisition points; no preconditioning/regularization). This is the
    /// physically meaningful gradient validated by the finite-difference check.
    ///
    /// # Errors
    /// Propagates solver errors; errors if `mu` does not match the grid shape.
    pub(super) fn data_misfit_and_gradient(
        &mut self,
        mu: &Array3<f64>,
    ) -> KwaversResult<(f64, Array3<f64>)> {
        let (j, mut grad, _illum) = self.forward_adjoint(mu)?;
        self.mute_acquisition_imprint(&mut grad);
        Ok((j, grad))
    }

    /// Data misfit and the **illumination-preconditioned**, muted gradient that
    /// drives the inversion: `g̃ = K_μ / (W + ε·max W)`, with `W` the forward
    /// strain-energy (pseudo-Hessian diagonal; Shin et al. 2001). This balances
    /// the near-acquisition region against the weakly-illuminated interior so
    /// steepest descent updates the whole field, not just near the transducers.
    ///
    /// # Errors
    /// Propagates solver errors.
    pub(super) fn data_misfit_and_preconditioned_gradient(
        &mut self,
        mu: &Array3<f64>,
    ) -> KwaversResult<(f64, Array3<f64>)> {
        let (j, mut grad, illum) = self.forward_adjoint(mu)?;
        let wmax = illum.iter().fold(0.0_f64, |m, &v| m.max(v));
        if wmax > 0.0 {
            let floor = self.config.precond_eps * wmax;
            leto_ops::zip_mut_with(&mut grad.view_mut(), &illum.view(), |g, w| {
                *g /= *w + floor;
            })
            .expect("invariant: gradient and illumination field shapes asserted equal");
        }
        self.mute_acquisition_imprint(&mut grad);
        Ok((j, grad))
    }

    /// Zero the gradient within `config.mute_radius` cells of every source and
    /// receiver. Strain — hence `∂J/∂μ` — is near-singular at the point sources
    /// and receivers; without muting, those cells dominate the max-norm-
    /// normalized step and starve the interior reconstruction.
    fn mute_acquisition_imprint(&self, grad: &mut Array3<f64>) {
        let r = self.config.mute_radius;
        if r == 0 {
            return;
        }
        let [nx, ny, nz] = grad.shape();
        let r2 = (r * r) as i64;
        let centers = self
            .config
            .receivers
            .iter()
            .copied()
            .chain(self.config.source.iter().map(|f| f.index));
        for (ci, cj, ck) in centers {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let di = i as i64 - ci as i64;
                        let dj = j as i64 - cj as i64;
                        let dk = k as i64 - ck as i64;
                        if di * di + dj * dj + dk * dk <= r2 {
                            grad[[i, j, k]] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

/// Per-receiver, per-step, per-component displacement residual `d_syn − d_obs`.
fn residual(syn: &ReceiverTraces, obs: &ReceiverTraces) -> ReceiverTraces {
    syn.iter()
        .zip(obs.iter())
        .map(|(sr, or)| {
            sr.iter()
                .zip(or.iter())
                .map(|(sn, on)| [sn[0] - on[0], sn[1] - on[1], sn[2] - on[2]])
                .collect()
        })
        .collect()
}

/// Build the adjoint source: each receiver becomes a point force whose
/// per-step components are the **time-reversed** residual. Running this source
/// forward through the (self-adjoint) elastic operator produces the adjoint
/// field in reverse time order; [`stream_k_mu_kernel`] re-aligns it.
fn build_adjoint_forces(
    residual: &ReceiverTraces,
    receivers: &[(usize, usize, usize)],
    n_steps: usize,
) -> Vec<ElasticPointForce> {
    receivers
        .iter()
        .zip(residual.iter())
        .map(|(&index, r)| {
            let mut force = ElasticPointForce::zeros(index, n_steps);
            for m in 0..n_steps {
                let rev = n_steps - 1 - m;
                force.fx[m] = r[rev][0];
                force.fy[m] = r[rev][1];
                force.fz[m] = r[rev][2];
            }
            force
        })
        .collect()
}

/// Stream the adjoint propagation into the shear-modulus sensitivity kernel.
///
/// The lossy forward solve remains stored because it cannot be reconstructed
/// backward, but each adjoint state is consumed immediately. This retains one
/// six-component adjoint field instead of `n_steps` cloned fields.
fn stream_k_mu_kernel(
    solver: &ElasticWaveSolver,
    fwd: &[ElasticWaveField],
    adj_forces: &[ElasticPointForce],
    dt: f64,
    grid_spacing: (f64, f64, f64),
) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
    let first = fwd.first().ok_or_else(|| {
        NumericalError::InvalidOperation(
            "elastic FWI gradient requires at least one time step".to_owned(),
        )
    })?;
    let dim = first.ux.shape();
    let mut grad = Array3::<f64>::zeros(dim);
    let mut illum = Array3::<f64>::zeros(dim);
    let n_steps = fwd.len();

    solver.propagate_point_forces_observing(n_steps, dt, adj_forces, |adjoint_step, adjoint| {
        let forward = &fwd[n_steps - 1 - adjoint_step];
        accumulate_k_mu_step(&mut grad, &mut illum, forward, adjoint, dt, grid_spacing);
    })?;
    Ok((grad, illum))
}

/// Accumulate one reverse-time-aligned shear-strain cross-correlation step.
///
/// Shear-modulus sensitivity kernel (Tromp, Tape & Liu 2005; Köhn 2011):
///
/// `K_μ(x) = −∫₀ᵀ Σ_ij (∂_i u_j + ∂_j u_i)_fwd (∂_i u_j + ∂_j u_i)_adj dt`,
///
/// the full 3-D strain cross-correlation
///
/// `S = 4(ε_xx^f ε_xx^a + ε_yy^f ε_yy^a + ε_zz^f ε_zz^a)
///    + 2(γ_xy^f γ_xy^a + γ_xz^f γ_xz^a + γ_yz^f γ_yz^a)`
///
/// with `ε_ii = ∂_i u_i` and `γ_ij = ∂_i u_j + ∂_j u_i` (`f` = forward, `a` =
/// adjoint). For 2-D plane strain (`nz = 1`) the out-of-plane displacement and
/// every `∂_z` vanish, so the `zz`, `xz`, `yz` terms drop and `S` reduces to the
/// in-plane form — the 3-D kernel subsumes the 2-D case exactly.
///
/// Spatial derivatives use second-order central differences (zero at the grid
/// boundary, inside the PML). `illum` accumulates the forward strain-energy
/// (pseudo-Hessian diagonal): the same auto-correlation with `f` in both slots.
fn accumulate_k_mu_step(
    grad: &mut Array3<f64>,
    illum: &mut Array3<f64>,
    uf: &ElasticWaveField,
    ua: &ElasticWaveField,
    dt: f64,
    (dx, dy, dz): (f64, f64, f64),
) {
    let [nx, ny, nz] = grad.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // Forward strains.
                let exx_f = ddx(&uf.ux, i, j, k, nx, dx);
                let eyy_f = ddy(&uf.uy, i, j, k, ny, dy);
                let ezz_f = ddz(&uf.uz, i, j, k, nz, dz);
                let gxy_f = ddx(&uf.uy, i, j, k, nx, dx) + ddy(&uf.ux, i, j, k, ny, dy);
                let gxz_f = ddx(&uf.uz, i, j, k, nx, dx) + ddz(&uf.ux, i, j, k, nz, dz);
                let gyz_f = ddy(&uf.uz, i, j, k, ny, dy) + ddz(&uf.uy, i, j, k, nz, dz);
                // Adjoint strains.
                let exx_a = ddx(&ua.ux, i, j, k, nx, dx);
                let eyy_a = ddy(&ua.uy, i, j, k, ny, dy);
                let ezz_a = ddz(&ua.uz, i, j, k, nz, dz);
                let gxy_a = ddx(&ua.uy, i, j, k, nx, dx) + ddy(&ua.ux, i, j, k, ny, dy);
                let gxz_a = ddx(&ua.uz, i, j, k, nx, dx) + ddz(&ua.ux, i, j, k, nz, dz);
                let gyz_a = ddy(&ua.uz, i, j, k, ny, dy) + ddz(&ua.uy, i, j, k, nz, dz);

                illum[[i, j, k]] += dt
                    * 2.0f64.mul_add(
                        gxy_f.mul_add(gxy_f, gxz_f.mul_add(gxz_f, gyz_f * gyz_f)),
                        4.0 * exx_f.mul_add(exx_f, eyy_f.mul_add(eyy_f, ezz_f * ezz_f)),
                    );
                let s = 4.0f64.mul_add(
                    exx_f.mul_add(exx_a, eyy_f.mul_add(eyy_a, ezz_f * ezz_a)),
                    2.0 * gxy_f.mul_add(gxy_a, gxz_f.mul_add(gxz_a, gyz_f * gyz_a)),
                );
                // `K_μ = −∫ S dt` (ADR 033 §2). The minus sign yields a valid
                // descent direction (κ ≈ +1.4, stable) per the directional
                // gradient check; κ ≠ 1 is the approximate-adjoint deviation
                // (PML + velocity-Verlet are not an exact self-adjoint pair).
                grad[[i, j, k]] -= dt * s;
            }
        }
    }
}

/// Full-history oracle for the streamed adjoint accumulator.
///
/// The per-voxel formula is intentionally re-expressed instead of calling the
/// production accumulator so component-association defects remain observable.
#[cfg(test)]
fn k_mu_kernel_from_histories(
    fwd: &[ElasticWaveField],
    adj: &[ElasticWaveField],
    dt: f64,
    grid_spacing: (f64, f64, f64),
) -> (Array3<f64>, Array3<f64>) {
    assert_eq!(fwd.len(), adj.len(), "forward/adjoint history length");
    let dim = fwd.first().expect("non-empty forward history").ux.shape();
    let [nx, ny, nz] = dim;
    let mut grad = Array3::<f64>::zeros(dim);
    let mut illum = Array3::<f64>::zeros(dim);
    let n_steps = fwd.len();
    for (adjoint_step, adjoint) in adj.iter().enumerate() {
        let forward = &fwd[n_steps - 1 - adjoint_step];
        let (dx, dy, dz) = grid_spacing;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let exx_f = ddx(&forward.ux, i, j, k, nx, dx);
                    let eyy_f = ddy(&forward.uy, i, j, k, ny, dy);
                    let ezz_f = ddz(&forward.uz, i, j, k, nz, dz);
                    let gxy_f =
                        ddx(&forward.uy, i, j, k, nx, dx) + ddy(&forward.ux, i, j, k, ny, dy);
                    let gxz_f =
                        ddx(&forward.uz, i, j, k, nx, dx) + ddz(&forward.ux, i, j, k, nz, dz);
                    let gyz_f =
                        ddy(&forward.uz, i, j, k, ny, dy) + ddz(&forward.uy, i, j, k, nz, dz);
                    let exx_a = ddx(&adjoint.ux, i, j, k, nx, dx);
                    let eyy_a = ddy(&adjoint.uy, i, j, k, ny, dy);
                    let ezz_a = ddz(&adjoint.uz, i, j, k, nz, dz);
                    let gxy_a =
                        ddx(&adjoint.uy, i, j, k, nx, dx) + ddy(&adjoint.ux, i, j, k, ny, dy);
                    let gxz_a =
                        ddx(&adjoint.uz, i, j, k, nx, dx) + ddz(&adjoint.ux, i, j, k, nz, dz);
                    let gyz_a =
                        ddy(&adjoint.uz, i, j, k, ny, dy) + ddz(&adjoint.uy, i, j, k, nz, dz);

                    illum[[i, j, k]] += dt
                        * 2.0f64.mul_add(
                            gxy_f.mul_add(gxy_f, gxz_f.mul_add(gxz_f, gyz_f * gyz_f)),
                            4.0 * exx_f.mul_add(exx_f, eyy_f.mul_add(eyy_f, ezz_f * ezz_f)),
                        );
                    let correlation = 4.0f64.mul_add(
                        exx_f.mul_add(exx_a, eyy_f.mul_add(eyy_a, ezz_f * ezz_a)),
                        2.0 * gxy_f.mul_add(gxy_a, gxz_f.mul_add(gxz_a, gyz_f * gyz_a)),
                    );
                    grad[[i, j, k]] -= dt * correlation;
                }
            }
        }
    }
    (grad, illum)
}

/// Second-order central difference along x at `(i, j, k)`; zero at the x-edges.
#[inline]
fn ddx(a: &Array3<f64>, i: usize, j: usize, k: usize, nx: usize, dx: f64) -> f64 {
    if i == 0 || i + 1 >= nx {
        return 0.0;
    }
    (a[[i + 1, j, k]] - a[[i - 1, j, k]]) / (2.0 * dx)
}

/// Second-order central difference along y at `(i, j, k)`; zero at the y-edges.
#[inline]
fn ddy(a: &Array3<f64>, i: usize, j: usize, k: usize, ny: usize, dy: f64) -> f64 {
    if j == 0 || j + 1 >= ny {
        return 0.0;
    }
    (a[[i, j + 1, k]] - a[[i, j - 1, k]]) / (2.0 * dy)
}

/// Second-order central difference along z at `(i, j, k)`; zero at the z-edges
/// (so a singleton `nz = 1` axis contributes no out-of-plane derivative).
#[inline]
fn ddz(a: &Array3<f64>, i: usize, j: usize, k: usize, nz: usize, dz: f64) -> f64 {
    if k == 0 || k + 1 >= nz {
        return 0.0;
    }
    (a[[i, j, k + 1]] - a[[i, j, k - 1]]) / (2.0 * dz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::elastic::swe::ElasticWaveConfig;
    use kwavers_grid::Grid;
    use kwavers_medium::homogeneous::HomogeneousMedium;

    #[test]
    fn streamed_kernel_matches_full_history_oracle() {
        let grid = Grid::new(10, 10, 10, 1.0e-3, 1.0e-3, 1.0e-3).expect("grid");
        let medium = HomogeneousMedium::elastic_homogeneous(1000.0, 3.464_101_6, 2.0, &grid)
            .expect("medium");
        let config = ElasticWaveConfig {
            pml_thickness: 2,
            ..ElasticWaveConfig::default()
        };
        let solver = ElasticWaveSolver::new(&grid, &medium, config).expect("solver");
        let dt = solver.recommended_timestep(0.3);
        let n_steps = 24;

        let mut forward_force = ElasticPointForce::zeros((4, 5, 5), n_steps);
        forward_force.fx[0] = 4.0e6;
        forward_force.fy[2] = -7.0e6;
        forward_force.fz[5] = 3.0e6;
        let forward = solver
            .propagate_point_forces(n_steps, dt, &[forward_force])
            .expect("forward history");

        let mut adjoint_force = ElasticPointForce::zeros((5, 5, 5), n_steps);
        adjoint_force.fx[1] = -2.0e6;
        adjoint_force.fy[4] = 5.0e6;
        adjoint_force.fz[7] = 6.0e6;
        let adjoint_forces = [adjoint_force];
        let adjoint = solver
            .propagate_point_forces(n_steps, dt, &adjoint_forces)
            .expect("adjoint history");

        let expected = k_mu_kernel_from_histories(&forward, &adjoint, dt, grid.spacing());
        let actual = stream_k_mu_kernel(&solver, &forward, &adjoint_forces, dt, grid.spacing())
            .expect("streamed kernel");

        assert!(expected.0.iter().any(|value| *value != 0.0));
        assert!(expected.1.iter().any(|value| *value > 0.0));
        assert_eq!(actual.0, expected.0, "streamed gradient");
        assert_eq!(actual.1, expected.1, "streamed illumination");
    }
}
