//! Adjoint run and the `K_μ` shear-strain cross-correlation imaging condition
//! (ADR 033 increment 2).

use kwavers_core::error::KwaversResult;
use leto::Array3;

use super::{l2_misfit, sample_receivers, ElasticFwi, ReceiverTraces};
use crate::forward::elastic::swe::{ElasticPointForce, ElasticWaveField};

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
        let adj = self.solver.propagate_point_forces(n, dt, &adj_forces)?;

        let (grad, illum) = k_mu_kernel(&fwd, &adj, dt, self.grid_spacing);
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
/// field in reverse time order; [`k_mu_kernel`] re-aligns it.
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
/// `adj` is in reverse time order (forward-run adjoint), so the adjoint field at
/// physical step `n` is `adj[N−1−n]`. Spatial derivatives use second-order
/// central differences (zero at the grid boundary, inside the PML). The second
/// returned array is the forward strain-energy (pseudo-Hessian diagonal /
/// illumination): the same auto-correlation with `f` in both slots.
fn k_mu_kernel(
    fwd: &[ElasticWaveField],
    adj: &[ElasticWaveField],
    dt: f64,
    (dx, dy, dz): (f64, f64, f64),
) -> (Array3<f64>, Array3<f64>) {
    let n = fwd.len();
    let dim = fwd[0].ux.shape();
    let [nx, ny, nz] = dim;
    let mut grad = Array3::<f64>::zeros(dim);
    let mut illum = Array3::<f64>::zeros(dim);

    for step in 0..n {
        let uf = &fwd[step];
        let ua = &adj[n - 1 - step];
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
