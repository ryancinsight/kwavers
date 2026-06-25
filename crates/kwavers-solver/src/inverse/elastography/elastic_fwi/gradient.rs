//! Adjoint run and the `K_μ` shear-strain cross-correlation imaging condition
//! (ADR 033 increment 2).

use kwavers_core::error::KwaversResult;
use ndarray::Array3;

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

        let (dx, dy, _dz) = self.grid_spacing;
        let (grad, illum) = k_mu_kernel(&fwd, &adj, dt, dx, dy);
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
            ndarray::Zip::from(&mut grad)
                .and(&illum)
                .for_each(|g, &w| *g /= w + floor);
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
        let (nx, ny, _nz) = grad.dim();
        let r2 = (r * r) as i64;
        let centers = self
            .config
            .receivers
            .iter()
            .copied()
            .chain(self.config.source.iter().map(|f| f.index));
        for (ci, cj, _ck) in centers {
            for i in 0..nx {
                for j in 0..ny {
                    let di = i as i64 - ci as i64;
                    let dj = j as i64 - cj as i64;
                    if di * di + dj * dj <= r2 {
                        grad[[i, j, 0]] = 0.0;
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
/// `K_μ(x) = −∫₀ᵀ Σ_ij (∂_i u_j + ∂_j u_i)_fwd (∂_i u_j + ∂_j u_i)_adj dt`.
///
/// `adj` is in reverse time order (forward-run adjoint), so the adjoint field at
/// physical step `n` is `adj[N−1−n]`. For 2-D plane strain (`nz = 1`) the
/// out-of-plane displacement and all `∂_z` vanish, leaving the in-plane sum
///
/// `S = 4 ∂_x u_x ∂_x w_x + 4 ∂_y u_y ∂_y w_y + 2(∂_x u_y + ∂_y u_x)(∂_x w_y + ∂_y w_x)`
///
/// where `u = u_fwd`, `w = u_adj`. Spatial derivatives use second-order central
/// differences (zero at the grid boundary, inside the PML).
fn k_mu_kernel(
    fwd: &[ElasticWaveField],
    adj: &[ElasticWaveField],
    dt: f64,
    dx: f64,
    dy: f64,
) -> (Array3<f64>, Array3<f64>) {
    let n = fwd.len();
    let dim = fwd[0].ux.dim();
    let (nx, ny, _nz) = dim;
    let mut grad = Array3::<f64>::zeros(dim);
    let mut illum = Array3::<f64>::zeros(dim);

    for step in 0..n {
        let uf = &fwd[step];
        let ua = &adj[n - 1 - step];
        for i in 0..nx {
            for j in 0..ny {
                let dx_uxf = ddx(&uf.ux, i, j, nx, dx);
                let dy_uyf = ddy(&uf.uy, i, j, ny, dy);
                let dx_uyf = ddx(&uf.uy, i, j, nx, dx);
                let dy_uxf = ddy(&uf.ux, i, j, ny, dy);

                let dx_uxa = ddx(&ua.ux, i, j, nx, dx);
                let dy_uya = ddy(&ua.uy, i, j, ny, dy);
                let dx_uya = ddx(&ua.uy, i, j, nx, dx);
                let dy_uxa = ddy(&ua.ux, i, j, ny, dy);

                let shear_f = dx_uyf + dy_uxf;
                let shear_a = dx_uya + dy_uxa;
                // Forward strain-energy (pseudo-Hessian diagonal): the f·f
                // auto-correlation with the same operator weights as the kernel.
                illum[[i, j, 0]] += dt
                    * 4.0f64.mul_add(
                        dx_uxf * dx_uxf,
                        2.0f64.mul_add(shear_f * shear_f, 4.0 * dy_uyf * dy_uyf),
                    );
                let s = 4.0 * dx_uxf * dx_uxa + 4.0 * dy_uyf * dy_uya + 2.0 * shear_f * shear_a;
                // `K_μ = −∫ S dt` (ADR 033 §2). With the adjoint source taken as
                // the raw residual (d_syn − d_obs) injected as a body force and
                // the forward-run reverse-time adjoint, the directional gradient
                // check `k_mu_gradient_is_valid_descent_direction` confirms this
                // minus sign yields κ = (g·δ)/FD ≈ +1.4 (stable, positive across
                // spatial bands) — a valid descent direction. κ ≠ 1 is the
                // expected approximate-adjoint deviation (PML + velocity-Verlet
                // are not an exact discrete self-adjoint pair; ADR §Verification).
                grad[[i, j, 0]] -= dt * s;
            }
        }
    }
    (grad, illum)
}

/// Second-order central difference along x at `(i, j, 0)`; zero at the x-edges.
#[inline]
fn ddx(a: &Array3<f64>, i: usize, j: usize, nx: usize, dx: f64) -> f64 {
    if i == 0 || i + 1 >= nx {
        return 0.0;
    }
    (a[[i + 1, j, 0]] - a[[i - 1, j, 0]]) / (2.0 * dx)
}

/// Second-order central difference along y at `(i, j, 0)`; zero at the y-edges.
#[inline]
fn ddy(a: &Array3<f64>, i: usize, j: usize, ny: usize, dy: f64) -> f64 {
    if j == 0 || j + 1 >= ny {
        return 0.0;
    }
    (a[[i, j + 1, 0]] - a[[i, j - 1, 0]]) / (2.0 * dy)
}
