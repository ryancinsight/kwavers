//! Elastic / shear-wave full-waveform inversion (ADR 033).
//!
//! Adjoint-state FWI that reconstructs the Lamé shear modulus `μ(x)` from
//! post-burst shear-wave displacement data, closing the Ch26 §26 / Ch11 §11.14
//! "not implemented" disclosure. It is the full-waveform refinement of the
//! linear [`super::ShearWaveInversion`] (`c_S` → `μ = ρ c_S²`): where the linear
//! method resolves a smooth shear-speed map, the FWI resolves sub-wavelength
//! stiffness contrast by fitting the entire waveform.
//!
//! ## Method
//!
//! Mirrors the acoustic [`crate::inverse::fwi`] control flow
//! (forward → residual → adjoint → gradient → regularize → line search) with the
//! elastic physics swapped in:
//!
//! 1. **Forward** — the [`ElasticWaveSolver`] is run over the current `μ` map
//!    from a localized shear-wave source (point forces), recording the
//!    displacement traces `d_syn(x_r, t)` at the receivers.
//! 2. **Adjoint** — the per-component, time-reversed displacement residual is
//!    injected at the receivers as a body force and run through the *same*
//!    elastic solver (the elastic operator is self-adjoint), yielding the adjoint
//!    field `λ`.
//! 3. **Gradient** — the shear-modulus sensitivity kernel (Tromp, Tape & Liu
//!    2005; Köhn 2011)
//!    `K_μ(x) = −∫ Σ_ij (∂_i u_j + ∂_j u_i)_fwd (∂_i u_j + ∂_j u_i)_adj dt`
//!    assembled by a strain cross-correlation imaging condition.
//!
//! ## Scope (v1)
//!
//! 2-D plane strain (`nz = 1`) and `μ`-only (`ρ`, `λ` held fixed from the medium).
//! The forward field history is stored in full (`O(n_steps·N)`): the PML +
//! velocity-Verlet stepping is **not** time-reversible, so the acoustic
//! self-adjoint reconstruction trick (ADR 016) does not apply. PyO3 bindings,
//! optimal checkpointing, joint `λ/ρ`, and 3-D are deferred (ADR 033 increment 5).

mod gradient;
mod inversion;
#[cfg(test)]
mod tests;

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use ndarray::Array3;

use crate::forward::elastic::swe::{
    ElasticPointForce, ElasticWaveConfig, ElasticWaveField, ElasticWaveSolver,
};

/// Per-receiver, per-step displacement traces: `traces[receiver][step] = [ux, uy, uz]`.
pub type ReceiverTraces = Vec<Vec<[f64; 3]>>;

/// Configuration for the elastic shear-wave FWI.
#[derive(Debug, Clone)]
pub struct ElasticFwiConfig {
    /// Number of forward/adjoint time steps.
    pub n_steps: usize,
    /// Time step \[s] (fixed; obtain a stable value from
    /// [`ElasticWaveSolver::recommended_timestep`]).
    pub dt: f64,
    /// Receiver grid indices where displacement is recorded / residual injected.
    pub receivers: Vec<(usize, usize, usize)>,
    /// Forward shear-wave source (point forces).
    pub source: Vec<ElasticPointForce>,
    /// Lower clamp on the reconstructed shear modulus \[Pa].
    pub mu_min: f64,
    /// Upper clamp on the reconstructed shear modulus \[Pa].
    pub mu_max: f64,
    /// Maximum descent iterations.
    pub iterations: usize,
    /// Initial line-search step (multiplies the max-norm-normalized gradient).
    pub step_size: f64,
    /// Tikhonov weight `λ_Tik` penalizing departure from the starting model.
    pub tikhonov_weight: f64,
    /// Isotropic total-variation weight `λ_TV` (edge-preserving smoothing).
    pub tv_weight: f64,
    /// Radius \[cells] within which the gradient is zeroed around every source
    /// and receiver, suppressing the near-singular strain imprint that would
    /// otherwise dominate the max-norm-normalized descent step.
    pub mute_radius: usize,
    /// Illumination-preconditioner floor as a fraction of the peak forward
    /// strain-energy (`ε` in `g̃ = K_μ/(W + ε·max W)`). Stabilizes division in
    /// dark regions; typical `1e-2`. Set `0` to use the raw gradient.
    pub precond_eps: f64,
}

impl ElasticFwiConfig {
    /// Construct a configuration with regularization disabled and sensible
    /// optimization defaults (10 iterations, unit step, μ ∈ \[1 kPa, 1 MPa]).
    #[must_use]
    pub fn new(
        n_steps: usize,
        dt: f64,
        receivers: Vec<(usize, usize, usize)>,
        source: Vec<ElasticPointForce>,
    ) -> Self {
        Self {
            n_steps,
            dt,
            receivers,
            source,
            mu_min: 1.0e3,
            mu_max: 1.0e6,
            iterations: 10,
            step_size: 1.0,
            tikhonov_weight: 0.0,
            tv_weight: 0.0,
            mute_radius: 3,
            precond_eps: 1.0e-2,
        }
    }
}

/// Adjoint-state elastic shear-wave FWI engine (ADR 033).
///
/// Owns the [`ElasticWaveSolver`] (carrying the fixed `λ`, `ρ` and the current
/// `μ` estimate) and the observed receiver data. Build it with [`Self::new`],
/// then call [`Self::run`] to reconstruct `μ`.
#[derive(Debug)]
pub struct ElasticFwi {
    solver: ElasticWaveSolver,
    config: ElasticFwiConfig,
    observed: ReceiverTraces,
    /// Starting model, retained as the Tikhonov reference.
    mu_start: Array3<f64>,
    /// Grid spacing `(dx, dy, dz)` \[m] for the strain operators.
    grid_spacing: (f64, f64, f64),
}

impl ElasticFwi {
    /// Build the engine. `base_medium` supplies the fixed `λ`/`ρ` (its `μ` is
    /// overridden by `mu_start`, the inversion's starting model); `observed` is
    /// the measured/synthesized receiver data the inversion fits.
    ///
    /// # Errors
    /// Propagates solver construction errors; errors if `mu_start` does not match
    /// the grid shape.
    pub fn new(
        grid: &Grid,
        swe_config: ElasticWaveConfig,
        base_medium: &dyn Medium,
        mu_start: Array3<f64>,
        observed: ReceiverTraces,
        config: ElasticFwiConfig,
    ) -> KwaversResult<Self> {
        let mut solver = ElasticWaveSolver::new(grid, base_medium, swe_config)?;
        solver.set_mu(&mu_start)?;
        Ok(Self {
            solver,
            config,
            observed,
            mu_start,
            grid_spacing: grid.spacing(),
        })
    }

    /// Synthesize observed receiver data: run a forward shear-wave simulation
    /// over the true model `mu_true` and sample displacement at the receivers.
    /// `base_medium` supplies `λ`/`ρ`; the source, receivers, step count and `dt`
    /// are taken from `config` (so the synthetic acquisition matches the
    /// inversion exactly).
    ///
    /// # Errors
    /// Propagates solver construction / propagation errors.
    pub fn synthesize_observed(
        grid: &Grid,
        swe_config: ElasticWaveConfig,
        base_medium: &dyn Medium,
        mu_true: &Array3<f64>,
        config: &ElasticFwiConfig,
    ) -> KwaversResult<ReceiverTraces> {
        let mut solver = ElasticWaveSolver::new(grid, base_medium, swe_config)?;
        solver.set_mu(mu_true)?;
        let history = solver.propagate_point_forces(config.n_steps, config.dt, &config.source)?;
        Ok(sample_receivers(&history, &config.receivers))
    }

    /// Current shear-modulus estimate `μ` \[Pa].
    #[must_use]
    pub fn mu(&self) -> &Array3<f64> {
        self.solver.mu()
    }

    /// L2 data misfit `J = (Δt/2) Σ_r Σ_n Σ_c (d_syn − d_obs)²` at the model `mu`.
    ///
    /// Runs a single forward simulation; does not touch the gradient.
    ///
    /// # Errors
    /// Propagates solver errors; errors if `mu` does not match the grid shape.
    pub fn forward_misfit(&mut self, mu: &Array3<f64>) -> KwaversResult<f64> {
        self.solver.set_mu(mu)?;
        let history = self.solver.propagate_point_forces(
            self.config.n_steps,
            self.config.dt,
            &self.config.source,
        )?;
        let syn = sample_receivers(&history, &self.config.receivers);
        Ok(l2_misfit(&syn, &self.observed, self.config.dt))
    }
}

/// Sample per-receiver displacement traces from a forward/adjoint history.
pub(super) fn sample_receivers(
    history: &[ElasticWaveField],
    receivers: &[(usize, usize, usize)],
) -> ReceiverTraces {
    receivers
        .iter()
        .map(|&(i, j, k)| {
            history
                .iter()
                .map(|f| [f.ux[[i, j, k]], f.uy[[i, j, k]], f.uz[[i, j, k]]])
                .collect()
        })
        .collect()
}

/// L2 displacement misfit `J = (Δt/2) Σ_r Σ_n Σ_c (syn − obs)²`.
pub(super) fn l2_misfit(syn: &ReceiverTraces, obs: &ReceiverTraces, dt: f64) -> f64 {
    let mut acc = 0.0;
    for (sr, or) in syn.iter().zip(obs.iter()) {
        for (sn, on) in sr.iter().zip(or.iter()) {
            for c in 0..3 {
                let d = sn[c] - on[c];
                acc += d * d;
            }
        }
    }
    0.5 * dt * acc
}
