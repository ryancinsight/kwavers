//! Reusable transmission-acquisition setup and a convenience phantom-
//! reconstruction entry point for the elastic shear-wave FWI (ADR 033).
//!
//! This is the shared domain logic behind both the `elastic_shear_fwi_lesion`
//! example and the PyO3 `elastic_shear_fwi_reconstruct` binding, so neither
//! reimplements the acquisition (the binding layer holds no domain logic).

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use ndarray::Array3;

use super::{ElasticFwi, ElasticFwiConfig};
use crate::forward::elastic::swe::{ElasticPointForce, ElasticWaveConfig, ElasticWaveSolver};

/// In-plane force component for [`ricker_point_force`].
#[derive(Clone, Copy)]
pub enum ForceAxis {
    /// Force along x.
    X,
    /// Force along y.
    Y,
}

/// A Ricker (Mexican-hat) point force at `index` along `axis`, amplitude `amp`
/// \[N/m³], length `n_steps`.
#[must_use]
pub fn ricker_point_force(
    index: (usize, usize, usize),
    n_steps: usize,
    dt: f64,
    f0: f64,
    amp: f64,
    axis: ForceAxis,
) -> ElasticPointForce {
    let mut f = ElasticPointForce::zeros(index, n_steps);
    let t0 = 1.0 / f0; // causal delay
    let a = std::f64::consts::PI * std::f64::consts::PI * f0 * f0;
    for n in 0..n_steps {
        let t = n as f64 * dt - t0;
        let arg = a * t * t;
        let v = amp * 2.0f64.mul_add(-arg, 1.0) * (-arg).exp();
        match axis {
            ForceAxis::X => f.fx[n] = v,
            ForceAxis::Y => f.fy[n] = v,
        }
    }
    f
}

/// Build a crossed four-side transmission acquisition sized to `grid`: lines of
/// in-phase shear forces on the left/right (y-polarized) and top/bottom
/// (x-polarized), with a receiver ring just inside the source ring. Returns
/// `(sources, receivers)`. Each plane shear wave traverses the interior, giving
/// the full angular coverage that constrains an interior inclusion.
#[must_use]
pub fn four_side_transmission_acquisition(
    grid: &Grid,
    n_steps: usize,
    dt: f64,
    f0: f64,
    amp: f64,
) -> (Vec<ElasticPointForce>, Vec<(usize, usize, usize)>) {
    let (nx, ny, _nz) = grid.dimensions();
    let n = nx.min(ny);
    // Source ring two cells inside the grid edge; receiver ring one cell further.
    let s_lo = 7;
    let s_hi = n.saturating_sub(8);
    let r_lo = 9;
    let r_hi = n.saturating_sub(10);
    let mut sources = Vec::new();
    let mut receivers = Vec::new();
    let mut k = r_lo;
    while k < r_hi {
        sources.push(ricker_point_force(
            (s_lo, k, 0),
            n_steps,
            dt,
            f0,
            amp,
            ForceAxis::Y,
        ));
        sources.push(ricker_point_force(
            (s_hi, k, 0),
            n_steps,
            dt,
            f0,
            amp,
            ForceAxis::Y,
        ));
        sources.push(ricker_point_force(
            (k, s_lo, 0),
            n_steps,
            dt,
            f0,
            amp,
            ForceAxis::X,
        ));
        sources.push(ricker_point_force(
            (k, s_hi, 0),
            n_steps,
            dt,
            f0,
            amp,
            ForceAxis::X,
        ));
        receivers.push((r_lo, k, 0));
        receivers.push((r_hi, k, 0));
        receivers.push((k, r_lo, 0));
        receivers.push((k, r_hi, 0));
        k += 2;
    }
    (sources, receivers)
}

/// Parameters for [`reconstruct_lesion_transmission`].
#[derive(Debug, Clone)]
pub struct TransmissionFwiParams {
    /// Number of forward/adjoint time steps.
    pub n_steps: usize,
    /// Descent iterations.
    pub iterations: usize,
    /// Source centre frequency \[Hz].
    pub f0: f64,
    /// Source force-density amplitude \[N/m³].
    pub amp: f64,
    /// CFL factor for the (stiffest-model) time step.
    pub cfl: f64,
    /// Initial line-search step as a fraction of the background μ.
    pub step_frac: f64,
    /// Illumination-preconditioner floor (`0` ⇒ raw gradient).
    pub precond_eps: f64,
    /// Gradient mute radius \[cells] around acquisition points.
    pub mute_radius: usize,
}

impl Default for TransmissionFwiParams {
    fn default() -> Self {
        Self {
            n_steps: 200,
            iterations: 16,
            f0: 300.0,
            amp: 1.0e7,
            cfl: 0.3,
            step_frac: 1.0,
            precond_eps: 0.1,
            mute_radius: 4,
        }
    }
}

/// Reconstruct `μ` from a known phantom via four-side transmission FWI.
///
/// `medium` supplies the fixed `λ`/`ρ` and the homogeneous background `μ` (the
/// inversion's start model); `mu_true` is the phantom whose synthetic shear-wave
/// response is inverted. Returns the recovered `μ` map. The CFL-stable `dt` is
/// taken from the *stiffest* μ in `mu_true` so the forward run is stable
/// everywhere.
///
/// # Errors
/// Propagates solver construction / propagation errors.
pub fn reconstruct_lesion_transmission(
    grid: &Grid,
    medium: &dyn Medium,
    mu_true: &Array3<f64>,
    params: &TransmissionFwiParams,
) -> KwaversResult<Array3<f64>> {
    let swe = || ElasticWaveConfig {
        time_step: 0.0,
        save_every: 1,
        pml_thickness: 6,
        ..ElasticWaveConfig::default()
    };

    // Background (start) model and a CFL-stable dt for the stiffest cell.
    let base = ElasticWaveSolver::new(grid, medium, swe())?;
    let mu_start = base.mu().clone();
    let mu_max = mu_true.iter().cloned().fold(0.0_f64, f64::max);
    let dt = {
        let mut s = ElasticWaveSolver::new(grid, medium, swe())?;
        s.set_mu(&Array3::from_elem(grid.dimensions(), mu_max))?;
        s.recommended_timestep(params.cfl)
    };

    let (sources, receivers) =
        four_side_transmission_acquisition(grid, params.n_steps, dt, params.f0, params.amp);

    let mu_bg = mu_start.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    let mut cfg = ElasticFwiConfig::new(params.n_steps, dt, receivers, sources);
    cfg.iterations = params.iterations;
    cfg.step_size = params.step_frac * mu_bg;
    cfg.mu_min = 0.25 * mu_bg;
    cfg.mu_max = 8.0 * mu_bg;
    cfg.precond_eps = params.precond_eps;
    cfg.mute_radius = params.mute_radius;

    let observed = ElasticFwi::synthesize_observed(grid, swe(), medium, mu_true, &cfg)?;
    let mut fwi = ElasticFwi::new(grid, swe(), medium, mu_start, observed, cfg)?;
    fwi.run()
}
