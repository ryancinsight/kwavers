//! Non-rigid free-form deformation (FFD) reparametrisation for MOFI.
//!
//! Generalises the rigid SE(2) reparametrisation ([`super::transform`]) to a
//! smooth deformation driven by a coarse lattice of control-point displacements
//! (the paper's "larger Lie groups / non-rigid" direction, §Discussion). A dense
//! displacement field `u(x)` is bilinearly interpolated from the control lattice;
//! the warped model is `c_u(x) = c_template(x − u(x))`. As with rigid MOFI, the
//! parameters are optimised against the *acoustic* misfit through the chained
//! gradient `∂f/∂u_cp = (∂c_u/∂u_cp)ᵀ ∂f/∂c`, using the exact self-adjoint
//! `∂f/∂c` (ADR 016). A bending-energy (first-difference) smoothness penalty on
//! the lattice keeps the high-DOF problem well-posed.
//!
//! Per pixel `x` with control-lattice bilinear weights `w_cp(x)` and template
//! gradient `∇c` at the back-mapped point:
//! ```text
//! ∂c_u(x)/∂u_x,cp = −(∂c/∂x)·w_cp(x),   ∂c_u(x)/∂u_y,cp = −(∂c/∂y)·w_cp(x)
//! ∂f/∂u_x,cp = Σ_x g(x)·∂c_u(x)/∂u_x,cp   (g = ∂f/∂c)
//! ```

use super::transform::{bilinear_with_gradient, PlaneGeometry};
use crate::inverse::fwi::time_domain::{FwiEngine, FwiGeometry, FwiProcessor};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use leto::{
    Array2,
    Array3,
};

/// Control-lattice interpolation basis for the FFD displacement field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FfdBasis {
    /// Bilinear (2×2 support, C⁰) — fewest operations.
    #[default]
    Bilinear,
    /// Uniform cubic B-spline (4×4 support, C²) — smoother deformations
    /// (standard Rueckert FFD).
    CubicBSpline,
}

/// Free-form-deformation settings.
#[derive(Debug, Clone, Copy)]
pub struct FfdConfig {
    /// Control points along x (≥ 2).
    pub n_ctrl_x: usize,
    /// Control points along y (≥ 2).
    pub n_ctrl_y: usize,
    /// Control-lattice interpolation basis.
    pub basis: FfdBasis,
    /// Maximum gradient-descent iterations.
    pub max_iterations: usize,
    /// Initial line-search step [m] (displacement magnitude).
    pub initial_step_m: f64,
    /// Armijo sufficient-decrease constant.
    pub armijo_c1: f64,
    /// Maximum Armijo backtracking halvings.
    pub max_line_search: usize,
    /// Bending-energy smoothness weight, **relative** to the data-misfit scale.
    /// The effective absolute penalty is `smoothness_weight · J₀_data / dx²`, so
    /// `(Δu/dx)²` lattice differences are penalised as a fraction of the initial
    /// data misfit — robust to the problem's absolute amplitude scale. O(1e-2)
    /// is a light regulariser; larger values enforce stiffer deformations.
    pub smoothness_weight: f64,
    /// Background sound speed for out-of-domain template pixels [m/s].
    pub background_c: f64,
    /// Relative-misfit-change convergence tolerance.
    pub tolerance: f64,
}

impl Default for FfdConfig {
    fn default() -> Self {
        Self {
            n_ctrl_x: 4,
            n_ctrl_y: 4,
            basis: FfdBasis::Bilinear,
            max_iterations: 40,
            initial_step_m: 2e-3,
            armijo_c1: 1e-4,
            max_line_search: 15,
            smoothness_weight: 1e-4,
            background_c: 1500.0,
            tolerance: 1e-6,
        }
    }
}

/// Control-lattice displacement field (metres), row-major `[iy*n_ctrl_x + ix]`.
#[derive(Debug, Clone, PartialEq)]
pub struct FfdField {
    pub n_ctrl_x: usize,
    pub n_ctrl_y: usize,
    pub ux: Vec<f64>,
    pub uy: Vec<f64>,
    /// Interpolation basis used to expand this lattice to a dense field.
    pub basis: FfdBasis,
}

impl FfdField {
    /// A zero (identity) deformation lattice with the given interpolation basis.
    #[must_use]
    pub fn zeros(n_ctrl_x: usize, n_ctrl_y: usize, basis: FfdBasis) -> Self {
        let n = n_ctrl_x.max(2) * n_ctrl_y.max(2);
        Self {
            n_ctrl_x: n_ctrl_x.max(2),
            n_ctrl_y: n_ctrl_y.max(2),
            ux: vec![0.0; n],
            uy: vec![0.0; n],
            basis,
        }
    }

    fn len(&self) -> usize {
        self.ux.len()
    }
}

/// Result of a non-rigid FFD alignment.
#[derive(Debug, Clone)]
pub struct FfdResult {
    pub field: FfdField,
    pub initial_misfit: f64,
    pub final_misfit: f64,
    pub iterations: usize,
}

/// 1-D control weights along one axis for a pixel index `i` (grid size `n`,
/// `nc` control points). Returns `(control_index, weight)` pairs and their count:
/// 2 for [`FfdBasis::Bilinear`], 4 for [`FfdBasis::CubicBSpline`]. Indices are
/// clamped to `[0, nc−1]` (coincident clamped entries simply sum downstream).
fn axis_weights(i: usize, n: usize, nc: usize, basis: FfdBasis) -> ([(usize, f64); 4], usize) {
    let u = (i as f64) * (nc - 1) as f64 / ((n - 1).max(1) as f64);
    match basis {
        FfdBasis::Bilinear => {
            let p0 = (u.floor() as usize).min(nc - 1);
            let p1 = (p0 + 1).min(nc - 1);
            let t = u - p0 as f64;
            ([(p0, 1.0 - t), (p1, t), (0, 0.0), (0, 0.0)], 2)
        }
        FfdBasis::CubicBSpline => {
            let p = u.floor() as i64;
            let t = u - p as f64;
            let t2 = t * t;
            let t3 = t2 * t;
            // Uniform cubic B-spline basis (sum = 1) for points p−1..p+2.
            let b0 = (1.0 - t).powi(3) / 6.0;
            let b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
            let b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
            let b3 = t3 / 6.0;
            let clamp = |q: i64| q.clamp(0, nc as i64 - 1) as usize;
            (
                [
                    (clamp(p - 1), b0),
                    (clamp(p), b1),
                    (clamp(p + 1), b2),
                    (clamp(p + 2), b3),
                ],
                4,
            )
        }
    }
}

/// Per-pixel control-lattice weights (tensor product of the 1-D axis weights):
/// up to 4×4 = 16 `(control_index, weight)` pairs, with the active count.
fn control_weights(
    i: usize,
    j: usize,
    nx: usize,
    ny: usize,
    field: &FfdField,
) -> ([(usize, f64); 16], usize) {
    let (wx, nwx) = axis_weights(i, nx, field.n_ctrl_x, field.basis);
    let (wy, nwy) = axis_weights(j, ny, field.n_ctrl_y, field.basis);
    let mut out = [(0usize, 0.0_f64); 16];
    let mut n = 0usize;
    for &(ay, wyv) in &wy[..nwy] {
        for &(ax, wxv) in &wx[..nwx] {
            out[n] = (ay * field.n_ctrl_x + ax, wxv * wyv);
            n += 1;
        }
    }
    (out, n)
}

/// Warp the template by the FFD field: `c_u(x) = c_template(x − u(x))`.
pub(super) fn warp_template(
    template: &Array3<f64>,
    field: &FfdField,
    geom: &PlaneGeometry,
    background: f64,
) -> Array3<f64> {
    let (nx, ny, nz) = template.dim();
    let mut out = Array3::from_elem((nx, ny, nz), background);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (w, nw) = control_weights(i, j, nx, ny, field);
                let mut ux = 0.0;
                let mut uy = 0.0;
                for &(cp, wt) in &w[..nw] {
                    ux += wt * field.ux[cp];
                    uy += wt * field.uy[cp];
                }
                let fi = i as f64 - ux / geom.dx;
                let fj = j as f64 - uy / geom.dy;
                let (val, _, _) = bilinear_with_gradient(template, k, fi, fj, geom, background);
                out[[i, j, k]] = val;
            }
        }
    }
    out
}

/// Scatter the pixel-wise model gradient `g = ∂f/∂c` onto the control lattice.
fn project_warp_gradient(
    template: &Array3<f64>,
    field: &FfdField,
    g: &Array3<f64>,
    geom: &PlaneGeometry,
    background: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (nx, ny, nz) = template.dim();
    let mut g_ux = vec![0.0; field.len()];
    let mut g_uy = vec![0.0; field.len()];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (w, nw) = control_weights(i, j, nx, ny, field);
                let mut ux = 0.0;
                let mut uy = 0.0;
                for &(cp, wt) in &w[..nw] {
                    ux += wt * field.ux[cp];
                    uy += wt * field.uy[cp];
                }
                let fi = i as f64 - ux / geom.dx;
                let fj = j as f64 - uy / geom.dy;
                let (_, dcx, dcy) = bilinear_with_gradient(template, k, fi, fj, geom, background);
                let gv = g[[i, j, k]];
                // ∂c_u/∂u_{x,cp} = −(∂c/∂x)·w_cp ; likewise for y.
                for &(cp, wt) in &w[..nw] {
                    g_ux[cp] += gv * (-dcx) * wt;
                    g_uy[cp] += gv * (-dcy) * wt;
                }
            }
        }
    }
    (g_ux, g_uy)
}

/// First-difference (bending-energy) smoothness penalty on the lattice and its
/// gradient: `R = λ Σ_{adjacent (a,b)} (u_a − u_b)²`.
fn smoothness(field: &FfdField, weight: f64) -> (f64, Vec<f64>, Vec<f64>) {
    let (ncx, ncy) = (field.n_ctrl_x, field.n_ctrl_y);
    let mut r = 0.0;
    let mut g_ux = vec![0.0; field.len()];
    let mut g_uy = vec![0.0; field.len()];
    let idx = |p: usize, q: usize| q * ncx + p;
    let accumulate = |a: usize, b: usize, g_ux: &mut [f64], g_uy: &mut [f64], r: &mut f64| {
        let dux = field.ux[a] - field.ux[b];
        let duy = field.uy[a] - field.uy[b];
        *r += weight * (dux * dux + duy * duy);
        g_ux[a] += 2.0 * weight * dux;
        g_ux[b] -= 2.0 * weight * dux;
        g_uy[a] += 2.0 * weight * duy;
        g_uy[b] -= 2.0 * weight * duy;
    };
    for q in 0..ncy {
        for p in 0..ncx {
            if p + 1 < ncx {
                accumulate(idx(p, q), idx(p + 1, q), &mut g_ux, &mut g_uy, &mut r);
            }
            if q + 1 < ncy {
                accumulate(idx(p, q), idx(p, q + 1), &mut g_ux, &mut g_uy, &mut r);
            }
        }
    }
    (r, g_ux, g_uy)
}

/// Non-rigid FFD alignment by regularised gradient descent with an Armijo line
/// search on the acoustic misfit + lattice bending energy.
///
/// `processor` must select [`FwiEngine::SecondOrderSelfAdjoint`]. Returns the
/// recovered control-lattice deformation. Typically run *after* rigid MOFI, with
/// `template` already rigidly aligned, to capture the residual non-rigid mismatch.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] off the self-adjoint engine;
///   propagates solve errors.
pub fn align_nonrigid(
    processor: &FwiProcessor,
    template: &Array3<f64>,
    observed: &Array2<f64>,
    geometry: &FwiGeometry,
    grid: &Grid,
    config: &FfdConfig,
) -> KwaversResult<FfdResult> {
    if processor.engine != FwiEngine::SecondOrderSelfAdjoint {
        return Err(KwaversError::InvalidInput(
            "MOFI non-rigid FFD requires FwiEngine::SecondOrderSelfAdjoint".to_owned(),
        ));
    }
    let (nx, ny, _) = grid.dimensions();
    let geom = PlaneGeometry::centered(nx, ny, grid.dx, grid.dy);
    let bg = config.background_c;

    let mut field = FfdField::zeros(config.n_ctrl_x, config.n_ctrl_y, config.basis);
    let n = field.len();

    let data_misfit = |f: &FfdField| -> KwaversResult<f64> {
        let model = warp_template(template, f, &geom, bg);
        let synth = processor.forward_model_sensor_only(&model, geometry, grid)?;
        processor.compute_misfit_objective(observed, &synth)
    };

    let initial_misfit = data_misfit(&field)?;
    // Auto-scale the bending-energy weight to the data-misfit scale so the
    // regulariser is a fixed fraction of J₀ regardless of absolute amplitude
    // (penalising lattice differences measured in cells, (Δu/dx)²).
    let lambda =
        config.smoothness_weight * initial_misfit.max(f64::MIN_POSITIVE) / (grid.dx * grid.dx);
    let mut current = initial_misfit + smoothness(&field, lambda).0;
    let mut iterations = 0usize;

    for _ in 0..config.max_iterations {
        // Data gradient at the current field.
        let model = warp_template(template, &field, &geom, bg);
        let (synth, history) = processor.forward_model(&model, geometry, grid)?;
        let data_f = processor.compute_misfit_objective(observed, &synth)?;
        let residual = processor.compute_adjoint_source(observed, &synth)?;
        let g = processor
            .adjoint_gradient_self_adjoint(&residual, &model, geometry, grid, &history, None)?;
        let (gd_ux, gd_uy) = project_warp_gradient(template, &field, &g, &geom, bg);
        let (smooth_f, gs_ux, gs_uy) = smoothness(&field, lambda);
        current = data_f + smooth_f;

        // Total gradient and its norm.
        let mut g_ux = vec![0.0; n];
        let mut g_uy = vec![0.0; n];
        let mut gnorm2 = 0.0;
        for c in 0..n {
            g_ux[c] = gd_ux[c] + gs_ux[c];
            g_uy[c] = gd_uy[c] + gs_uy[c];
            gnorm2 += g_ux[c] * g_ux[c] + g_uy[c] * g_uy[c];
        }
        let gnorm = gnorm2.sqrt();
        if gnorm <= f64::MIN_POSITIVE {
            break;
        }
        // Unit descent direction; directional derivative gᵀd = −‖g‖.
        let gd = -gnorm;

        let mut step = config.initial_step_m;
        let mut accepted: Option<(FfdField, f64)> = None;
        for _ in 0..config.max_line_search {
            let mut trial = field.clone();
            for c in 0..n {
                trial.ux[c] -= step * g_ux[c] / gnorm;
                trial.uy[c] -= step * g_uy[c] / gnorm;
            }
            let trial_total = data_misfit(&trial)? + smoothness(&trial, lambda).0;
            if trial_total <= current + config.armijo_c1 * step * gd {
                accepted = Some((trial, trial_total));
                break;
            }
            step *= 0.5;
        }
        let Some((trial, trial_total)) = accepted else {
            break;
        };
        let rel = (current - trial_total).abs() / current.max(f64::EPSILON);
        field = trial;
        current = trial_total;
        iterations += 1;
        if rel < config.tolerance {
            break;
        }
    }

    Ok(FfdResult {
        field,
        initial_misfit,
        final_misfit: current,
        iterations,
    })
}

/// Evaluate the dense displacement field `(u_x, u_y)` [m] at pixel `(i, j)` — for
/// inspecting / applying a recovered FFD.
#[must_use]
pub fn sample_displacement(
    field: &FfdField,
    i: usize,
    j: usize,
    nx: usize,
    ny: usize,
) -> (f64, f64) {
    let (w, nw) = control_weights(i, j, nx, ny, field);
    let mut ux = 0.0;
    let mut uy = 0.0;
    for &(cp, wt) in &w[..nw] {
        ux += wt * field.ux[cp];
        uy += wt * field.uy[cp];
    }
    (ux, uy)
}
