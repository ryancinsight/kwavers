//! Linearised 2D FDTD Green's function precomputation.
//!
//! # Mathematical model
//!
//! Solves the linearised wave equation on a 2D grid:
//!
//! ```text
//! ∂²p/∂t² = c(x,y)² ∇²p
//! ```
//!
//! with a sinusoidal point source at `(source_x, element_y)` and a
//! quadratic PML absorbing boundary.
//!
//! # Lock-in detection
//!
//! The monochromatic complex Green's function is extracted via:
//!
//! ```text
//! G(x,y) = (2 / N) Σ_{n=n₀}^{N+n₀} p(x,y,n·dt) exp(−iω n·dt)
//! ```
//!
//! Accumulation begins after `accum_start` steps (once reflections have
//! arrived and steady state is established), giving the full complex
//! pressure field due to unit-amplitude driving of element i.

use leto::Array2;
use moirai_parallel::{map_collect_with, Adaptive};

use super::config::StandingWaveOptConfig;
use crate::parallel::{zip_mut_four_refs, zip_mut_ref, zip_two_mut_ref};

/// Five-point Laplacian with edge-replication (Neumann) boundary.
///
/// Returns ∇²f / dx² (scaled by 1/dx² already included via `idx = 1/dx²`).
fn laplacian(field: &Array2<f64>, idx: f64) -> Array2<f64> {
    let nx = field.shape()[0];
    let ny = field.shape()[1];
    let mut lap = Array2::zeros((nx, ny));
    for xi in 0..nx {
        for yi in 0..ny {
            let c = field[[xi, yi]];
            let l = if xi > 0 { field[[xi - 1, yi]] } else { c };
            let r = if xi + 1 < nx { field[[xi + 1, yi]] } else { c };
            let d = if yi > 0 { field[[xi, yi - 1]] } else { c };
            let u = if yi + 1 < ny { field[[xi, yi + 1]] } else { c };
            lap[[xi, yi]] = (l + r + d + u - 4.0 * c) * idx;
        }
    }
    lap
}

/// FDTD Green's function for one transducer element.
///
/// Drives `p_src = sin(ω t)` at `(source_x, element_y)` and returns
/// `(G_re, G_im)` — the real and imaginary parts of the monochromatic
/// complex pressure field at every grid point.
///
/// The pair is normalised to unit source amplitude; callers multiply by
/// `exp(iφ_i)` for array steering.
pub(super) fn compute_green_function(
    c_map: &Array2<f64>,
    damp: &Array2<f64>,
    element_y: usize,
    config: &StandingWaveOptConfig,
) -> (Array2<f64>, Array2<f64>) {
    let nx = config.nx;
    let ny = config.ny;
    let dx = config.dx_m;
    let dt = config.dt();
    let omega = config.omega();
    let idx = 1.0 / (dx * dx);

    let c2: Array2<f64> = c_map.mapv(|c| c * c);

    // Propagation time (round trip source→far edge→back) for steady state
    let prop_steps = ((2.0 * nx as f64 * dx / config.c_ref_m_s / dt).ceil() as usize).max(20);
    let burst_steps = ((config.burst_cycles / (config.frequency_hz * dt)).ceil() as usize).max(10);
    let skip_steps =
        ((config.accum_skip_cycles / (config.frequency_hz * dt)).ceil() as usize).max(10);
    let total_steps = prop_steps + burst_steps + skip_steps;
    // Start lock-in after half the propagation time + quarter of the burst
    let accum_start = prop_steps / 2 + burst_steps / 4;

    let mut p_prev = Array2::<f64>::zeros((nx, ny));
    let mut p_curr = Array2::<f64>::zeros((nx, ny));
    let mut acc_re = Array2::<f64>::zeros((nx, ny));
    let mut acc_im = Array2::<f64>::zeros((nx, ny));
    let mut count: usize = 0;

    for step in 0..total_steps {
        let t = step as f64 * dt;
        let lap = laplacian(&p_curr, idx);
        let dt2 = dt * dt;

        // Second-order leapfrog update
        let mut p_next = Array2::<f64>::zeros((nx, ny));
        zip_mut_four_refs(
            p_next.view_mut(),
            p_curr.view(),
            p_prev.view(),
            c2.view(),
            lap.view(),
            |pn, &pc, &pp, &c2, &l| {
                *pn = 2.0 * pc - pp + dt2 * c2 * l;
            },
        );

        // Unit sinusoidal point source
        p_next[[config.source_x, element_y]] += (omega * t).sin();

        // PML absorption
        zip_mut_ref(p_next.view_mut(), damp.view(), |pn, &d| *pn *= d);

        // Lock-in accumulation: G(x,y) += p(x,y,t) × exp(−iωt)
        if step >= accum_start {
            let cos_t = (omega * t).cos();
            let sin_t = (omega * t).sin();
            zip_two_mut_ref(
                acc_re.view_mut(),
                acc_im.view_mut(),
                p_next.view(),
                |re, im, &p| {
                    *re += p * cos_t;
                    *im -= p * sin_t;
                },
            );
            count += 1;
        }

        p_prev = p_curr;
        p_curr = p_next;
    }

    let scale = 2.0 / count.max(1) as f64;
    (acc_re.mapv(|v| v * scale), acc_im.mapv(|v| v * scale))
}

/// Precompute Green's function columns for all elements through the Atlas
/// parallel execution provider.
///
/// Returns `(G_re, G_im)` each of length `n_elements`, where `G_re`i`` and
/// `G_im`i`` are `Array2<f64>` of shape `(nx, ny)`.
pub(super) fn compute_all_green_functions(
    c_map: &Array2<f64>,
    damp: &Array2<f64>,
    element_ys: &[usize],
    config: &StandingWaveOptConfig,
) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
    let pairs: Vec<(Array2<f64>, Array2<f64>)> =
        map_collect_with::<Adaptive, _, _, _>(element_ys, |&ey| {
            compute_green_function(c_map, damp, ey, config)
        });
    pairs.into_iter().unzip()
}
