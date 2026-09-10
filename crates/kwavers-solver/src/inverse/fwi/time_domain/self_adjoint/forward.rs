//! The forward sweep, in the three shapes its callers need: full field,
//! sensor traces only, and the tail a reconstruction restarts from.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::{Array2, Array3, Array4, ArrayView3};

use super::operators::{
    apply_helmholtz, array3_from_view, coeffs, leapfrog_combine, validate, w_inverse,
};
use super::types::{Acquisition, SelfAdjointConfig, Spacing};

/// Run the self-adjoint forward model.
///
/// Returns `(synthetic, history)` where `synthetic` is `(n_receivers, nt)` and
/// `history` is `(nt, nx, ny, nz)` holding `p^0 … p^{nt−1}` (`p^0 = 0`).
/// `damping` is the optional self-adjoint sponge `b(x) ≥ 0` (`None` ⇒ lossless,
/// reflecting boundaries).
pub(crate) fn forward(
    model_c: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
    grid: &Grid,
    cfg: &SelfAdjointConfig,
    acq: &Acquisition<'_>,
    damping: Option<ArrayView3<'_, f64>>,
) -> KwaversResult<(Array2<f64>, Array4<f64>)> {
    validate(model_c, density, grid, cfg, acq)?;
    let (nx, ny, nz) = grid.dimensions();
    let sp = Spacing::new(grid);
    let inv_rho = array3_from_view(density).mapv(|r| 1.0 / r);
    let wm1 = w_inverse(model_c, density);
    let co = coeffs(&wm1, damping, cfg.dt);
    let scalar_src = acq.source_signal.shape()[0] == 1;

    let mut history = Array4::<f64>::zeros((cfg.nt, nx, ny, nz));
    let mut p_prev = Array3::<f64>::zeros((nx, ny, nz)); // p^{n-1}
    let mut p_curr = Array3::<f64>::zeros((nx, ny, nz)); // p^{n}
    let mut p_next = Array3::<f64>::zeros((nx, ny, nz)); // reused scratch (fully overwritten)
    let mut dlap = Array3::<f64>::zeros((nx, ny, nz));

    // history[0] = p^0 = 0 (already zero).
    for n in 0..cfg.nt - 1 {
        apply_helmholtz(p_curr.view(), inv_rho.view(), &sp, &mut dlap);
        leapfrog_combine(&mut p_next, &dlap, &co, &p_curr, &p_prev);
        for (idx, &(i, j, k)) in acq.source_voxels.iter().enumerate() {
            let s = if scalar_src {
                acq.source_signal[[0, n]]
            } else {
                acq.source_signal[[idx, n]]
            };
            p_next[[i, j, k]] += co.inv_a_plus[[i, j, k]] * s;
        }
        history.index_axis_mut(0, n + 1).unwrap().assign(&p_next);
        // Rotate buffers (p_prev, p_curr, p_next) ← (p_curr, p_next, old p_prev):
        // two pointer swaps, no allocation; old p_prev becomes next step's scratch.
        std::mem::swap(&mut p_prev, &mut p_curr);
        std::mem::swap(&mut p_curr, &mut p_next);
    }

    let mut synthetic = Array2::<f64>::zeros((acq.receiver_voxels.len(), cfg.nt));
    for (r, &(i, j, k)) in acq.receiver_voxels.iter().enumerate() {
        for n in 0..cfg.nt {
            synthetic[[r, n]] = history[[n, i, j, k]];
        }
    }
    Ok((synthetic, history))
}

/// Sensor-only forward (no history retained); used for the line search / FD.
pub(crate) fn forward_sensor_only(
    model_c: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
    grid: &Grid,
    cfg: &SelfAdjointConfig,
    acq: &Acquisition<'_>,
    damping: Option<ArrayView3<'_, f64>>,
) -> KwaversResult<Array2<f64>> {
    validate(model_c, density, grid, cfg, acq)?;
    let (nx, ny, nz) = grid.dimensions();
    let sp = Spacing::new(grid);
    let inv_rho = array3_from_view(density).mapv(|r| 1.0 / r);
    let wm1 = w_inverse(model_c, density);
    let co = coeffs(&wm1, damping, cfg.dt);
    let scalar_src = acq.source_signal.shape()[0] == 1;

    let mut p_prev = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_curr = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_next = Array3::<f64>::zeros((nx, ny, nz)); // reused scratch (fully overwritten)
    let mut dlap = Array3::<f64>::zeros((nx, ny, nz));
    let mut synthetic = Array2::<f64>::zeros((acq.receiver_voxels.len(), cfg.nt));
    // n = 0 trace is p^0 = 0 (already zero).
    for n in 0..cfg.nt - 1 {
        apply_helmholtz(p_curr.view(), inv_rho.view(), &sp, &mut dlap);
        leapfrog_combine(&mut p_next, &dlap, &co, &p_curr, &p_prev);
        for (idx, &(i, j, k)) in acq.source_voxels.iter().enumerate() {
            let s = if scalar_src {
                acq.source_signal[[0, n]]
            } else {
                acq.source_signal[[idx, n]]
            };
            p_next[[i, j, k]] += co.inv_a_plus[[i, j, k]] * s;
        }
        for (r, &(i, j, k)) in acq.receiver_voxels.iter().enumerate() {
            synthetic[[r, n + 1]] = p_next[[i, j, k]];
        }
        std::mem::swap(&mut p_prev, &mut p_curr);
        std::mem::swap(&mut p_curr, &mut p_next);
    }
    Ok(synthetic)
}

/// Lossless forward keeping only the **final two** states `(p^{N−1}, p^{N−2})`
/// plus the receiver traces — `O(N)` memory instead of the `O(nt·N)` full
/// history. Used to seed the reverse-reconstruction gradient
/// ([`gradient_reconstructed`]), which re-derives the forward field backward in
/// lockstep with the adjoint sweep.
///
/// Returns `(synthetic, p_last, p_second_last)` with `p_last = p^{N−1}` and
/// `p_second_last = p^{N−2}`. Requires the lossless scheme (no sponge): the
/// energy-conserving leapfrog is exactly reversible, whereas a damped step would
/// anti-amplify round-off when reconstructed backward.
pub(crate) fn forward_tail(
    model_c: ArrayView3<'_, f64>,
    density: ArrayView3<'_, f64>,
    grid: &Grid,
    cfg: &SelfAdjointConfig,
    acq: &Acquisition<'_>,
) -> KwaversResult<(Array2<f64>, Array3<f64>, Array3<f64>)> {
    validate(model_c, density, grid, cfg, acq)?;
    let (nx, ny, nz) = grid.dimensions();
    let sp = Spacing::new(grid);
    let inv_rho = array3_from_view(density).mapv(|r| 1.0 / r);
    let wm1 = w_inverse(model_c, density);
    let co = coeffs(&wm1, None, cfg.dt);
    let scalar_src = acq.source_signal.shape()[0] == 1;

    let mut p_prev = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_curr = Array3::<f64>::zeros((nx, ny, nz));
    let mut p_next = Array3::<f64>::zeros((nx, ny, nz)); // reused scratch (fully overwritten)
    let mut dlap = Array3::<f64>::zeros((nx, ny, nz));
    let mut synthetic = Array2::<f64>::zeros((acq.receiver_voxels.len(), cfg.nt));
    for n in 0..cfg.nt - 1 {
        apply_helmholtz(p_curr.view(), inv_rho.view(), &sp, &mut dlap);
        leapfrog_combine(&mut p_next, &dlap, &co, &p_curr, &p_prev);
        for (idx, &(i, j, k)) in acq.source_voxels.iter().enumerate() {
            let s = if scalar_src {
                acq.source_signal[[0, n]]
            } else {
                acq.source_signal[[idx, n]]
            };
            p_next[[i, j, k]] += co.inv_a_plus[[i, j, k]] * s;
        }
        for (r, &(i, j, k)) in acq.receiver_voxels.iter().enumerate() {
            synthetic[[r, n + 1]] = p_next[[i, j, k]];
        }
        std::mem::swap(&mut p_prev, &mut p_curr);
        std::mem::swap(&mut p_curr, &mut p_next);
    }
    // After the loop: p_curr = p^{N-1}, p_prev = p^{N-2}.
    Ok((synthetic, p_curr, p_prev))
}
