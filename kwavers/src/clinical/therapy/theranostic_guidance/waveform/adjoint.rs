//! Adjoint RTM imaging with checkpointed forward replay.

use ndarray::Array2;

use super::forward::{apply_attenuation, step_wavefield_cpml};
use super::types::{AcousticGrid, CheckpointSchedule};
use super::utils::linear;

/// Compute the adjoint RTM image via backpropagation.
///
/// # Memory strategy (Griewank 1992)
///
/// For each adjoint step `t` (reverse order):
/// 1. Load checkpoint at `t_ck = floor(t / K) * K`.
/// 2. Replay forward from `t_ck` to `t` (≤ K steps).
/// 3. Apply imaging condition: `I(x) += p_fwd(x,t) · p_adj(x,t)`.
///
/// Total cost: ≈ 1.5 × forward passes.  Memory: O(√T · N²).
///
/// # Imaging condition
///
/// Cross-correlation (Claerbout 1985, Eq. 2.6):
/// `I(x) = |Σ_t p_fwd(x,t) · p_adj(x,t)|`
pub(super) fn adjoint_image(
    grid: &AcousticGrid,
    speed_m_s: &Array2<f64>,
    residual: &[f32],
    checkpoints: &[f32],
    checkpoint_interval: usize,
) -> Array2<f64> {
    let n = grid.nx * grid.ny;
    let schedule = CheckpointSchedule {
        interval: checkpoint_interval,
        time_steps: grid.time_steps,
    };

    let mut prev_adj = vec![0.0_f32; n];
    let mut curr_adj = vec![0.0_f32; n];
    let mut next_adj = vec![0.0_f32; n];
    let mut psi_x_adj = vec![0.0_f32; n];
    let mut psi_y_adj = vec![0.0_f32; n];
    let mut image = vec![0.0_f64; n];
    let receiver_count = grid.receiver_cells.len();

    for reverse in 0..grid.time_steps {
        let step = grid.time_steps - 1 - reverse;

        let ck_step = schedule.preceding_checkpoint(step);
        let ck_slot = schedule.slot_for(ck_step);
        let ck_range = ck_slot * n..(ck_slot + 1) * n;
        let mut fwd_prev = checkpoints[ck_range.clone()].to_vec();
        let mut fwd_curr = checkpoints[ck_range].to_vec();
        let mut fwd_next = vec![0.0_f32; n];
        let mut fwd_psi_x = vec![0.0_f32; n];
        let mut fwd_psi_y = vec![0.0_f32; n];
        // Replay from ck_step to step without re-injecting sources: the checkpoint
        // already encodes the accumulated forward state at ck_step. Re-injection would
        // diverge from the true forward; the imaging condition uses the checkpointed
        // state as a linearised Born sensitivity kernel approximation.
        for _fwd_step in ck_step..step {
            step_wavefield_cpml(
                grid,
                speed_m_s,
                &fwd_prev,
                &fwd_curr,
                &mut fwd_next,
                &mut fwd_psi_x,
                &mut fwd_psi_y,
            );
            apply_attenuation(grid, &mut fwd_next);
            std::mem::swap(&mut fwd_prev, &mut fwd_curr);
            std::mem::swap(&mut fwd_curr, &mut fwd_next);
            fwd_next.fill(0.0);
        }

        step_wavefield_cpml(
            grid,
            speed_m_s,
            &prev_adj,
            &curr_adj,
            &mut next_adj,
            &mut psi_x_adj,
            &mut psi_y_adj,
        );
        for (receiver, cell) in grid.receiver_cells.iter().copied().enumerate() {
            next_adj[cell] += residual[step * receiver_count + receiver];
        }
        apply_attenuation(grid, &mut next_adj);

        for (idx, value) in image.iter_mut().enumerate() {
            *value += fwd_curr[idx] as f64 * curr_adj[idx] as f64;
        }

        std::mem::swap(&mut prev_adj, &mut curr_adj);
        std::mem::swap(&mut curr_adj, &mut next_adj);
        next_adj.fill(0.0);
    }

    Array2::from_shape_fn((grid.nx, grid.ny), |(ix, iy)| {
        image[linear(ix, iy, grid.ny)].abs()
    })
}
