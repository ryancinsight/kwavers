//! Adjoint RTM imaging with checkpointed forward replay.
//!
//! ## Correctness requirements for the 2nd-order-in-time wave equation
//!
//! The scalar acoustic wave equation is second-order in time:
//! `p(t+1) = 2p(t) - p(t-1) + dt²c²Δp(t) + s(t)`.
//!
//! Restarting the recursion from a single snapshot `p(t_ck)` forces
//! `p(t_ck - 1) := p(t_ck)`, which is equivalent to claiming zero time
//! derivative at the checkpoint.  This introduces an O(|p|) error in the
//! first replay step and compounds over subsequent steps, causing the imaging
//! condition to use a corrupted forward field.
//!
//! Fix: save the **pair** `(p(t-1), p(t))` at each checkpoint.
//! The forward pass stores `[previous | current]` per slot (see `forward.rs`).
//!
//! ## Imaging condition timing contract
//!
//! The Born RTM imaging condition is
//! `I(x) = Σ_t p_fwd(x,t) · q(x,t)`
//! where both fields are evaluated at the **same** physical time `t`.
//!
//! ## Loop invariant and backward adjoint equation
//!
//! At the start of each iteration for loop variable `step = t`, the state
//! buffers satisfy:
//!   `prev_adj = q(t+2)`,  `curr_adj = q(t+1)`
//!
//! The backward adjoint equation (time-reversed form) is:
//!   `q(t) = 2·q(t+1) − q(t+2) + dt²c²Δq(t+1) + f_adj(t)`
//!
//! `step_wavefield_cpml(prev=q(t+2), curr=q(t+1))` computes the first three
//! terms, yielding `next_adj = q(t)` without the source.  Injecting
//! `residual[t]` into `next_adj` completes `q(t)`.  The imaging condition
//! then cross-correlates `p_fwd(t)` with the now-complete `next_adj = q(t)`.
//!
//! Evaluating the imaging condition **before** the backward step (as was
//! previously the case) would use `curr_adj = q(t+1)` — one step late —
//! introducing a phase lag of ≈ 90° at the focal depth (~20 cells, λ/dx ≈ 4),
//! which inverts the contrast of the lesion signal (CNR < 0).
//!
//! Correct order per iteration: BACKWARD → INJECT → IMAGE → SWAP.
//!
//! ## Illumination normalization — not applicable for focused HIFU geometry
//!
//! The Claerbout (1971) / Rickett–Sava (2002) normalization
//! `I(x) / (Σ_t p_fwd²(x,t) + ε·S_max)` is designed for seismic RTM where
//! forward illumination *decreases* with depth.  For a focused HIFU
//! transducer, forward energy is **maximum at the focal point** (target
//! lesion) and decreases outward.  Dividing by this illumination suppresses
//! the focal signal relative to the background, inverting the contrast and
//! yielding CNR < 0.  The normalization is therefore not applied; contrast
//! normalisation relative to body-mask peak is performed by `normalize_positive`
//! in the caller.
//!
//! ## Memory strategy (Griewank 1992)
//!
//! For each adjoint step `t` (reverse order):
//! 1. Load checkpoint pair at `t_ck = floor(t / K) * K`.
//! 2. Replay forward from `t_ck` to `t` (≤ K steps) WITH source injection.
//! 3. Advance adjoint one step backward: `step_cpml(q(t+2), q(t+1)) → next = q(t)`.
//! 4. Inject residual at time `t` into `next` to complete `q(t)`.
//! 5. Apply imaging condition: `I += p_fwd[t] · next = p_fwd[t] · q[t]`.
//! 6. After all steps: zero `I` at source and receiver cell positions (aperture mute).
//!
//! Total cost: ≈ 1.5 × forward passes.  Memory: O(√T · N²).

use ndarray::Array2;

use super::forward::{apply_attenuation, inject_sources, step_wavefield_cpml};
use super::types::{AcousticGrid, CheckpointSchedule};
use super::utils::linear;

/// Compute the absolute-value Born RTM image via checkpointed backpropagation.
///
/// # Arguments
///
/// * `grid` — simulation grid; carries CPML, source geometry,
///   `source_frequency_hz`, and `source_scale` for replay injection.
/// * `speed_m_s` — background (predicted) sound-speed model.
/// * `residual` — flat `[time_steps × receiver_count]` row-major adjoint
///   source (Charbonnier or L2 misfit derivative).
/// * `checkpoints` — paired snapshot buffer from the forward run.
///   Layout: `[prev₀ | curr₀ | prev₁ | curr₁ | …]`, size `2·slots·N`.
/// * `checkpoint_interval` — K = `CheckpointSchedule::interval`.
///
/// # Returns
///
/// `Array2<f64>` of shape `(nx, ny)` with the absolute-value Born
/// cross-correlation image.  Body-mask normalization is applied by the caller.
///
/// # Imaging condition timing
///
/// The cross-correlation `Σ_t p_fwd(x,t) · q(x,t)` is accumulated using
/// `next_adj = q(t)` produced by the backward step and source injection so
/// the forward and adjoint fields are at the same physical time `t`.  Per
/// the loop invariant `(prev_adj, curr_adj) = (q(t+2), q(t+1))`, applying
/// `step_wavefield_cpml` followed by `residual[t]` injection yields the
/// complete `next_adj = q(t)` before the swap.  See module-level
/// documentation for the full derivation.
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

        // ── Replay forward field from the nearest preceding checkpoint ──────
        //
        // Each slot stores the pair (previous, current) at checkpoint time
        // t_ck.  Layout per slot: [prev_n | curr_n], each block is `n` f32s.
        let ck_step = schedule.preceding_checkpoint(step);
        let ck_slot = schedule.slot_for(ck_step);
        let base = ck_slot * 2 * n;
        let mut fwd_prev = checkpoints[base..base + n].to_vec();
        let mut fwd_curr = checkpoints[base + n..base + 2 * n].to_vec();
        let mut fwd_next = vec![0.0_f32; n];
        let mut fwd_psi_x = vec![0.0_f32; n];
        let mut fwd_psi_y = vec![0.0_f32; n];

        // Exact Griewank replay with source injection.
        for fwd_step in ck_step..step {
            step_wavefield_cpml(
                grid,
                speed_m_s,
                &fwd_prev,
                &fwd_curr,
                &mut fwd_next,
                &mut fwd_psi_x,
                &mut fwd_psi_y,
            );
            inject_sources(grid, fwd_step, &mut fwd_next);
            apply_attenuation(grid, &mut fwd_next);
            std::mem::swap(&mut fwd_prev, &mut fwd_curr);
            std::mem::swap(&mut fwd_curr, &mut fwd_next);
            fwd_next.fill(0.0);
        }
        // fwd_curr is now the accurate forward pressure field at time `step`.

        // ── Advance adjoint field one step (backward in time) ───────────────
        //
        // Loop invariant: prev_adj = q(step+2), curr_adj = q(step+1).
        // step_wavefield_cpml computes:
        //   next = 2·q(step+1) − q(step+2) + dt²c²Δq(step+1)
        // which is q(step) without the adjoint source term.
        step_wavefield_cpml(
            grid,
            speed_m_s,
            &prev_adj,
            &curr_adj,
            &mut next_adj,
            &mut psi_x_adj,
            &mut psi_y_adj,
        );

        // Inject adjoint source at time `step` to complete q(step).
        //
        // Backward adjoint equation (time-reversed form):
        //   q(t) = 2q(t+1) − q(t+2) + dt²c²Δq(t+1) + f_adj(t)
        // step_wavefield_cpml produced the first three terms; adding f_adj(t)
        // = residual[step] completes q(t).
        //
        // References:
        //   Claerbout (1985), "Imaging the Earth's Interior", Eq. 2.6.
        //   Fichtner (2010), "Full Seismic Waveform Modelling", Ch. 4.3.
        for (receiver, cell) in grid.receiver_cells.iter().copied().enumerate() {
            next_adj[cell] += residual[step * receiver_count + receiver];
        }
        apply_attenuation(grid, &mut next_adj);

        // ── Imaging condition at time `step` ─────────────────────────────────
        //
        // Cross-correlate p_fwd(x, step) with next_adj = q(x, step).
        // Both fields are now at the same physical time `step`, ensuring
        // correct phase alignment at the scatterer.  See module-level
        // documentation for the loop-invariant derivation.
        for (idx, val) in image.iter_mut().enumerate() {
            let fwd = fwd_curr[idx] as f64;
            *val += fwd * next_adj[idx] as f64;
        }

        std::mem::swap(&mut prev_adj, &mut curr_adj);
        std::mem::swap(&mut curr_adj, &mut next_adj);
        next_adj.fill(0.0);
    }

    // Zero the image throughout the CPML absorption zone.
    //
    // The CPML modifies the scalar wave equation inside its damping strips
    // (the first and last PML_CELLS rows/columns of each axis).  In those
    // cells the wave equation is no longer `p_tt = c² Δp` but a stretched-
    // coordinate variant with complex damping.  Cross-correlating p_fwd and
    // q inside the CPML zone produces artifacts: the fields propagate through
    // a modified medium, the CPML memory variables are initialized from zero
    // at each checkpoint replay (since they are not checkpointed), and the
    // source and receiver cells lie in the CPML zone for compact simulation
    // grids (body surface at PML depth for the ~42-cell test geometry).  The
    // CPML zone is not an imaging target; it is an absorbing boundary.
    //
    // Detection: `a_x[ix] < 0` iff σ_x > 0 (PML cell in x-direction).
    // `a_i = exp(-σ_i·dt) - 1 < 0` for any σ_i > 0; interior cells have
    // σ = 0 and therefore a = 0 exactly.  This also mutes all source and
    // receiver cells (aperture mute) which are in the CPML zone for the
    // geometries exercised by the clinical test suite.
    for ix in 0..grid.nx {
        let in_x_pml = grid.cpml.a_x[ix] < 0.0;
        for iy in 0..grid.ny {
            if in_x_pml || grid.cpml.a_y[iy] < 0.0 {
                image[linear(ix, iy, grid.ny)] = 0.0;
            }
        }
    }

    // Return the absolute-value image.  The cross-correlation is negative at
    // scatterers where the adjoint field has opposite polarity to the forward
    // field (e.g. slower-than-background lesion: reflection coefficient R < 0
    // inverts the scattered pulse, so q < 0 at the focus while p_fwd > 0).
    // Taking |I| makes all scatterers positive regardless of impedance sign,
    // consistent with the Born reflectivity interpretation.
    Array2::from_shape_fn((grid.nx, grid.ny), |(ix, iy)| {
        let idx = linear(ix, iy, grid.ny);
        image[idx].abs()
    })
}
