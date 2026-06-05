//! Candidate parameter update kernels for line-search steps.

use kwavers_core::constants::fundamental::SOUND_SPEED_AIR;

pub(super) fn fill_candidate_speed(
    out: &mut [f64],
    current_speed: &[f64],
    background_speed: &[f64],
    body: &[bool],
    grad_speed: &[f64],
    scale: f64,
    base_step: f64,
) {
    debug_assert_eq!(out.len(), current_speed.len());
    debug_assert_eq!(out.len(), background_speed.len());
    debug_assert_eq!(out.len(), body.len());
    debug_assert_eq!(out.len(), grad_speed.len());
    for ((((dst, c), b), active), g) in out
        .iter_mut()
        .zip(current_speed.iter())
        .zip(background_speed.iter())
        .zip(body.iter())
        .zip(grad_speed.iter())
    {
        *dst = if *active {
            (c - scale * base_step * g).clamp((b - 160.0).max(SOUND_SPEED_AIR), b + 160.0)
        } else {
            *c
        };
    }
}

pub(super) fn fill_candidate_beta(
    out: &mut [f64],
    current_beta: &[f64],
    body: &[bool],
    grad_beta: &[f64],
    scale: f64,
    base_step: f64,
) {
    debug_assert_eq!(out.len(), current_beta.len());
    debug_assert_eq!(out.len(), body.len());
    debug_assert_eq!(out.len(), grad_beta.len());
    for (((dst, b), active), g) in out
        .iter_mut()
        .zip(current_beta.iter())
        .zip(body.iter())
        .zip(grad_beta.iter())
    {
        *dst = if *active {
            (b - scale * base_step * g).clamp(1.0, 12.0)
        } else {
            *b
        };
    }
}

pub(super) fn max_body_abs(values: &[f64], body: &[bool]) -> f64 {
    values
        .iter()
        .zip(body.iter())
        .filter_map(|(value, active)| active.then_some(value.abs()))
        .fold(0.0, f64::max)
}
