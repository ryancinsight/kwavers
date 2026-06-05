//! Stress sub-field update phase for the split-field Bérenger PML.

use super::super::kspace::{spectral_mul_x, spectral_mul_y, spectral_mul_z};
use super::super::split_field_pml::SplitFieldState;
use super::super::types::ElasticPstdMedium;
use super::SpectralOperators;
use kwavers_math::fft::{fft_3d_array_into, ifft_3d_array_into};
use kwavers_physics::acoustics::mechanics::elastic_wave::{
    fields::VelocityFields, spectral_fields::SpectralVelocityFields,
};
use ndarray::{Array3, Zip};

/// PHASE 1 — Stress sub-field updates.
///
/// For each velocity component `v_α`, FFT it, compute each spectral derivative
/// `∂_β v_α` via `spectral_mul_*`, IFFT to real space, then advance the stress
/// sub-field with the Bérenger PML recurrence:
///   `sf^{n+1} = α_β · sf^n + β_β · C · ∂_β v_α`
#[allow(clippy::too_many_arguments)]
pub(super) fn update_stress_subfields(
    velocity: &VelocityFields,
    state: &mut SplitFieldState,
    medium: &ElasticPstdMedium,
    ops: &SpectralOperators<'_>,
    spec_in: &mut SpectralVelocityFields,
    spec_scratch: &mut SpectralVelocityFields,
    scratch_r: &mut Array3<f64>,
    ax_s: &[f64],
    bx_s: &[f64],
    ay_s: &[f64],
    by_s: &[f64],
    az_s: &[f64],
    bz_s: &[f64],
) {
    let op_x_neg = ops.dkx_neg.as_slice().expect("dkx_neg contiguous");
    let op_y_neg = ops.dky_neg.as_slice().expect("dky_neg contiguous");
    let op_z_neg = ops.dkz_neg.as_slice().expect("dkz_neg contiguous");
    let kappa = &ops.kappa;

    // ── Derivatives of vx ────────────────────────────────────────────────
    fft_3d_array_into(&velocity.vx, &mut spec_in.vx);

    // ∂_x vx → txx_x (λ+2μ), tyy_x (λ), tzz_x (λ)
    spectral_mul_x(&spec_in.vx, op_x_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    Zip::indexed(state.txx_x.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_lambda.view())
        .and(medium.lame_mu.view())
        .for_each(|(i, _, _), txx_x, &g, &lam, &mu| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *txx_x = ax * *txx_x + bx * (lam + 2.0 * mu) * g;
        });
    Zip::indexed(state.tyy_x.view_mut())
        .and(state.tzz_x.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_lambda.view())
        .for_each(|(i, _, _), tyy_x, tzz_x, &g, &lam| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *tyy_x = ax * *tyy_x + bx * lam * g;
            *tzz_x = ax * *tzz_x + bx * lam * g;
        });

    // ∂_y vx → txy_y (μ)
    spectral_mul_y(&spec_in.vx, op_y_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    Zip::indexed(state.txy_y.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(_, j, _), txy_y, &g, &mu| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *txy_y = ay * *txy_y + by * mu * g;
        });

    // ∂_z vx → txz_z (μ)
    spectral_mul_z(&spec_in.vx, op_z_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    Zip::indexed(state.txz_z.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(_, _, k), txz_z, &g, &mu| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *txz_z = az * *txz_z + bz * mu * g;
        });

    // ── Derivatives of vy ────────────────────────────────────────────────
    fft_3d_array_into(&velocity.vy, &mut spec_in.vy);

    // ∂_x vy → txy_x (μ)
    spectral_mul_x(&spec_in.vy, op_x_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    Zip::indexed(state.txy_x.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(i, _, _), txy_x, &g, &mu| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *txy_x = ax * *txy_x + bx * mu * g;
        });

    // ∂_y vy → txx_y (λ), tyy_y (λ+2μ), tzz_y (λ)
    spectral_mul_y(&spec_in.vy, op_y_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    Zip::indexed(state.tyy_y.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_lambda.view())
        .and(medium.lame_mu.view())
        .for_each(|(_, j, _), tyy_y, &g, &lam, &mu| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *tyy_y = ay * *tyy_y + by * (lam + 2.0 * mu) * g;
        });
    Zip::indexed(state.txx_y.view_mut())
        .and(state.tzz_y.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_lambda.view())
        .for_each(|(_, j, _), txx_y, tzz_y, &g, &lam| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *txx_y = ay * *txx_y + by * lam * g;
            *tzz_y = ay * *tzz_y + by * lam * g;
        });

    // ∂_z vy → tyz_z (μ)
    spectral_mul_z(&spec_in.vy, op_z_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    Zip::indexed(state.tyz_z.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(_, _, k), tyz_z, &g, &mu| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *tyz_z = az * *tyz_z + bz * mu * g;
        });

    // ── Derivatives of vz ────────────────────────────────────────────────
    fft_3d_array_into(&velocity.vz, &mut spec_in.vz);

    // ∂_x vz → txz_x (μ)
    spectral_mul_x(&spec_in.vz, op_x_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    Zip::indexed(state.txz_x.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(i, _, _), txz_x, &g, &mu| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *txz_x = ax * *txz_x + bx * mu * g;
        });

    // ∂_y vz → tyz_y (μ)
    spectral_mul_y(&spec_in.vz, op_y_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    Zip::indexed(state.tyz_y.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(_, j, _), tyz_y, &g, &mu| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *tyz_y = ay * *tyz_y + by * mu * g;
        });

    // ∂_z vz → txx_z (λ), tyy_z (λ), tzz_z (λ+2μ)
    spectral_mul_z(&spec_in.vz, op_z_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    Zip::indexed(state.tzz_z.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_lambda.view())
        .and(medium.lame_mu.view())
        .for_each(|(_, _, k), tzz_z, &g, &lam, &mu| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *tzz_z = az * *tzz_z + bz * (lam + 2.0 * mu) * g;
        });
    Zip::indexed(state.txx_z.view_mut())
        .and(state.tyy_z.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_lambda.view())
        .for_each(|(_, _, k), txx_z, tyy_z, &g, &lam| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *txx_z = az * *txx_z + bz * lam * g;
            *tyy_z = az * *tyy_z + bz * lam * g;
        });
}
