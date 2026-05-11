//! One leapfrog step with the Bérenger split-field PML.
//!
//! # Algorithm
//!
//! Each call advances the simulation by one time step `dt` using the exact
//! discrete integrator `f^{n+1} = α·f^n + β·g` (β absorbs dt; reduces to
//! standard leapfrog at σ = 0). Four phases per step:
//!
//! 1. **Stress sub-field updates** — for each velocity component `v_α`,
//!    FFT `v_α`, multiply by the spectral derivative operator and κ, IFFT to
//!    obtain real-space `∂_β v_α`, then update the 9 stress sub-fields.
//! 2. **Stress summation** — sum stress sub-fields into total stress
//!    components, FFT each.
//! 3. **Velocity sub-field updates** — multiply spectral stress by the
//!    derivative operator and κ, IFFT to obtain `∂_β σ_{αβ}`, update the 9
//!    velocity sub-fields.
//! 4. **Velocity summation** — sum velocity sub-fields into the real-space
//!    `VelocityFields` buffers used by the caller.
//!
//! Spectral derivatives use the staggered-grid operators: `dkα_neg` for
//! stress updates (stress grid) and `dkα_pos` for velocity updates (velocity
//! grid), consistent with the non-split leapfrog path.
//!
//! `spectral_stress` and `spectral_stress_next` are used as scratch FFT
//! buffers in this path; their contents after return are undefined.

use super::split_field_pml::{ElasticSplitFieldPml, SplitFieldState};
use super::types::ElasticPstdMedium;
use crate::math::fft::{fft_3d_array_into, ifft_3d_array_into};
use crate::physics::acoustics::mechanics::elastic_wave::{
    fields::VelocityFields,
    spectral_fields::{SpectralStressFields, SpectralVelocityFields},
};
use ndarray::{Array3, Zip};
use num_complex::Complex;

/// Advance by one split-field PML time step.
///
/// All `spectral_*` parameters are used as scratch; their contents on entry
/// are ignored and their contents on exit are undefined. The caller must not
/// rely on the spectral state after this call when using the split-field path.
#[allow(clippy::too_many_arguments)]
pub(super) fn propagate_split_field_step(
    velocity: &mut VelocityFields,
    spectral_stress: &mut SpectralStressFields,
    spectral_stress_next: &mut SpectralStressFields,
    spectral_velocity_in: &mut SpectralVelocityFields,
    spectral_velocity_next: &mut SpectralVelocityFields,
    pml: &ElasticSplitFieldPml,
    state: &mut SplitFieldState,
    medium: &ElasticPstdMedium,
    dkx_neg: &Array3<Complex<f64>>,
    dky_neg: &Array3<Complex<f64>>,
    dkz_neg: &Array3<Complex<f64>>,
    dkx_pos: &Array3<Complex<f64>>,
    dky_pos: &Array3<Complex<f64>>,
    dkz_pos: &Array3<Complex<f64>>,
    kappa: &Array3<f64>,
    scratch_r: &mut Array3<f64>,
) {
    let (alpha_x, beta_x) = pml.x_coeffs();
    let (alpha_y, beta_y) = pml.y_coeffs();
    let (alpha_z, beta_z) = pml.z_coeffs();
    let ax_s = alpha_x.as_slice().expect("alpha_x contiguous");
    let bx_s = beta_x.as_slice().expect("beta_x contiguous");
    let ay_s = alpha_y.as_slice().expect("alpha_y contiguous");
    let by_s = beta_y.as_slice().expect("beta_y contiguous");
    let az_s = alpha_z.as_slice().expect("alpha_z contiguous");
    let bz_s = beta_z.as_slice().expect("beta_z contiguous");
    let op_x_neg = dkx_neg.as_slice().expect("dkx_neg contiguous");
    let op_y_neg = dky_neg.as_slice().expect("dky_neg contiguous");
    let op_z_neg = dkz_neg.as_slice().expect("dkz_neg contiguous");
    let op_x_pos = dkx_pos.as_slice().expect("dkx_pos contiguous");
    let op_y_pos = dky_pos.as_slice().expect("dky_pos contiguous");
    let op_z_pos = dkz_pos.as_slice().expect("dkz_pos contiguous");

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 1 — Stress sub-field updates.
    //
    // Each sub-field sf_{αβ} driven by ∂_β v_α:
    //   sf^{n+1} = α_β · sf^n + β_β · C · ∂_β v_α
    //
    // vspec (spectral_velocity_in.{vx,vy,vz}) holds FFT(v_α) for current
    // velocity component. cspec (spectral_velocity_next.vx) is the product
    // vspec × op × kappa, IFFT'd into scratch_r for each derivative.
    // ═══════════════════════════════════════════════════════════════════════

    // ── Derivatives of vx ────────────────────────────────────────────────
    fft_3d_array_into(&velocity.vx, &mut spectral_velocity_in.vx);

    // ∂_x vx → txx_x (λ+2μ), tyy_x (λ), tzz_x (λ)
    spectral_mul_x(
        &spectral_velocity_in.vx,
        op_x_neg,
        kappa,
        &mut spectral_velocity_next.vx,
    );
    ifft_3d_array_into(&mut spectral_velocity_next.vx, scratch_r);
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
    spectral_mul_y(
        &spectral_velocity_in.vx,
        op_y_neg,
        kappa,
        &mut spectral_velocity_next.vx,
    );
    ifft_3d_array_into(&mut spectral_velocity_next.vx, scratch_r);
    Zip::indexed(state.txy_y.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(_, j, _), txy_y, &g, &mu| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *txy_y = ay * *txy_y + by * mu * g;
        });

    // ∂_z vx → txz_z (μ)
    spectral_mul_z(
        &spectral_velocity_in.vx,
        op_z_neg,
        kappa,
        &mut spectral_velocity_next.vx,
    );
    ifft_3d_array_into(&mut spectral_velocity_next.vx, scratch_r);
    Zip::indexed(state.txz_z.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(_, _, k), txz_z, &g, &mu| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *txz_z = az * *txz_z + bz * mu * g;
        });

    // ── Derivatives of vy ────────────────────────────────────────────────
    fft_3d_array_into(&velocity.vy, &mut spectral_velocity_in.vy);

    // ∂_x vy → txy_x (μ)
    spectral_mul_x(
        &spectral_velocity_in.vy,
        op_x_neg,
        kappa,
        &mut spectral_velocity_next.vx,
    );
    ifft_3d_array_into(&mut spectral_velocity_next.vx, scratch_r);
    Zip::indexed(state.txy_x.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(i, _, _), txy_x, &g, &mu| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *txy_x = ax * *txy_x + bx * mu * g;
        });

    // ∂_y vy → txx_y (λ), tyy_y (λ+2μ), tzz_y (λ)
    spectral_mul_y(
        &spectral_velocity_in.vy,
        op_y_neg,
        kappa,
        &mut spectral_velocity_next.vx,
    );
    ifft_3d_array_into(&mut spectral_velocity_next.vx, scratch_r);
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
    spectral_mul_z(
        &spectral_velocity_in.vy,
        op_z_neg,
        kappa,
        &mut spectral_velocity_next.vx,
    );
    ifft_3d_array_into(&mut spectral_velocity_next.vx, scratch_r);
    Zip::indexed(state.tyz_z.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(_, _, k), tyz_z, &g, &mu| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *tyz_z = az * *tyz_z + bz * mu * g;
        });

    // ── Derivatives of vz ────────────────────────────────────────────────
    fft_3d_array_into(&velocity.vz, &mut spectral_velocity_in.vz);

    // ∂_x vz → txz_x (μ)
    spectral_mul_x(
        &spectral_velocity_in.vz,
        op_x_neg,
        kappa,
        &mut spectral_velocity_next.vx,
    );
    ifft_3d_array_into(&mut spectral_velocity_next.vx, scratch_r);
    Zip::indexed(state.txz_x.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(i, _, _), txz_x, &g, &mu| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *txz_x = ax * *txz_x + bx * mu * g;
        });

    // ∂_y vz → tyz_y (μ)
    spectral_mul_y(
        &spectral_velocity_in.vz,
        op_y_neg,
        kappa,
        &mut spectral_velocity_next.vx,
    );
    ifft_3d_array_into(&mut spectral_velocity_next.vx, scratch_r);
    Zip::indexed(state.tyz_y.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|(_, j, _), tyz_y, &g, &mu| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *tyz_y = ay * *tyz_y + by * mu * g;
        });

    // ∂_z vz → txx_z (λ), tyy_z (λ), tzz_z (λ+2μ)
    spectral_mul_z(
        &spectral_velocity_in.vz,
        op_z_neg,
        kappa,
        &mut spectral_velocity_next.vx,
    );
    ifft_3d_array_into(&mut spectral_velocity_next.vx, scratch_r);
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

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2 — Sum stress sub-fields and FFT each total component.
    //
    // spectral_stress.{txx,...} are used as scratch FFT output buffers.
    // ═══════════════════════════════════════════════════════════════════════

    // txx = txx_x + txx_y + txx_z
    Zip::from(scratch_r.view_mut())
        .and(&state.txx_x)
        .and(&state.txx_y)
        .and(&state.txx_z)
        .for_each(|out, &a, &b, &c| *out = a + b + c);
    fft_3d_array_into(scratch_r, &mut spectral_stress.txx);

    // tyy = tyy_x + tyy_y + tyy_z
    Zip::from(scratch_r.view_mut())
        .and(&state.tyy_x)
        .and(&state.tyy_y)
        .and(&state.tyy_z)
        .for_each(|out, &a, &b, &c| *out = a + b + c);
    fft_3d_array_into(scratch_r, &mut spectral_stress.tyy);

    // tzz = tzz_x + tzz_y + tzz_z
    Zip::from(scratch_r.view_mut())
        .and(&state.tzz_x)
        .and(&state.tzz_y)
        .and(&state.tzz_z)
        .for_each(|out, &a, &b, &c| *out = a + b + c);
    fft_3d_array_into(scratch_r, &mut spectral_stress.tzz);

    // txy = txy_x + txy_y
    Zip::from(scratch_r.view_mut())
        .and(&state.txy_x)
        .and(&state.txy_y)
        .for_each(|out, &a, &b| *out = a + b);
    fft_3d_array_into(scratch_r, &mut spectral_stress.txy);

    // txz = txz_x + txz_z
    Zip::from(scratch_r.view_mut())
        .and(&state.txz_x)
        .and(&state.txz_z)
        .for_each(|out, &a, &b| *out = a + b);
    fft_3d_array_into(scratch_r, &mut spectral_stress.txz);

    // tyz = tyz_y + tyz_z
    Zip::from(scratch_r.view_mut())
        .and(&state.tyz_y)
        .and(&state.tyz_z)
        .for_each(|out, &a, &b| *out = a + b);
    fft_3d_array_into(scratch_r, &mut spectral_stress.tyz);

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 3 — Velocity sub-field updates.
    //
    // Each sub-field vsf_{α,β} driven by ∂_β σ_{αβ}:
    //   vsf^{n+1} = α_β · vsf^n + β_β · (1/ρ) · ∂_β σ_{αβ}
    //
    // spectral_stress_next.{txx,...} used as scratch cspec for derivative
    // products; IFFT'd into scratch_r per sub-field group.
    // ═══════════════════════════════════════════════════════════════════════

    // ── vx sub-fields: vxx ← ∂_x txx, vxy ← ∂_y txy, vxz ← ∂_z txz ────

    // ∂_x txx → vxx
    spectral_mul_x(
        &spectral_stress.txx,
        op_x_pos,
        kappa,
        &mut spectral_stress_next.txx,
    );
    ifft_3d_array_into(&mut spectral_stress_next.txx, scratch_r);
    Zip::indexed(state.vxx.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(i, _, _), vxx, &div, &rho| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *vxx = ax * *vxx + bx * div / rho;
        });

    // ∂_y txy → vxy
    spectral_mul_y(
        &spectral_stress.txy,
        op_y_pos,
        kappa,
        &mut spectral_stress_next.txy,
    );
    ifft_3d_array_into(&mut spectral_stress_next.txy, scratch_r);
    Zip::indexed(state.vxy.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, j, _), vxy, &div, &rho| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *vxy = ay * *vxy + by * div / rho;
        });

    // ∂_z txz → vxz
    spectral_mul_z(
        &spectral_stress.txz,
        op_z_pos,
        kappa,
        &mut spectral_stress_next.txz,
    );
    ifft_3d_array_into(&mut spectral_stress_next.txz, scratch_r);
    Zip::indexed(state.vxz.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, _, k), vxz, &div, &rho| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *vxz = az * *vxz + bz * div / rho;
        });

    // ── vy sub-fields: vyx ← ∂_x txy, vyy ← ∂_y tyy, vyz ← ∂_z tyz ────

    // ∂_x txy → vyx
    spectral_mul_x(
        &spectral_stress.txy,
        op_x_pos,
        kappa,
        &mut spectral_stress_next.txy,
    );
    ifft_3d_array_into(&mut spectral_stress_next.txy, scratch_r);
    Zip::indexed(state.vyx.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(i, _, _), vyx, &div, &rho| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *vyx = ax * *vyx + bx * div / rho;
        });

    // ∂_y tyy → vyy
    spectral_mul_y(
        &spectral_stress.tyy,
        op_y_pos,
        kappa,
        &mut spectral_stress_next.tyy,
    );
    ifft_3d_array_into(&mut spectral_stress_next.tyy, scratch_r);
    Zip::indexed(state.vyy.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, j, _), vyy, &div, &rho| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *vyy = ay * *vyy + by * div / rho;
        });

    // ∂_z tyz → vyz
    spectral_mul_z(
        &spectral_stress.tyz,
        op_z_pos,
        kappa,
        &mut spectral_stress_next.tyz,
    );
    ifft_3d_array_into(&mut spectral_stress_next.tyz, scratch_r);
    Zip::indexed(state.vyz.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, _, k), vyz, &div, &rho| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *vyz = az * *vyz + bz * div / rho;
        });

    // ── vz sub-fields: vzx ← ∂_x txz, vzy ← ∂_y tyz, vzz ← ∂_z tzz ────

    // ∂_x txz → vzx
    spectral_mul_x(
        &spectral_stress.txz,
        op_x_pos,
        kappa,
        &mut spectral_stress_next.txz,
    );
    ifft_3d_array_into(&mut spectral_stress_next.txz, scratch_r);
    Zip::indexed(state.vzx.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(i, _, _), vzx, &div, &rho| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *vzx = ax * *vzx + bx * div / rho;
        });

    // ∂_y tyz → vzy
    spectral_mul_y(
        &spectral_stress.tyz,
        op_y_pos,
        kappa,
        &mut spectral_stress_next.tyz,
    );
    ifft_3d_array_into(&mut spectral_stress_next.tyz, scratch_r);
    Zip::indexed(state.vzy.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, j, _), vzy, &div, &rho| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *vzy = ay * *vzy + by * div / rho;
        });

    // ∂_z tzz → vzz
    spectral_mul_z(
        &spectral_stress.tzz,
        op_z_pos,
        kappa,
        &mut spectral_stress_next.tzz,
    );
    ifft_3d_array_into(&mut spectral_stress_next.tzz, scratch_r);
    Zip::indexed(state.vzz.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, _, k), vzz, &div, &rho| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *vzz = az * *vzz + bz * div / rho;
        });

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 4 — Sum velocity sub-fields → real-space total velocity.
    // ═══════════════════════════════════════════════════════════════════════
    Zip::from(velocity.vx.view_mut())
        .and(&state.vxx)
        .and(&state.vxy)
        .and(&state.vxz)
        .for_each(|out, &a, &b, &c| *out = a + b + c);
    Zip::from(velocity.vy.view_mut())
        .and(&state.vyx)
        .and(&state.vyy)
        .and(&state.vyz)
        .for_each(|out, &a, &b, &c| *out = a + b + c);
    Zip::from(velocity.vz.view_mut())
        .and(&state.vzx)
        .and(&state.vzy)
        .and(&state.vzz)
        .for_each(|out, &a, &b, &c| *out = a + b + c);
}

// ─── Spectral derivative helpers ─────────────────────────────────────────────

/// Compute `output[i,j,k] = input[i,j,k] · op_x[i] · kappa[i,j,k]`.
///
/// `op_x` is a contiguous slice of length `nx` from a `(nx, 1, 1)` operator
/// array. The x-axis index `i` selects the per-wavenumber multiplier.
#[inline]
fn spectral_mul_x(
    input: &Array3<Complex<f64>>,
    op_x: &[Complex<f64>],
    kappa: &Array3<f64>,
    output: &mut Array3<Complex<f64>>,
) {
    Zip::indexed(output.view_mut())
        .and(input.view())
        .and(kappa.view())
        .for_each(|(i, _, _), out, inp, kap| {
            *out = *inp * op_x[i] * kap;
        });
}

/// Compute `output[i,j,k] = input[i,j,k] · op_y[j] · kappa[i,j,k]`.
///
/// `op_y` is a contiguous slice of length `ny` from a `(ny, 1, 1)` operator
/// array indexed by the y-axis position `j`.
#[inline]
fn spectral_mul_y(
    input: &Array3<Complex<f64>>,
    op_y: &[Complex<f64>],
    kappa: &Array3<f64>,
    output: &mut Array3<Complex<f64>>,
) {
    Zip::indexed(output.view_mut())
        .and(input.view())
        .and(kappa.view())
        .for_each(|(_, j, _), out, inp, kap| {
            *out = *inp * op_y[j] * kap;
        });
}

/// Compute `output[i,j,k] = input[i,j,k] · op_z[k] · kappa[i,j,k]`.
///
/// `op_z` is a contiguous slice of length `nz` from a `(nz, 1, 1)` operator
/// array indexed by the z-axis position `k`.
#[inline]
fn spectral_mul_z(
    input: &Array3<Complex<f64>>,
    op_z: &[Complex<f64>],
    kappa: &Array3<f64>,
    output: &mut Array3<Complex<f64>>,
) {
    Zip::indexed(output.view_mut())
        .and(input.view())
        .and(kappa.view())
        .for_each(|(_, _, k), out, inp, kap| {
            *out = *inp * op_z[k] * kap;
        });
}
