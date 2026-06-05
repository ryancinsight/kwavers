//! Real-space-coefficient leapfrog step (no PML) for the elastic PSTD solver.
//!
//! # Why this exists
//!
//! Spatial derivatives are computed in k-space (`i·k·κ`) but the Lamé moduli
//! `λ, μ` and density `ρ` are **spatially varying real-space fields**. A
//! spectral field may only be multiplied by a *constant* in the spectral
//! domain; multiplying it by a space-varying coefficient pointwise in k-space
//! is a convolution error and produces the wrong wave speed in heterogeneous
//! media (a wave at the reference speed is dragged toward the spatial average
//! of the surrounding speeds).
//!
//! The correct k-space pseudospectral scheme — used by the acoustic PSTD
//! propagator and the elastic split-field PML path — computes each derivative
//! `∂_β f = IFFT(i·k_β·κ·FFT(f))`, transforms it to **real space**, and only
//! then multiplies by the local coefficient. For a homogeneous medium the
//! FFT round-trip of a constant-coefficient linear operator is the identity,
//! so this reproduces the pure-spectral update bit-for-bit; for a
//! heterogeneous medium it is the physically correct update.
//!
//! Stress and velocity are held in real space between steps. Each step:
//!   1. FFT velocity; for each `∂_β v_α` derivative IFFT to real space and
//!      accumulate `σ += dt · C : ε` with real-space `λ, μ`.
//!   2. FFT the updated stress; for each `∂_β σ_{αβ}` derivative IFFT to real
//!      space and accumulate `v += (dt/ρ) · ∇·σ` with real-space `ρ`.

use super::kspace::{spectral_mul_x, spectral_mul_y, spectral_mul_z};
use super::split_field_step::SpectralOperators;
use super::types::ElasticPstdMedium;
use kwavers_math::fft::{fft_3d_array_into, ifft_3d_array_into};
use kwavers_physics::acoustics::mechanics::elastic_wave::{
    fields::{StressFields, VelocityFields},
    spectral_fields::{SpectralStressFields, SpectralVelocityFields},
};
use ndarray::{Array3, Zip};
use num_complex::Complex;

/// Axis-specific `out ← op·κ·field_k` spectral derivative multiplier.
type SpectralMul =
    fn(&Array3<Complex<f64>>, &[Complex<f64>], &Array3<f64>, &mut Array3<Complex<f64>>);

/// Advance `(stress, velocity)` one leapfrog step with real-space coefficients.
#[allow(clippy::too_many_arguments)]
pub(super) fn propagate_leapfrog_step(
    velocity: &mut VelocityFields,
    stress: &mut StressFields,
    medium: &ElasticPstdMedium,
    ops: &SpectralOperators<'_>,
    spec_vel: &mut SpectralVelocityFields,
    spec_stress: &mut SpectralStressFields,
    spec_scratch: &mut SpectralStressFields,
    scratch_r: &mut Array3<f64>,
    dt: f64,
) {
    let op_x_pos = ops.dkx_pos.as_slice().expect("dkx_pos contiguous");
    let op_y_pos = ops.dky_pos.as_slice().expect("dky_pos contiguous");
    let op_z_pos = ops.dkz_pos.as_slice().expect("dkz_pos contiguous");
    let kappa = ops.kappa;

    // ── PHASE 1 — stress update (real-space λ, μ) ───────────────────────────
    fft_3d_array_into(&velocity.vx, &mut spec_vel.vx);
    fft_3d_array_into(&velocity.vy, &mut spec_vel.vy);
    fft_3d_array_into(&velocity.vz, &mut spec_vel.vz);
    accumulate_stress(
        spec_vel,
        stress,
        medium,
        ops,
        &mut spec_scratch.txx,
        scratch_r,
        dt,
    );

    // ── PHASE 2 — velocity update (real-space 1/ρ) ──────────────────────────
    fft_3d_array_into(&stress.txx, &mut spec_stress.txx);
    fft_3d_array_into(&stress.tyy, &mut spec_stress.tyy);
    fft_3d_array_into(&stress.tzz, &mut spec_stress.tzz);
    fft_3d_array_into(&stress.txy, &mut spec_stress.txy);
    fft_3d_array_into(&stress.txz, &mut spec_stress.txz);
    fft_3d_array_into(&stress.tyz, &mut spec_stress.tyz);
    let buf = &mut spec_scratch.txx;

    // v_x += (dt/ρ)·(∂_x τ_xx + ∂_y τ_xy + ∂_z τ_xz).
    vel(
        spectral_mul_x,
        &spec_stress.txx,
        op_x_pos,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut velocity.vx,
    );
    vel(
        spectral_mul_y,
        &spec_stress.txy,
        op_y_pos,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut velocity.vx,
    );
    vel(
        spectral_mul_z,
        &spec_stress.txz,
        op_z_pos,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut velocity.vx,
    );
    // v_y += (dt/ρ)·(∂_x τ_xy + ∂_y τ_yy + ∂_z τ_yz).
    vel(
        spectral_mul_x,
        &spec_stress.txy,
        op_x_pos,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut velocity.vy,
    );
    vel(
        spectral_mul_y,
        &spec_stress.tyy,
        op_y_pos,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut velocity.vy,
    );
    vel(
        spectral_mul_z,
        &spec_stress.tyz,
        op_z_pos,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut velocity.vy,
    );
    // v_z += (dt/ρ)·(∂_x τ_xz + ∂_y τ_yz + ∂_z τ_zz).
    vel(
        spectral_mul_x,
        &spec_stress.txz,
        op_x_pos,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut velocity.vz,
    );
    vel(
        spectral_mul_y,
        &spec_stress.tyz,
        op_y_pos,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut velocity.vz,
    );
    vel(
        spectral_mul_z,
        &spec_stress.tzz,
        op_z_pos,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut velocity.vz,
    );
}

/// Accumulate `σ += dt · C : ε(v)` from a spectral velocity field, applying
/// `λ, μ` in real space. Shared by the leapfrog step and the IVP seed.
fn accumulate_stress(
    spec_vel: &SpectralVelocityFields,
    stress: &mut StressFields,
    medium: &ElasticPstdMedium,
    ops: &SpectralOperators<'_>,
    buf: &mut Array3<Complex<f64>>,
    scratch_r: &mut Array3<f64>,
    dt: f64,
) {
    let op_x_neg = ops.dkx_neg.as_slice().expect("dkx_neg contiguous");
    let op_y_neg = ops.dky_neg.as_slice().expect("dky_neg contiguous");
    let op_z_neg = ops.dkz_neg.as_slice().expect("dkz_neg contiguous");
    let kappa = ops.kappa;

    spectral_mul_x(&spec_vel.vx, op_x_neg, kappa, buf);
    ifft_3d_array_into(buf, scratch_r);
    accumulate_normal(stress, scratch_r, medium, dt, NormalAxis::X);
    spectral_mul_y(&spec_vel.vy, op_y_neg, kappa, buf);
    ifft_3d_array_into(buf, scratch_r);
    accumulate_normal(stress, scratch_r, medium, dt, NormalAxis::Y);
    spectral_mul_z(&spec_vel.vz, op_z_neg, kappa, buf);
    ifft_3d_array_into(buf, scratch_r);
    accumulate_normal(stress, scratch_r, medium, dt, NormalAxis::Z);

    shear(
        spectral_mul_y,
        &spec_vel.vx,
        op_y_neg,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut stress.txy,
    );
    shear(
        spectral_mul_x,
        &spec_vel.vy,
        op_x_neg,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut stress.txy,
    );
    shear(
        spectral_mul_z,
        &spec_vel.vx,
        op_z_neg,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut stress.txz,
    );
    shear(
        spectral_mul_x,
        &spec_vel.vz,
        op_x_neg,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut stress.txz,
    );
    shear(
        spectral_mul_z,
        &spec_vel.vy,
        op_z_neg,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut stress.tyz,
    );
    shear(
        spectral_mul_y,
        &spec_vel.vz,
        op_y_neg,
        kappa,
        buf,
        scratch_r,
        dt,
        medium,
        &mut stress.tyz,
    );
}

/// Seed the initial stress for an initial-value problem with displacement `u0`
/// along `axis` (0=x, 1=y, 2=z), other components and initial velocity zero.
/// The initial stress `σ = λ(∇·u)I + μ(∇u + ∇uᵀ)` is the `dt = 1` stress
/// increment driven by `u0` in place of velocity, so it reuses
/// [`accumulate_stress`] for an exactly grid-consistent discretisation.
#[allow(clippy::too_many_arguments)]
pub(super) fn seed_stress_from_displacement(
    stress: &mut StressFields,
    u0: &Array3<f64>,
    axis: usize,
    medium: &ElasticPstdMedium,
    ops: &SpectralOperators<'_>,
    spec_vel: &mut SpectralVelocityFields,
    buf: &mut Array3<Complex<f64>>,
    scratch_r: &mut Array3<f64>,
) {
    stress.reset();
    spec_vel.vx.fill(Complex::new(0.0, 0.0));
    spec_vel.vy.fill(Complex::new(0.0, 0.0));
    spec_vel.vz.fill(Complex::new(0.0, 0.0));
    let slot = match axis {
        0 => &mut spec_vel.vx,
        1 => &mut spec_vel.vy,
        _ => &mut spec_vel.vz,
    };
    fft_3d_array_into(u0, slot);
    accumulate_stress(spec_vel, stress, medium, ops, buf, scratch_r, 1.0);
}

enum NormalAxis {
    X,
    Y,
    Z,
}

/// `σ_αα += dt·(λ+2μ)·g`, `σ_ββ += dt·λ·g` for the two transverse components.
fn accumulate_normal(
    stress: &mut StressFields,
    g: &Array3<f64>,
    medium: &ElasticPstdMedium,
    dt: f64,
    axis: NormalAxis,
) {
    Zip::from(stress.txx.view_mut())
        .and(stress.tyy.view_mut())
        .and(stress.tzz.view_mut())
        .and(g.view())
        .and(medium.lame_lambda.view())
        .and(medium.lame_mu.view())
        .for_each(|txx, tyy, tzz, &g, &lam, &mu| {
            let lam_dt = dt * lam * g;
            let p2 = dt * 2.0 * mu * g;
            *txx += lam_dt;
            *tyy += lam_dt;
            *tzz += lam_dt;
            match axis {
                NormalAxis::X => *txx += p2,
                NormalAxis::Y => *tyy += p2,
                NormalAxis::Z => *tzz += p2,
            }
        });
}

/// `τ += dt · μ(x) · ∂_β v_α` (derivative in k-space, μ in real space).
#[allow(clippy::too_many_arguments)]
fn shear(
    mul: SpectralMul,
    field_k: &Array3<Complex<f64>>,
    op: &[Complex<f64>],
    kappa: &Array3<f64>,
    buf: &mut Array3<Complex<f64>>,
    scratch_r: &mut Array3<f64>,
    dt: f64,
    medium: &ElasticPstdMedium,
    tau: &mut Array3<f64>,
) {
    mul(field_k, op, kappa, buf);
    ifft_3d_array_into(buf, scratch_r);
    Zip::from(tau.view_mut())
        .and(scratch_r.view())
        .and(medium.lame_mu.view())
        .for_each(|t, &g, &mu| *t += dt * mu * g);
}

/// `v += (dt/ρ(x)) · ∂_β τ_{αβ}`; preserves `v` where `ρ ≤ 0`.
#[allow(clippy::too_many_arguments)]
fn vel(
    mul: SpectralMul,
    field_k: &Array3<Complex<f64>>,
    op: &[Complex<f64>],
    kappa: &Array3<f64>,
    buf: &mut Array3<Complex<f64>>,
    scratch_r: &mut Array3<f64>,
    dt: f64,
    medium: &ElasticPstdMedium,
    v: &mut Array3<f64>,
) {
    mul(field_k, op, kappa, buf);
    ifft_3d_array_into(buf, scratch_r);
    Zip::from(v.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|vv, &g, &rho| {
            if rho > 0.0 {
                *vv += (dt / rho) * g;
            }
        });
}
