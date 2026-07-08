//! Stress sub-field update phase for the split-field Bérenger PML.

use super::super::kspace::{spectral_mul_x, spectral_mul_y, spectral_mul_z};
use super::super::split_field_pml::SplitFieldState;
use super::super::types::ElasticPstdMedium;
use super::SpectralOperators;
use kwavers_math::fft::{fft_3d_array_into, ifft_3d_array_into};
use kwavers_physics::acoustics::mechanics::elastic_wave::{
    fields::VelocityFields, spectral_fields::SpectralVelocityFields,
};
use leto::Array3;

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
    let [nx, ny, nz] = scratch_r.shape();

    // ── Derivatives of vx ────────────────────────────────────────────────
    fft_3d_array_into(&velocity.vx, &mut spec_in.vx);

    // ∂_x vx → txx_x (λ+2μ), tyy_x (λ), tzz_x (λ)
    spectral_mul_x(&spec_in.vx, op_x_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    for i in 0..nx {
        let (ax, bx) = (ax_s[i], bx_s[i]);
        for j in 0..ny {
            for k in 0..nz {
                let g = scratch_r[[i, j, k]];
                let lam = medium.lame_lambda[[i, j, k]];
                let mu = medium.lame_mu[[i, j, k]];
                state.txx_x[[i, j, k]] = ax * state.txx_x[[i, j, k]] + bx * (lam + 2.0 * mu) * g;
                state.tyy_x[[i, j, k]] = ax * state.tyy_x[[i, j, k]] + bx * lam * g;
                state.tzz_x[[i, j, k]] = ax * state.tzz_x[[i, j, k]] + bx * lam * g;
            }
        }
    }

    // ∂_y vx → txy_y (μ)
    spectral_mul_y(&spec_in.vx, op_y_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            let (ay, by) = (ay_s[j], by_s[j]);
            for k in 0..nz {
                let g = scratch_r[[i, j, k]];
                let mu = medium.lame_mu[[i, j, k]];
                state.txy_y[[i, j, k]] = ay * state.txy_y[[i, j, k]] + by * mu * g;
            }
        }
    }

    // ∂_z vx → txz_z (μ)
    spectral_mul_z(&spec_in.vx, op_z_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let (az, bz) = (az_s[k], bz_s[k]);
                let g = scratch_r[[i, j, k]];
                let mu = medium.lame_mu[[i, j, k]];
                state.txz_z[[i, j, k]] = az * state.txz_z[[i, j, k]] + bz * mu * g;
            }
        }
    }

    // ── Derivatives of vy ────────────────────────────────────────────────
    fft_3d_array_into(&velocity.vy, &mut spec_in.vy);

    // ∂_x vy → txy_x (μ)
    spectral_mul_x(&spec_in.vy, op_x_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    for i in 0..nx {
        let (ax, bx) = (ax_s[i], bx_s[i]);
        for j in 0..ny {
            for k in 0..nz {
                let g = scratch_r[[i, j, k]];
                let mu = medium.lame_mu[[i, j, k]];
                state.txy_x[[i, j, k]] = ax * state.txy_x[[i, j, k]] + bx * mu * g;
            }
        }
    }

    // ∂_y vy → txx_y (λ), tyy_y (λ+2μ), tzz_y (λ)
    spectral_mul_y(&spec_in.vy, op_y_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            let (ay, by) = (ay_s[j], by_s[j]);
            for k in 0..nz {
                let g = scratch_r[[i, j, k]];
                let lam = medium.lame_lambda[[i, j, k]];
                let mu = medium.lame_mu[[i, j, k]];
                state.tyy_y[[i, j, k]] = ay * state.tyy_y[[i, j, k]] + by * (lam + 2.0 * mu) * g;
                state.txx_y[[i, j, k]] = ay * state.txx_y[[i, j, k]] + by * lam * g;
                state.tzz_y[[i, j, k]] = ay * state.tzz_y[[i, j, k]] + by * lam * g;
            }
        }
    }

    // ∂_z vy → tyz_z (μ)
    spectral_mul_z(&spec_in.vy, op_z_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let (az, bz) = (az_s[k], bz_s[k]);
                let g = scratch_r[[i, j, k]];
                let mu = medium.lame_mu[[i, j, k]];
                state.tyz_z[[i, j, k]] = az * state.tyz_z[[i, j, k]] + bz * mu * g;
            }
        }
    }

    // ── Derivatives of vz ────────────────────────────────────────────────
    fft_3d_array_into(&velocity.vz, &mut spec_in.vz);

    // ∂_x vz → txz_x (μ)
    spectral_mul_x(&spec_in.vz, op_x_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    for i in 0..nx {
        let (ax, bx) = (ax_s[i], bx_s[i]);
        for j in 0..ny {
            for k in 0..nz {
                let g = scratch_r[[i, j, k]];
                let mu = medium.lame_mu[[i, j, k]];
                state.txz_x[[i, j, k]] = ax * state.txz_x[[i, j, k]] + bx * mu * g;
            }
        }
    }

    // ∂_y vz → tyz_y (μ)
    spectral_mul_y(&spec_in.vz, op_y_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            let (ay, by) = (ay_s[j], by_s[j]);
            for k in 0..nz {
                let g = scratch_r[[i, j, k]];
                let mu = medium.lame_mu[[i, j, k]];
                state.tyz_y[[i, j, k]] = ay * state.tyz_y[[i, j, k]] + by * mu * g;
            }
        }
    }

    // ∂_z vz → txx_z (λ), tyy_z (λ), tzz_z (λ+2μ)
    spectral_mul_z(&spec_in.vz, op_z_neg, kappa, &mut spec_scratch.vx);
    ifft_3d_array_into(&mut spec_scratch.vx, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let (az, bz) = (az_s[k], bz_s[k]);
                let g = scratch_r[[i, j, k]];
                let lam = medium.lame_lambda[[i, j, k]];
                let mu = medium.lame_mu[[i, j, k]];
                state.tzz_z[[i, j, k]] = az * state.tzz_z[[i, j, k]] + bz * (lam + 2.0 * mu) * g;
                state.txx_z[[i, j, k]] = az * state.txx_z[[i, j, k]] + bz * lam * g;
                state.tyy_z[[i, j, k]] = az * state.tyy_z[[i, j, k]] + bz * lam * g;
            }
        }
    }
}
