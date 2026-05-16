//! Velocity sub-field update phase for the split-field Bérenger PML.

use super::super::kspace::{spectral_mul_x, spectral_mul_y, spectral_mul_z};
use super::super::split_field_pml::SplitFieldState;
use super::super::types::ElasticPstdMedium;
use super::SpectralOperators;
use crate::math::fft::ifft_3d_array_into;
use crate::physics::acoustics::mechanics::elastic_wave::spectral_fields::SpectralStressFields;
use ndarray::{Array3, Zip};

/// PHASE 3 — Velocity sub-field updates.
///
/// For each velocity component `v_α`, multiplies spectral stress by the
/// positive-staggered derivative operator and κ, IFFTs to real space, then
/// advances the velocity sub-field with the Bérenger PML recurrence:
///   `vsf^{n+1} = α_β · vsf^n + β_β · (1/ρ) · ∂_β σ_{αβ}`
#[allow(clippy::too_many_arguments)]
pub(super) fn update_velocity_subfields(
    state: &mut SplitFieldState,
    medium: &ElasticPstdMedium,
    ops: &SpectralOperators<'_>,
    spec_stress: &SpectralStressFields,
    spec_scratch: &mut SpectralStressFields,
    scratch_r: &mut Array3<f64>,
    ax_s: &[f64],
    bx_s: &[f64],
    ay_s: &[f64],
    by_s: &[f64],
    az_s: &[f64],
    bz_s: &[f64],
) {
    let op_x_pos = ops.dkx_pos.as_slice().expect("dkx_pos contiguous");
    let op_y_pos = ops.dky_pos.as_slice().expect("dky_pos contiguous");
    let op_z_pos = ops.dkz_pos.as_slice().expect("dkz_pos contiguous");
    let kappa = &ops.kappa;

    // ── vx sub-fields: vxx ← ∂_x txx, vxy ← ∂_y txy, vxz ← ∂_z txz ────

    // ∂_x txx → vxx
    spectral_mul_x(&spec_stress.txx, op_x_pos, kappa, &mut spec_scratch.txx);
    ifft_3d_array_into(&mut spec_scratch.txx, scratch_r);
    Zip::indexed(state.vxx.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(i, _, _), vxx, &div, &rho| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *vxx = ax * *vxx + bx * div / rho;
        });

    // ∂_y txy → vxy
    spectral_mul_y(&spec_stress.txy, op_y_pos, kappa, &mut spec_scratch.txy);
    ifft_3d_array_into(&mut spec_scratch.txy, scratch_r);
    Zip::indexed(state.vxy.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, j, _), vxy, &div, &rho| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *vxy = ay * *vxy + by * div / rho;
        });

    // ∂_z txz → vxz
    spectral_mul_z(&spec_stress.txz, op_z_pos, kappa, &mut spec_scratch.txz);
    ifft_3d_array_into(&mut spec_scratch.txz, scratch_r);
    Zip::indexed(state.vxz.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, _, k), vxz, &div, &rho| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *vxz = az * *vxz + bz * div / rho;
        });

    // ── vy sub-fields: vyx ← ∂_x txy, vyy ← ∂_y tyy, vyz ← ∂_z tyz ────

    // ∂_x txy → vyx
    spectral_mul_x(&spec_stress.txy, op_x_pos, kappa, &mut spec_scratch.txy);
    ifft_3d_array_into(&mut spec_scratch.txy, scratch_r);
    Zip::indexed(state.vyx.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(i, _, _), vyx, &div, &rho| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *vyx = ax * *vyx + bx * div / rho;
        });

    // ∂_y tyy → vyy
    spectral_mul_y(&spec_stress.tyy, op_y_pos, kappa, &mut spec_scratch.tyy);
    ifft_3d_array_into(&mut spec_scratch.tyy, scratch_r);
    Zip::indexed(state.vyy.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, j, _), vyy, &div, &rho| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *vyy = ay * *vyy + by * div / rho;
        });

    // ∂_z tyz → vyz
    spectral_mul_z(&spec_stress.tyz, op_z_pos, kappa, &mut spec_scratch.tyz);
    ifft_3d_array_into(&mut spec_scratch.tyz, scratch_r);
    Zip::indexed(state.vyz.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, _, k), vyz, &div, &rho| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *vyz = az * *vyz + bz * div / rho;
        });

    // ── vz sub-fields: vzx ← ∂_x txz, vzy ← ∂_y tyz, vzz ← ∂_z tzz ────

    // ∂_x txz → vzx
    spectral_mul_x(&spec_stress.txz, op_x_pos, kappa, &mut spec_scratch.txz);
    ifft_3d_array_into(&mut spec_scratch.txz, scratch_r);
    Zip::indexed(state.vzx.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(i, _, _), vzx, &div, &rho| {
            let (ax, bx) = (ax_s[i], bx_s[i]);
            *vzx = ax * *vzx + bx * div / rho;
        });

    // ∂_y tyz → vzy
    spectral_mul_y(&spec_stress.tyz, op_y_pos, kappa, &mut spec_scratch.tyz);
    ifft_3d_array_into(&mut spec_scratch.tyz, scratch_r);
    Zip::indexed(state.vzy.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, j, _), vzy, &div, &rho| {
            let (ay, by) = (ay_s[j], by_s[j]);
            *vzy = ay * *vzy + by * div / rho;
        });

    // ∂_z tzz → vzz
    spectral_mul_z(&spec_stress.tzz, op_z_pos, kappa, &mut spec_scratch.tzz);
    ifft_3d_array_into(&mut spec_scratch.tzz, scratch_r);
    Zip::indexed(state.vzz.view_mut())
        .and(scratch_r.view())
        .and(medium.density.view())
        .for_each(|(_, _, k), vzz, &div, &rho| {
            let (az, bz) = (az_s[k], bz_s[k]);
            *vzz = az * *vzz + bz * div / rho;
        });
}
