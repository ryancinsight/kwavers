//! Velocity sub-field update phase for the split-field Bérenger PML.

use super::super::kspace::{spectral_mul_x, spectral_mul_y, spectral_mul_z};
use super::super::split_field_pml::SplitFieldState;
use super::super::types::ElasticPstdMedium;
use super::SpectralOperators;
use kwavers_math::fft::ifft_3d_array_into;
use kwavers_physics::acoustics::mechanics::elastic_wave::spectral_fields::SpectralStressFields;
use leto::Array3;

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
    let [nx, ny, nz] = scratch_r.shape();

    // ── vx sub-fields: vxx ← ∂_x txx, vxy ← ∂_y txy, vxz ← ∂_z txz ────

    // ∂_x txx → vxx
    spectral_mul_x(&spec_stress.txx, op_x_pos, kappa, &mut spec_scratch.txx);
    ifft_3d_array_into(&mut spec_scratch.txx, scratch_r);
    for i in 0..nx {
        let (ax, bx) = (ax_s[i], bx_s[i]);
        for j in 0..ny {
            for k in 0..nz {
                state.vxx[[i, j, k]] = ax * state.vxx[[i, j, k]]
                    + bx * scratch_r[[i, j, k]] / medium.density[[i, j, k]];
            }
        }
    }

    // ∂_y txy → vxy
    spectral_mul_y(&spec_stress.txy, op_y_pos, kappa, &mut spec_scratch.txy);
    ifft_3d_array_into(&mut spec_scratch.txy, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            let (ay, by) = (ay_s[j], by_s[j]);
            for k in 0..nz {
                state.vxy[[i, j, k]] = ay * state.vxy[[i, j, k]]
                    + by * scratch_r[[i, j, k]] / medium.density[[i, j, k]];
            }
        }
    }

    // ∂_z txz → vxz
    spectral_mul_z(&spec_stress.txz, op_z_pos, kappa, &mut spec_scratch.txz);
    ifft_3d_array_into(&mut spec_scratch.txz, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let (az, bz) = (az_s[k], bz_s[k]);
                state.vxz[[i, j, k]] = az * state.vxz[[i, j, k]]
                    + bz * scratch_r[[i, j, k]] / medium.density[[i, j, k]];
            }
        }
    }

    // ── vy sub-fields: vyx ← ∂_x txy, vyy ← ∂_y tyy, vyz ← ∂_z tyz ────

    // ∂_x txy → vyx
    spectral_mul_x(&spec_stress.txy, op_x_pos, kappa, &mut spec_scratch.txy);
    ifft_3d_array_into(&mut spec_scratch.txy, scratch_r);
    for i in 0..nx {
        let (ax, bx) = (ax_s[i], bx_s[i]);
        for j in 0..ny {
            for k in 0..nz {
                state.vyx[[i, j, k]] = ax * state.vyx[[i, j, k]]
                    + bx * scratch_r[[i, j, k]] / medium.density[[i, j, k]];
            }
        }
    }

    // ∂_y tyy → vyy
    spectral_mul_y(&spec_stress.tyy, op_y_pos, kappa, &mut spec_scratch.tyy);
    ifft_3d_array_into(&mut spec_scratch.tyy, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            let (ay, by) = (ay_s[j], by_s[j]);
            for k in 0..nz {
                state.vyy[[i, j, k]] = ay * state.vyy[[i, j, k]]
                    + by * scratch_r[[i, j, k]] / medium.density[[i, j, k]];
            }
        }
    }

    // ∂_z tyz → vyz
    spectral_mul_z(&spec_stress.tyz, op_z_pos, kappa, &mut spec_scratch.tyz);
    ifft_3d_array_into(&mut spec_scratch.tyz, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let (az, bz) = (az_s[k], bz_s[k]);
                state.vyz[[i, j, k]] = az * state.vyz[[i, j, k]]
                    + bz * scratch_r[[i, j, k]] / medium.density[[i, j, k]];
            }
        }
    }

    // ── vz sub-fields: vzx ← ∂_x txz, vzy ← ∂_y tyz, vzz ← ∂_z tzz ────

    // ∂_x txz → vzx
    spectral_mul_x(&spec_stress.txz, op_x_pos, kappa, &mut spec_scratch.txz);
    ifft_3d_array_into(&mut spec_scratch.txz, scratch_r);
    for i in 0..nx {
        let (ax, bx) = (ax_s[i], bx_s[i]);
        for j in 0..ny {
            for k in 0..nz {
                state.vzx[[i, j, k]] = ax * state.vzx[[i, j, k]]
                    + bx * scratch_r[[i, j, k]] / medium.density[[i, j, k]];
            }
        }
    }

    // ∂_y tyz → vzy
    spectral_mul_y(&spec_stress.tyz, op_y_pos, kappa, &mut spec_scratch.tyz);
    ifft_3d_array_into(&mut spec_scratch.tyz, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            let (ay, by) = (ay_s[j], by_s[j]);
            for k in 0..nz {
                state.vzy[[i, j, k]] = ay * state.vzy[[i, j, k]]
                    + by * scratch_r[[i, j, k]] / medium.density[[i, j, k]];
            }
        }
    }

    // ∂_z tzz → vzz
    spectral_mul_z(&spec_stress.tzz, op_z_pos, kappa, &mut spec_scratch.tzz);
    ifft_3d_array_into(&mut spec_scratch.tzz, scratch_r);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let (az, bz) = (az_s[k], bz_s[k]);
                state.vzz[[i, j, k]] = az * state.vzz[[i, j, k]]
                    + bz * scratch_r[[i, j, k]] / medium.density[[i, j, k]];
            }
        }
    }
}
