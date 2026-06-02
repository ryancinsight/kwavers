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

mod stress;
mod velocity;

use super::split_field_pml::{ElasticSplitFieldPml, SplitFieldState};
use super::types::ElasticPstdMedium;
use kwavers_math::fft::fft_3d_array_into;
use kwavers_physics::acoustics::mechanics::elastic_wave::{
    fields::VelocityFields,
    spectral_fields::{SpectralStressFields, SpectralVelocityFields},
};
use ndarray::{Array3, Zip};
use num_complex::Complex;

/// Mutable spectral scratch buffers for one split-field PML step.
///
/// All four fields are used as scratch; their contents on entry are ignored
/// and their contents on exit are undefined.
#[derive(Debug)]
pub(super) struct SpectralScratch<'a> {
    /// Stress spectral buffer (current step, input / scratch).
    pub stress: &'a mut SpectralStressFields,
    /// Stress spectral buffer (next step, output / scratch).
    pub stress_next: &'a mut SpectralStressFields,
    /// Velocity spectral buffer (current step, input / scratch).
    pub velocity_in: &'a mut SpectralVelocityFields,
    /// Velocity spectral buffer (next step, output / scratch).
    pub velocity_next: &'a mut SpectralVelocityFields,
}

/// Immutable spectral derivative operators and k-space correction factor.
#[derive(Debug, Clone, Copy)]
pub(super) struct SpectralOperators<'a> {
    /// Negative-staggered x derivative operator.
    pub dkx_neg: &'a Array3<Complex<f64>>,
    /// Negative-staggered y derivative operator.
    pub dky_neg: &'a Array3<Complex<f64>>,
    /// Negative-staggered z derivative operator.
    pub dkz_neg: &'a Array3<Complex<f64>>,
    /// Positive-staggered x derivative operator.
    pub dkx_pos: &'a Array3<Complex<f64>>,
    /// Positive-staggered y derivative operator.
    pub dky_pos: &'a Array3<Complex<f64>>,
    /// Positive-staggered z derivative operator.
    pub dkz_pos: &'a Array3<Complex<f64>>,
    /// K-space correction factor κ.
    pub kappa: &'a Array3<f64>,
}

/// Advance by one split-field PML time step.
///
/// All `scratch.*` fields are used as scratch; their contents on entry are
/// ignored and their contents on exit are undefined. The caller must not rely
/// on the spectral state after this call when using the split-field path.
pub(super) fn propagate_split_field_step(
    velocity: &mut VelocityFields,
    scratch: SpectralScratch<'_>,
    pml: &ElasticSplitFieldPml,
    state: &mut SplitFieldState,
    medium: &ElasticPstdMedium,
    operators: SpectralOperators<'_>,
    scratch_r: &mut Array3<f64>,
) {
    let SpectralScratch {
        stress: spectral_stress,
        stress_next: spectral_stress_next,
        velocity_in: spectral_velocity_in,
        velocity_next: spectral_velocity_next,
    } = scratch;
    let (alpha_x, beta_x) = pml.x_coeffs();
    let (alpha_y, beta_y) = pml.y_coeffs();
    let (alpha_z, beta_z) = pml.z_coeffs();
    let ax_s = alpha_x.as_slice().expect("alpha_x contiguous");
    let bx_s = beta_x.as_slice().expect("beta_x contiguous");
    let ay_s = alpha_y.as_slice().expect("alpha_y contiguous");
    let by_s = beta_y.as_slice().expect("beta_y contiguous");
    let az_s = alpha_z.as_slice().expect("alpha_z contiguous");
    let bz_s = beta_z.as_slice().expect("beta_z contiguous");

    // PHASE 1 — Stress sub-field updates.
    stress::update_stress_subfields(
        velocity,
        state,
        medium,
        &operators,
        spectral_velocity_in,
        spectral_velocity_next,
        scratch_r,
        ax_s,
        bx_s,
        ay_s,
        by_s,
        az_s,
        bz_s,
    );

    // PHASE 2 — Sum stress sub-fields and FFT each total component.
    //
    // spectral_stress.{txx,...} are used as scratch FFT output buffers.

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

    // PHASE 3 — Velocity sub-field updates.
    //
    // spectral_stress_next.{txx,...} used as scratch cspec for derivative
    // products; IFFT'd into scratch_r per sub-field group.
    velocity::update_velocity_subfields(
        state,
        medium,
        &operators,
        spectral_stress,
        spectral_stress_next,
        scratch_r,
        ax_s,
        bx_s,
        ay_s,
        by_s,
        az_s,
        bz_s,
    );

    // PHASE 4 — Sum velocity sub-fields → real-space total velocity.
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
