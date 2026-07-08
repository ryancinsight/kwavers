//! Source injection and sensor recording helpers for the elastic PSTD
//! orchestrator.
//!
//! # Scope
//!
//! Pure free functions operating on [`VelocityFields`] and [`SplitFieldState`].
//! Separated from `orchestrator.rs` to honour the 500-line structural limit
//! and enforce the Single Responsibility Principle: the orchestrator owns the
//! time loop; this module owns the signal injection / sensor bookkeeping.

use super::split_field_pml::SplitFieldState;
use super::types::{ElasticPstdSourceMode, ElasticPstdVelocitySource};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_physics::acoustics::mechanics::elastic_wave::fields::VelocityFields;
use leto::{Array1, Array2, Array3};

// ─── Validation ──────────────────────────────────────────────────────────────

/// Return `Err` if the source mask shape does not match the grid or any signal
/// vector has the wrong length.
pub(super) fn validate_source(
    src: &ElasticPstdVelocitySource,
    shape: (usize, usize, usize),
    n_steps: usize,
) -> KwaversResult<()> {
    let expected_shape = [shape.0, shape.1, shape.2];
    if src.mask.shape() != expected_shape {
        return Err(KwaversError::InvalidInput(format!(
            "ElasticPstdVelocitySource.mask shape {:?} must equal grid {:?}",
            src.mask.shape(),
            shape
        )));
    }
    for (axis, sig) in [("ux", &src.ux), ("uy", &src.uy), ("uz", &src.uz)] {
        if let Some(s) = sig {
            let len = s.shape()[0];
            if len != n_steps {
                return Err(KwaversError::InvalidInput(format!(
                    "{axis} signal length {} must equal n_steps {n_steps}",
                    len
                )));
            }
        }
    }
    Ok(())
}

// ─── Injection into total velocity ───────────────────────────────────────────

/// Inject the source signal at `step` into `velocity.vx/vy/vz` at all masked
/// positions.
pub(super) fn inject_velocity_source(
    velocity: &mut VelocityFields,
    src: &ElasticPstdVelocitySource,
    step: usize,
) {
    let active = masked_indices(src);
    inject_into(&mut velocity.vx, &src.ux, &active, src.mode, step);
    inject_into(&mut velocity.vy, &src.uy, &active, src.mode, step);
    inject_into(&mut velocity.vz, &src.uz, &active, src.mode, step);
}

/// Inject the velocity source into the x-directional velocity sub-fields
/// (`vxx`, `vyx`, `vzx`) so that the Phase-4 summation
/// `vα = vα_x + vα_y + vα_z` in [`super::split_field_step`] preserves the
/// source contribution.
///
/// # Invariant
///
/// For the source-injected step the following chain holds:
///
/// ```text
///   Phase 4: vx^{n+1} = vxx^{n+1} + vxy^{n+1} + vxz^{n+1}
///          = (α_x·vxx^n + β_x·∂_x·σ_{xx}^{n+1} + s[n])
///            + (α_x·vxy^n + β_x·∂_y·σ_{xy}^{n+1})
///            + (α_x·vxz^n + β_x·∂_z·σ_{xz}^{n+1})
///          = α_x·vx^n + β_x·(∇·σ)_x^{n+1} + s[n]    ✓
/// ```
///
/// Without this injection Phase 4 clobbers the source injected into
/// `velocity.vx` before the step, yielding ~1.5× amplitude error.
pub(super) fn inject_velocity_source_subfields(
    state: &mut SplitFieldState,
    src: &ElasticPstdVelocitySource,
    step: usize,
) {
    let active = masked_indices(src);
    inject_into(&mut state.vxx, &src.ux, &active, src.mode, step);
    inject_into(&mut state.vyx, &src.uy, &active, src.mode, step);
    inject_into(&mut state.vzx, &src.uz, &active, src.mode, step);
}

// ─── Sensor recording ────────────────────────────────────────────────────────

/// Copy the per-component velocity values at the indexed grid positions into
/// the sensor trace buffers at column `step`.
pub(super) fn record_sensors(
    velocity: &VelocityFields,
    sensor_indices: &[(usize, usize, usize)],
    step: usize,
    sensor_vx: &mut Option<Array2<f64>>,
    sensor_vy: &mut Option<Array2<f64>>,
    sensor_vz: &mut Option<Array2<f64>>,
) {
    if let Some(ref mut buf) = sensor_vx {
        for (row, &(i, j, k)) in sensor_indices.iter().enumerate() {
            buf[[row, step]] = velocity.vx[[i, j, k]];
        }
    }
    if let Some(ref mut buf) = sensor_vy {
        for (row, &(i, j, k)) in sensor_indices.iter().enumerate() {
            buf[[row, step]] = velocity.vy[[i, j, k]];
        }
    }
    if let Some(ref mut buf) = sensor_vz {
        for (row, &(i, j, k)) in sensor_indices.iter().enumerate() {
            buf[[row, step]] = velocity.vz[[i, j, k]];
        }
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Collect the (i,j,k) coordinates of all `true` entries in the source mask.
#[inline]
fn masked_indices(src: &ElasticPstdVelocitySource) -> Vec<(usize, usize, usize)> {
    let [nx, ny, nz] = src.mask.shape();
    let mut indices = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if src.mask[[i, j, k]] {
                    indices.push((i, j, k));
                }
            }
        }
    }
    indices
}

/// Inject signal value at `step` into `field` at all `active` positions.
#[inline]
fn inject_into(
    field: &mut Array3<f64>,
    sig: &Option<Array1<f64>>,
    active: &[(usize, usize, usize)],
    mode: ElasticPstdSourceMode,
    step: usize,
) {
    if let Some(s) = sig {
        if let Some(&val) = s.as_slice().and_then(|sl| sl.get(step)) {
            for &(i, j, k) in active {
                match mode {
                    ElasticPstdSourceMode::Additive => field[[i, j, k]] += val,
                    ElasticPstdSourceMode::Dirichlet => field[[i, j, k]] = val,
                }
            }
        }
    }
}
