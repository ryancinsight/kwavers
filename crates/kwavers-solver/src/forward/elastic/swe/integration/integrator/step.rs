//! Shared velocity-Verlet stepping across elastic dimensional regimes.

use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use super::acceleration::{PlaneStrainStress, SpatialStress, StressOperator};
use super::{body_force, TimeIntegrator};
use kwavers_core::error::KwaversResult;
use moirai_parallel::{for_each_chunk_triple_mut_enumerated_with, Adaptive};

const INTEGRATOR_CHUNK: usize = 4096;

impl TimeIntegrator<'_> {
    /// Perform one velocity-Verlet time step.
    ///
    /// The two acceleration evaluations bracket a half-step velocity update and
    /// a full displacement update. Reusing `scratch` makes the update
    /// allocation-free. Velocity-Verlet is second-order and symplectic; the
    /// separable PML then damps both displacement and velocity components.
    ///
    /// # Errors
    ///
    /// Propagates body-force validation and numerical errors.
    pub fn step(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_force: Option<&ElasticBodyForceConfig>,
        scratch: &mut ElasticStepScratch,
    ) -> KwaversResult<()> {
        if let Some(body_force) = body_force {
            body_force::validate(body_force)?;
        }
        self.integrate::<SpatialStress, _>(field, dt, scratch, |field, scratch, time| {
            self.compute_acceleration::<SpatialStress>(field, scratch, body_force, time)
        })
    }

    /// Perform one plane-strain point-force step.
    ///
    /// The caller guarantees a singleton z axis and zero z displacement,
    /// velocity, and forcing, plus fresh scratch storage whose out-of-plane
    /// arrays are zero. The stress operator is selected statically.
    pub(crate) fn step_plane_strain(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        scratch: &mut ElasticStepScratch,
    ) -> KwaversResult<()> {
        debug_assert_eq!(field.uz.shape()[2], 1);
        self.integrate::<PlaneStrainStress, _>(field, dt, scratch, |field, scratch, time| {
            self.compute_acceleration::<PlaneStrainStress>(field, scratch, None, time)
        })
    }

    /// Perform one step with multiple simultaneous distributed body forces.
    ///
    /// # Errors
    ///
    /// Propagates body-force validation and numerical errors.
    pub fn step_with_body_forces(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_forces: &[ElasticBodyForceConfig],
        scratch: &mut ElasticStepScratch,
    ) -> KwaversResult<()> {
        for body_force in body_forces {
            body_force::validate(body_force)?;
        }
        self.integrate::<SpatialStress, _>(field, dt, scratch, |field, scratch, time| {
            self.compute_acceleration_with_body_forces(field, scratch, body_forces, time)
        })
    }

    fn integrate<S, F>(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        scratch: &mut ElasticStepScratch,
        mut acceleration: F,
    ) -> KwaversResult<()>
    where
        S: StressOperator,
        F: FnMut(&ElasticWaveField, &mut ElasticStepScratch, f64) -> KwaversResult<()>,
    {
        acceleration(field, scratch, field.time)?;
        let half_dt = 0.5 * dt;

        update_components::<S>(
            &mut field.vx,
            &mut field.vy,
            &mut field.vz,
            &scratch.ax,
            &scratch.ay,
            &scratch.az,
            half_dt,
        );
        update_components::<S>(
            &mut field.ux,
            &mut field.uy,
            &mut field.uz,
            &field.vx,
            &field.vy,
            &field.vz,
            dt,
        );

        acceleration(field, scratch, field.time + dt)?;
        update_components::<S>(
            &mut field.vx,
            &mut field.vy,
            &mut field.vz,
            &scratch.ax,
            &scratch.ay,
            &scratch.az,
            half_dt,
        );
        self.apply_pml_damping_for::<S>(field, dt, scratch);
        Ok(())
    }
}

fn update_components<S: StressOperator>(
    x: &mut leto::Array3<f64>,
    y: &mut leto::Array3<f64>,
    z: &mut leto::Array3<f64>,
    delta_x: &leto::Array3<f64>,
    delta_y: &leto::Array3<f64>,
    delta_z: &leto::Array3<f64>,
    scale: f64,
) {
    if S::IS_PLANE_STRAIN {
        let x = x
            .as_slice_mut()
            .expect("invariant: x component uses standard layout");
        let y = y
            .as_slice_mut()
            .expect("invariant: y component uses standard layout");
        let delta_x = delta_x
            .as_slice()
            .expect("invariant: x delta uses standard layout");
        let delta_y = delta_y
            .as_slice()
            .expect("invariant: y delta uses standard layout");
        moirai_parallel::for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
            x,
            y,
            INTEGRATOR_CHUNK,
            |chunk_idx, x_chunk, y_chunk| {
                let start = chunk_idx * INTEGRATOR_CHUNK;
                for offset in 0..x_chunk.len() {
                    let idx = start + offset;
                    x_chunk[offset] += scale * delta_x[idx];
                    y_chunk[offset] += scale * delta_y[idx];
                }
            },
        );
        return;
    }

    let x = x
        .as_slice_mut()
        .expect("invariant: x component uses standard layout");
    let y = y
        .as_slice_mut()
        .expect("invariant: y component uses standard layout");
    let z = z
        .as_slice_mut()
        .expect("invariant: z component uses standard layout");
    let delta_x = delta_x
        .as_slice()
        .expect("invariant: x delta uses standard layout");
    let delta_y = delta_y
        .as_slice()
        .expect("invariant: y delta uses standard layout");
    let delta_z = delta_z
        .as_slice()
        .expect("invariant: z delta uses standard layout");
    for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
        x,
        y,
        z,
        INTEGRATOR_CHUNK,
        |chunk_idx, x_chunk, y_chunk, z_chunk| {
            let start = chunk_idx * INTEGRATOR_CHUNK;
            for offset in 0..x_chunk.len() {
                let idx = start + offset;
                x_chunk[offset] += scale * delta_x[idx];
                y_chunk[offset] += scale * delta_y[idx];
                z_chunk[offset] += scale * delta_z[idx];
            }
        },
    );
}
