//! Shared velocity-Verlet stepping across elastic dimensional regimes.

use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use super::acceleration::{PlaneStrainStress, SpatialStress, StressOperator};
use super::{body_force, PreparedBodyForces, TimeIntegrator};
use kwavers_core::error::KwaversResult;
use kwavers_core::traversal::{zip_mut_pair, zip_mut_triple};

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
            self.compute_acceleration_with_body_forces(field, scratch, |i, j, k| {
                let mut force = [0.0; 3];
                for body_force in body_forces {
                    let value = body_force::evaluate(self.grid, body_force, i, j, k, time);
                    force[0] += value[0];
                    force[1] += value[1];
                    force[2] += value[2];
                }
                force
            })
        })
    }

    /// Perform one step with spatially prepared distributed body forces.
    pub(crate) fn step_with_prepared_body_forces(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_forces: &mut PreparedBodyForces,
        scratch: &mut ElasticStepScratch,
    ) -> KwaversResult<()> {
        body_forces.validate_grid(self.grid)?;
        self.integrate::<SpatialStress, _>(field, dt, scratch, |field, scratch, time| {
            body_forces.update_time(time);
            self.compute_acceleration_with_body_forces(field, scratch, |i, j, k| {
                body_forces.force_at(i, j, k)
            })
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

/// `x += scale · delta_x` and likewise for y and z; plane strain leaves the
/// out-of-plane component alone.
pub(super) fn update_components<S: StressOperator>(
    x: &mut leto::Array3<f64>,
    y: &mut leto::Array3<f64>,
    z: &mut leto::Array3<f64>,
    delta_x: &leto::Array3<f64>,
    delta_y: &leto::Array3<f64>,
    delta_z: &leto::Array3<f64>,
    scale: f64,
) {
    if S::IS_PLANE_STRAIN {
        zip_mut_pair(
            x.view_mut(),
            y.view_mut(),
            (delta_x.view(), delta_y.view()),
            |x, y, (&dx, &dy)| {
                *x += scale * dx;
                *y += scale * dy;
            },
        );
        return;
    }
    zip_mut_triple(
        x.view_mut(),
        y.view_mut(),
        z.view_mut(),
        (delta_x.view(), delta_y.view(), delta_z.view()),
        |x, y, z, (&dx, &dy, &dz)| {
            *x += scale * dx;
            *y += scale * dy;
            *z += scale * dz;
        },
    );
}
