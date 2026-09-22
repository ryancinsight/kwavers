//! Shared velocity-Verlet stepping across elastic dimensional regimes.

use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use super::acceleration::{PlaneStrainStress, SpatialStress, StressOperator};
use super::{body_force, PreparedBodyForces, TimeIntegrator};
use kwavers_core::arena::last_level_cache_bytes;
use kwavers_core::error::KwaversResult;
use kwavers_core::traversal::{zip_mut_many, zip_mut_pair, zip_mut_triple};

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

        let live = if S::IS_PLANE_STRAIN { 4 } else { 6 };
        kick_then_drift::<S>(
            field,
            [&scratch.ax, &scratch.ay, &scratch.az],
            half_dt,
            dt,
            KickDriftRoute::for_live_set(field.vx.len(), live),
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

/// Which shape the kick and the drift take.
///
/// Composed, the kick writes the velocities and the drift reads them back:
/// the pair keeps `live` fields in flight. While those fit the last-level
/// cache the re-read is a cache hit and fusing buys nothing -- it only adds
/// stream pressure, since one pass keeps half again as many fields live at
/// once. Past that the re-read comes from DRAM, and saving it is the whole
/// point.
///
/// Measured through the probe's paired arms at 64, 80 and 96 cubed (fastest
/// repeat, three runs each, against a last-level cache of about 36 MB):
/// composed wins by 20% with 12.6 MB live and by 3-4% with 24.6 MB, and
/// loses by 21% with 42.5 MB. The crossover this places between 24.6 and
/// 42.5 MB is where it was measured, and the margin shrinking from 20% to
/// 3-4% as the live set approaches the cache is the trend the rule predicts.
///
/// A platform that reports no cache size keeps the composed route: without
/// a boundary there is no evidence the pair does not fit, and composed is
/// the route that wins whenever it does.
pub(super) enum KickDriftRoute {
    /// The kick and the drift as separate passes.
    Composed,
    /// Both in one pass over every field they write.
    Fused,
}

impl KickDriftRoute {
    /// The route for a field of `elements` whose composed form keeps `live`
    /// fields in flight.
    fn for_live_set(elements: usize, live: usize) -> Self {
        let past_cache = last_level_cache_bytes()
            .is_some_and(|cache| live * elements * size_of::<f64>() > cache);
        if past_cache {
            Self::Fused
        } else {
            Self::Composed
        }
    }
}

/// `v += half_dt · a` and `u += dt · v`, in one traversal or two.
///
/// Velocity-Verlet's kick and drift are separate passes over six fields: the
/// kick writes the three velocities from the three accelerations, and the
/// drift reads those velocities back to write the three displacements. Both
/// are element-local, so one pass does the same arithmetic on the same
/// operands in the same order -- the values are bit-identical either way --
/// while reading each velocity once instead of writing it, evicting it and
/// reading it back.
///
/// One pass is possible because the traversal takes its six destinations as
/// one array; through the pair and triple forms it would be three calls and
/// three parallel regions. Whether it is worth taking is
/// [`fusion_pays`]'s question, and at the grid sizes the tests run the
/// answer is no.
///
/// Plane strain leaves the out-of-plane component alone, so it writes four
/// fields from two accelerations.
pub(super) fn kick_then_drift<S: StressOperator>(
    field: &mut ElasticWaveField,
    [ax, ay, az]: [&leto::Array3<f64>; 3],
    half_dt: f64,
    dt: f64,
    route: KickDriftRoute,
) {
    let ElasticWaveField {
        ux,
        uy,
        uz,
        vx,
        vy,
        vz,
        ..
    } = field;
    if S::IS_PLANE_STRAIN {
        if matches!(route, KickDriftRoute::Composed) {
            zip_mut_pair(
                vx.view_mut(),
                vy.view_mut(),
                (ax.view(), ay.view()),
                |vx, vy, (&ax, &ay)| {
                    *vx += half_dt * ax;
                    *vy += half_dt * ay;
                },
            );
            zip_mut_pair(
                ux.view_mut(),
                uy.view_mut(),
                (vx.view(), vy.view()),
                |ux, uy, (&vx, &vy)| {
                    *ux += dt * vx;
                    *uy += dt * vy;
                },
            );
            return;
        }
        zip_mut_many(
            [vx.view_mut(), vy.view_mut(), ux.view_mut(), uy.view_mut()],
            (ax.view(), ay.view()),
            |[vx, vy, ux, uy], (&ax, &ay)| {
                *vx += half_dt * ax;
                *vy += half_dt * ay;
                *ux += dt * *vx;
                *uy += dt * *vy;
            },
        );
        return;
    }
    if matches!(route, KickDriftRoute::Composed) {
        zip_mut_triple(
            vx.view_mut(),
            vy.view_mut(),
            vz.view_mut(),
            (ax.view(), ay.view(), az.view()),
            |vx, vy, vz, (&ax, &ay, &az)| {
                *vx += half_dt * ax;
                *vy += half_dt * ay;
                *vz += half_dt * az;
            },
        );
        zip_mut_triple(
            ux.view_mut(),
            uy.view_mut(),
            uz.view_mut(),
            (vx.view(), vy.view(), vz.view()),
            |ux, uy, uz, (&vx, &vy, &vz)| {
                *ux += dt * vx;
                *uy += dt * vy;
                *uz += dt * vz;
            },
        );
        return;
    }
    zip_mut_many(
        [
            vx.view_mut(),
            vy.view_mut(),
            vz.view_mut(),
            ux.view_mut(),
            uy.view_mut(),
            uz.view_mut(),
        ],
        (ax.view(), ay.view(), az.view()),
        |[vx, vy, vz, ux, uy, uz], (&ax, &ay, &az)| {
            *vx += half_dt * ax;
            *vy += half_dt * ay;
            *vz += half_dt * az;
            *ux += dt * *vx;
            *uy += dt * *vy;
            *uz += dt * *vz;
        },
    );
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

#[cfg(test)]
mod tests {
    use super::super::super::super::types::ElasticWaveField;
    use super::{kick_then_drift, KickDriftRoute, PlaneStrainStress, SpatialStress};
    use leto::Array3;

    /// A field whose every component differs and whose values do not round
    /// to each other, so a swapped or dropped component changes the result.
    fn seeded(n: usize) -> (ElasticWaveField, [Array3<f64>; 3]) {
        let mut field = ElasticWaveField::new(n, n, n);
        let mut accelerations = [(); 3].map(|()| Array3::zeros([n, n, n]));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let position = ((i * n + j) * n + k) as f64;
                    let at = [i, j, k];
                    for (slot, component) in [
                        &mut field.ux,
                        &mut field.uy,
                        &mut field.uz,
                        &mut field.vx,
                        &mut field.vy,
                        &mut field.vz,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        component[at] = (position + slot as f64 * 0.125).sin();
                    }
                    for (slot, component) in accelerations.iter_mut().enumerate() {
                        component[at] = (position * 0.5 + slot as f64 * 0.375).cos();
                    }
                }
            }
        }
        (field, accelerations)
    }

    /// The routes differ in traversal shape only: both do the same arithmetic
    /// on the same operands in the same order, so they agree to the bit. That
    /// is what makes choosing between them by cache size free of any effect
    /// on the result.
    #[test]
    fn both_kick_drift_routes_agree_to_the_bit() {
        for n in [1, 2, 5, 9] {
            let (mut composed, accelerations) = seeded(n);
            let (mut fused, _) = seeded(n);
            let terms = [&accelerations[0], &accelerations[1], &accelerations[2]];
            let (half_dt, dt) = (0.5 * 0.014, 0.014);

            kick_then_drift::<SpatialStress>(
                &mut composed,
                terms,
                half_dt,
                dt,
                KickDriftRoute::Composed,
            );
            kick_then_drift::<SpatialStress>(&mut fused, terms, half_dt, dt, KickDriftRoute::Fused);

            for (label, left, right) in [
                ("ux", &composed.ux, &fused.ux),
                ("uy", &composed.uy, &fused.uy),
                ("uz", &composed.uz, &fused.uz),
                ("vx", &composed.vx, &fused.vx),
                ("vy", &composed.vy, &fused.vy),
                ("vz", &composed.vz, &fused.vz),
            ] {
                for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{label} at flat {index} of {n} cubed: {a} against {b}"
                    );
                }
            }
        }
    }

    /// Plane strain writes four fields from two accelerations and leaves the
    /// out-of-plane component untouched, on either route.
    #[test]
    fn both_plane_strain_routes_agree_and_leave_z_alone() {
        const N: usize = 7;
        let (mut composed, accelerations) = seeded(N);
        let (mut fused, _) = seeded(N);
        let (untouched_uz, untouched_vz) = (composed.uz.clone(), composed.vz.clone());
        let terms = [&accelerations[0], &accelerations[1], &accelerations[2]];
        let (half_dt, dt) = (0.5 * 0.014, 0.014);

        kick_then_drift::<PlaneStrainStress>(
            &mut composed,
            terms,
            half_dt,
            dt,
            KickDriftRoute::Composed,
        );
        kick_then_drift::<PlaneStrainStress>(&mut fused, terms, half_dt, dt, KickDriftRoute::Fused);

        for (label, left, right) in [
            ("ux", &composed.ux, &fused.ux),
            ("uy", &composed.uy, &fused.uy),
            ("vx", &composed.vx, &fused.vx),
            ("vy", &composed.vy, &fused.vy),
        ] {
            for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{label} at flat {index}");
            }
        }
        for (label, moved, held) in [
            ("uz", &composed.uz, &untouched_uz),
            ("vz", &composed.vz, &untouched_vz),
        ] {
            for (index, (a, b)) in moved.iter().zip(held.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{label} at flat {index} moved");
            }
        }
    }
}
