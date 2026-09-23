//! Stress-divergence evaluation and acceleration assembly.

use super::super::super::scratch::ElasticStepScratch;
use super::super::super::stress::{
    stress_divergence_into, stress_divergence_plane_strain_into, stress_kick_into, DensityScale,
    Lame, VelocityKick,
};
use super::super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use super::{body_force, TimeIntegrator};
use kwavers_core::error::KwaversResult;
use kwavers_core::traversal::{zip_mut_pair, zip_mut_triple, zip_mut_triple_indexed};
use kwavers_grid::Grid;
use leto::Array3;

/// What one acceleration evaluation leaves for the step to apply.
pub(super) enum Evaluated {
    /// The accelerations are in the scratch fields; the kick is the
    /// step's to apply.
    Stored,
    /// The evaluation has already kicked the velocities by half a step.
    Kicked,
}

/// Compile-time stress operator selected once per propagation.
pub(super) trait StressOperator {
    const IS_PLANE_STRAIN: bool;

    fn evaluate(
        grid: &Grid,
        lambda: &Array3<f64>,
        mu: &Array3<f64>,
        field: &ElasticWaveField,
        scratch: &mut ElasticStepScratch,
    );
}

/// Full three-dimensional stress evaluation.
pub(super) struct SpatialStress;

impl StressOperator for SpatialStress {
    const IS_PLANE_STRAIN: bool = false;

    #[inline]
    fn evaluate(
        grid: &Grid,
        lambda: &Array3<f64>,
        mu: &Array3<f64>,
        field: &ElasticWaveField,
        scratch: &mut ElasticStepScratch,
    ) {
        stress_divergence_into(grid, lambda, mu, field, scratch);
    }
}

/// Plane-strain stress evaluation for singleton-z in-plane propagation.
pub(super) struct PlaneStrainStress;

impl StressOperator for PlaneStrainStress {
    const IS_PLANE_STRAIN: bool = true;

    #[inline]
    fn evaluate(
        grid: &Grid,
        lambda: &Array3<f64>,
        mu: &Array3<f64>,
        field: &ElasticWaveField,
        scratch: &mut ElasticStepScratch,
    ) {
        stress_divergence_plane_strain_into(grid, lambda, mu, field, scratch);
    }
}

impl TimeIntegrator<'_> {
    /// The acceleration of `field` at `time`, either stored in `scratch`'s
    /// acceleration fields or already applied to the velocities as the
    /// `half_dt` kick; the result says which.
    pub(super) fn compute_acceleration<S: StressOperator>(
        &self,
        field: &mut ElasticWaveField,
        scratch: &mut ElasticStepScratch,
        body_force: Option<&ElasticBodyForceConfig>,
        time: f64,
        half_dt: f64,
    ) -> KwaversResult<Evaluated> {
        // Without a body force the scale and the kick are two operations
        // per lane, so they ride the divergence pass instead of costing a
        // pass that stores three accelerations and one that reads them
        // back. The full evaluation is the only one that fuses; the
        // plane-strain route keeps its own kernel, and `S::IS_PLANE_STRAIN`
        // resolves per instantiation, so neither carries the other's branch.
        if !S::IS_PLANE_STRAIN && body_force.is_none() {
            let scale = self.uniform_inverse_density.map_or_else(
                || DensityScale::Field(self.density.view()),
                DensityScale::UniformReciprocal,
            );
            let lame = self.uniform_lame.map_or_else(
                || Lame::Field {
                    lambda: self.lambda.view(),
                    mu: self.mu.view(),
                },
                |(lambda, mu)| Lame::Uniform { lambda, mu },
            );
            let kick = VelocityKick { scale, half_dt };
            stress_kick_into(self.grid, lame, field, &kick, scratch);
            return Ok(Evaluated::Kicked);
        }
        S::evaluate(self.grid, self.lambda, self.mu, field, scratch);
        let ElasticStepScratch {
            div_x,
            div_y,
            div_z,
            ax,
            ay,
            az,
            ..
        } = scratch;
        let density = self.density.view();

        if S::IS_PLANE_STRAIN {
            debug_assert!(body_force.is_none());
            let divergence = (div_x.view(), div_y.view());
            if let Some(inverse_density) = self.uniform_inverse_density {
                zip_mut_pair(
                    ax.view_mut(),
                    ay.view_mut(),
                    divergence,
                    |ax, ay, (&dx, &dy)| {
                        *ax = dx * inverse_density;
                        *ay = dy * inverse_density;
                    },
                );
            } else {
                zip_mut_pair(
                    ax.view_mut(),
                    ay.view_mut(),
                    (divergence.0, divergence.1, density),
                    |ax, ay, (&dx, &dy, &rho)| {
                        *ax = dx / rho;
                        *ay = dy / rho;
                    },
                );
            }
            return Ok(Evaluated::Stored);
        }

        let divergence = (div_x.view(), div_y.view(), div_z.view());
        if let Some(body_force) = body_force {
            let grid = self.grid;
            zip_mut_triple_indexed(
                ax.view_mut(),
                ay.view_mut(),
                az.view_mut(),
                (divergence.0, divergence.1, divergence.2, density),
                |[i, j, k], ax, ay, az, (&dx, &dy, &dz, &rho)| {
                    let force = body_force::evaluate(grid, body_force, i, j, k, time);
                    *ax = (dx + force[0]) / rho;
                    *ay = (dy + force[1]) / rho;
                    *az = (dz + force[2]) / rho;
                },
            );
        } else if let Some(inverse_density) = self.uniform_inverse_density {
            zip_mut_triple(
                ax.view_mut(),
                ay.view_mut(),
                az.view_mut(),
                divergence,
                |ax, ay, az, (&dx, &dy, &dz)| {
                    *ax = dx * inverse_density;
                    *ay = dy * inverse_density;
                    *az = dz * inverse_density;
                },
            );
        } else {
            zip_mut_triple(
                ax.view_mut(),
                ay.view_mut(),
                az.view_mut(),
                (divergence.0, divergence.1, divergence.2, density),
                |ax, ay, az, (&dx, &dy, &dz, &rho)| {
                    *ax = dx / rho;
                    *ay = dy / rho;
                    *az = dz / rho;
                },
            );
        }
        Ok(Evaluated::Stored)
    }

    pub(super) fn compute_acceleration_with_body_forces<F>(
        &self,
        field: &ElasticWaveField,
        scratch: &mut ElasticStepScratch,
        force_at: F,
    ) -> KwaversResult<()>
    where
        F: Fn(usize, usize, usize) -> [f64; 3] + Sync,
    {
        SpatialStress::evaluate(self.grid, self.lambda, self.mu, field, scratch);
        let ElasticStepScratch {
            div_x,
            div_y,
            div_z,
            ax,
            ay,
            az,
            ..
        } = scratch;
        zip_mut_triple_indexed(
            ax.view_mut(),
            ay.view_mut(),
            az.view_mut(),
            (
                div_x.view(),
                div_y.view(),
                div_z.view(),
                self.density.view(),
            ),
            |[i, j, k], ax, ay, az, (&dx, &dy, &dz, &rho)| {
                let force = force_at(i, j, k);
                *ax = (dx + force[0]) / rho;
                *ay = (dy + force[1]) / rho;
                *az = (dz + force[2]) / rho;
            },
        );
        Ok(())
    }
}
