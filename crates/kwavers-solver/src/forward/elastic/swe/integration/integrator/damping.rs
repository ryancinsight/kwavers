//! Separable per-axis PML damping.

use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::ElasticWaveField;
#[cfg(test)]
use super::acceleration::SpatialStress;
use super::acceleration::StressOperator;
use super::TimeIntegrator;
use moirai_parallel::{for_each_chunk_triple_mut_enumerated_with, Adaptive};

const DAMPING_CHUNK: usize = 4096;

impl TimeIntegrator<'_> {
    #[cfg(test)]
    pub(crate) fn apply_pml_damping(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        scratch: &mut ElasticStepScratch,
    ) {
        self.apply_pml_damping_for::<SpatialStress>(field, dt, scratch);
    }

    pub(super) fn apply_pml_damping_for<S: StressOperator>(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        scratch: &mut ElasticStepScratch,
    ) {
        let [nx, ny, nz] = field.vx.shape();
        let (factor_x, factor_y, factor_z) =
            scratch.pml_factors(&self.sigma_x, &self.sigma_y, &self.sigma_z, dt);
        debug_assert_eq!(factor_x.len(), nx);
        debug_assert_eq!(factor_y.len(), ny);
        debug_assert_eq!(factor_z.len(), nz);

        if S::IS_PLANE_STRAIN {
            let z_factor = factor_z
                .first()
                .copied()
                .expect("invariant: plane-strain z axis has one damping factor");
            damp_plane_components(
                &mut field.vx,
                &mut field.vy,
                factor_x,
                factor_y,
                z_factor,
                ny,
            );
            damp_plane_components(
                &mut field.ux,
                &mut field.uy,
                factor_x,
                factor_y,
                z_factor,
                ny,
            );
            return;
        }

        damp_components(
            &mut field.vx,
            &mut field.vy,
            &mut field.vz,
            factor_x,
            factor_y,
            factor_z,
            ny,
            nz,
        );
        damp_components(
            &mut field.ux,
            &mut field.uy,
            &mut field.uz,
            factor_x,
            factor_y,
            factor_z,
            ny,
            nz,
        );
    }
}

fn damp_plane_components(
    x: &mut leto::Array3<f64>,
    y: &mut leto::Array3<f64>,
    factor_x: &[f64],
    factor_y: &[f64],
    factor_z: f64,
    ny: usize,
) {
    let x = x
        .as_slice_mut()
        .expect("invariant: x component uses standard layout");
    let y = y
        .as_slice_mut()
        .expect("invariant: y component uses standard layout");
    moirai_parallel::for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
        x,
        y,
        DAMPING_CHUNK,
        |chunk_idx, x_chunk, y_chunk| {
            let start = chunk_idx * DAMPING_CHUNK;
            for offset in 0..x_chunk.len() {
                let idx = start + offset;
                let i = idx / ny;
                let j = idx % ny;
                let factor = factor_x[i] * factor_y[j] * factor_z;
                if factor < 1.0 {
                    x_chunk[offset] *= factor;
                    y_chunk[offset] *= factor;
                }
            }
        },
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "three fields and three separable axes form one kernel"
)]
fn damp_components(
    x: &mut leto::Array3<f64>,
    y: &mut leto::Array3<f64>,
    z: &mut leto::Array3<f64>,
    factor_x: &[f64],
    factor_y: &[f64],
    factor_z: &[f64],
    ny: usize,
    nz: usize,
) {
    let x = x
        .as_slice_mut()
        .expect("invariant: x component uses standard layout");
    let y = y
        .as_slice_mut()
        .expect("invariant: y component uses standard layout");
    let z = z
        .as_slice_mut()
        .expect("invariant: z component uses standard layout");
    for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
        x,
        y,
        z,
        DAMPING_CHUNK,
        |chunk_idx, x_chunk, y_chunk, z_chunk| {
            let start = chunk_idx * DAMPING_CHUNK;
            for offset in 0..x_chunk.len() {
                let idx = start + offset;
                let i = idx / (ny * nz);
                let j = (idx / nz) % ny;
                let k = idx % nz;
                let factor = factor_x[i] * factor_y[j] * factor_z[k];
                if factor < 1.0 {
                    x_chunk[offset] *= factor;
                    y_chunk[offset] *= factor;
                    z_chunk[offset] *= factor;
                }
            }
        },
    );
}
