//! Separable per-axis PML damping.

use super::super::super::coordinates::GridPosition;
use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::ElasticWaveField;
#[cfg(test)]
use super::acceleration::SpatialStress;
use super::acceleration::StressOperator;
use super::TimeIntegrator;
use moirai_parallel::{for_each_chunk_buffers_mut_enumerated_with, Adaptive};

// Four chunks cover the hosted 16³ workflow, while larger grids retain broad
// task fanout and each task amortizes coordinate decoding over 1,024 cells.
const DAMPING_CHUNK: usize = 1024;

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
            damp_components(
                [&mut field.vx, &mut field.vy, &mut field.ux, &mut field.uy],
                factor_x,
                factor_y,
                factor_z,
                ny,
                nz,
            );
            return;
        }

        damp_components(
            [
                &mut field.vx,
                &mut field.vy,
                &mut field.vz,
                &mut field.ux,
                &mut field.uy,
                &mut field.uz,
            ],
            factor_x,
            factor_y,
            factor_z,
            ny,
            nz,
        );
    }
}

fn damp_components<const N: usize>(
    components: [&mut leto::Array3<f64>; N],
    factor_x: &[f64],
    factor_y: &[f64],
    factor_z: &[f64],
    ny: usize,
    nz: usize,
) {
    let buffers = components.map(|component| {
        component
            .as_slice_mut()
            .expect("invariant: damped component uses standard layout")
    });
    for_each_chunk_buffers_mut_enumerated_with::<Adaptive, _, _, N>(
        buffers,
        DAMPING_CHUNK,
        |chunk_idx, mut chunks| {
            let start = chunk_idx * DAMPING_CHUNK;
            let mut position = GridPosition::from_flat(start, ny, nz);
            let chunk_len = chunks.first().map_or(0, |chunk| chunk.len());
            for offset in 0..chunk_len {
                let [i, j, k] = position.coordinates();
                let factor = factor_x[i] * factor_y[j] * factor_z[k];
                if factor < 1.0 {
                    for chunk in &mut chunks {
                        chunk[offset] *= factor;
                    }
                }
                position.advance(ny, nz);
            }
        },
    )
    .expect("invariant: damped components have equal lengths");
}
