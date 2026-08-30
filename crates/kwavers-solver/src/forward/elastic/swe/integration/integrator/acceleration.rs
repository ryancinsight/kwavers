//! Stress-divergence evaluation and acceleration assembly.

use super::super::super::coordinates::GridPosition;
use super::super::super::scratch::ElasticStepScratch;
use super::super::super::stress::{stress_divergence_into, stress_divergence_plane_strain_into};
use super::super::super::types::{ElasticBodyForceConfig, ElasticWaveField};
use super::{body_force, TimeIntegrator};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::Array3;
use moirai_parallel::{
    for_each_chunk_pair_mut_enumerated_with, for_each_chunk_triple_mut_enumerated_with, Adaptive,
};

const ACCELERATION_CHUNK: usize = 4096;

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
    pub(super) fn compute_acceleration<S: StressOperator>(
        &self,
        field: &ElasticWaveField,
        scratch: &mut ElasticStepScratch,
        body_force: Option<&ElasticBodyForceConfig>,
        time: f64,
    ) -> KwaversResult<()> {
        S::evaluate(self.grid, self.lambda, self.mu, field, scratch);

        let div_x = scratch
            .div_x
            .as_slice()
            .expect("invariant: divergence x uses standard layout");
        let div_y = scratch
            .div_y
            .as_slice()
            .expect("invariant: divergence y uses standard layout");
        let div_z = scratch
            .div_z
            .as_slice()
            .expect("invariant: divergence z uses standard layout");
        let density = self
            .density
            .as_slice()
            .expect("invariant: density uses standard layout");

        if S::IS_PLANE_STRAIN {
            debug_assert!(body_force.is_none());
            let ax = scratch
                .ax
                .as_slice_mut()
                .expect("invariant: acceleration x uses standard layout");
            let ay = scratch
                .ay
                .as_slice_mut()
                .expect("invariant: acceleration y uses standard layout");
            if let Some(inverse_density) = self.uniform_inverse_density {
                fill_uniform_plane_acceleration(
                    ax,
                    ay,
                    div_x,
                    div_y,
                    ACCELERATION_CHUNK,
                    inverse_density,
                );
            } else {
                fill_variable_plane_acceleration(ax, ay, div_x, div_y, density, ACCELERATION_CHUNK);
            }
            return Ok(());
        }

        let ax = scratch
            .ax
            .as_slice_mut()
            .expect("invariant: acceleration x uses standard layout");
        let ay = scratch
            .ay
            .as_slice_mut()
            .expect("invariant: acceleration y uses standard layout");
        let az = scratch
            .az
            .as_slice_mut()
            .expect("invariant: acceleration z uses standard layout");

        if let Some(body_force) = body_force {
            let grid = self.grid;
            let (_, ny, nz) = (grid.nx, grid.ny, grid.nz);
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                ax,
                ay,
                az,
                ACCELERATION_CHUNK,
                |chunk_idx, ax_chunk, ay_chunk, az_chunk| {
                    let start = chunk_idx * ACCELERATION_CHUNK;
                    let mut position = GridPosition::from_flat(start, ny, nz);
                    for offset in 0..ax_chunk.len() {
                        let idx = start + offset;
                        let [i, j, k] = position.coordinates();
                        let force = body_force::evaluate(grid, body_force, i, j, k, time);
                        ax_chunk[offset] = (div_x[idx] + force[0]) / density[idx];
                        ay_chunk[offset] = (div_y[idx] + force[1]) / density[idx];
                        az_chunk[offset] = (div_z[idx] + force[2]) / density[idx];
                        position.advance(ny, nz);
                    }
                },
            );
            return Ok(());
        }

        if let Some(inverse_density) = self.uniform_inverse_density {
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                ax,
                ay,
                az,
                ACCELERATION_CHUNK,
                |chunk_idx, ax_chunk, ay_chunk, az_chunk| {
                    let start = chunk_idx * ACCELERATION_CHUNK;
                    for offset in 0..ax_chunk.len() {
                        let idx = start + offset;
                        ax_chunk[offset] = div_x[idx] * inverse_density;
                        ay_chunk[offset] = div_y[idx] * inverse_density;
                        az_chunk[offset] = div_z[idx] * inverse_density;
                    }
                },
            );
            return Ok(());
        }

        for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
            ax,
            ay,
            az,
            ACCELERATION_CHUNK,
            |chunk_idx, ax_chunk, ay_chunk, az_chunk| {
                let start = chunk_idx * ACCELERATION_CHUNK;
                for offset in 0..ax_chunk.len() {
                    let idx = start + offset;
                    ax_chunk[offset] = div_x[idx] / density[idx];
                    ay_chunk[offset] = div_y[idx] / density[idx];
                    az_chunk[offset] = div_z[idx] / density[idx];
                }
            },
        );
        Ok(())
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
        let ax = scratch
            .ax
            .as_slice_mut()
            .expect("invariant: acceleration x uses standard layout");
        let ay = scratch
            .ay
            .as_slice_mut()
            .expect("invariant: acceleration y uses standard layout");
        let az = scratch
            .az
            .as_slice_mut()
            .expect("invariant: acceleration z uses standard layout");
        let div_x = scratch
            .div_x
            .as_slice()
            .expect("invariant: divergence x uses standard layout");
        let div_y = scratch
            .div_y
            .as_slice()
            .expect("invariant: divergence y uses standard layout");
        let div_z = scratch
            .div_z
            .as_slice()
            .expect("invariant: divergence z uses standard layout");
        let density = self
            .density
            .as_slice()
            .expect("invariant: density uses standard layout");
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
            ax,
            ay,
            az,
            ACCELERATION_CHUNK,
            |chunk_idx, ax_chunk, ay_chunk, az_chunk| {
                let start = chunk_idx * ACCELERATION_CHUNK;
                let mut position = GridPosition::from_flat(start, ny, nz);
                for offset in 0..ax_chunk.len() {
                    let idx = start + offset;
                    let [i, j, k] = position.coordinates();
                    let force = force_at(i, j, k);
                    ax_chunk[offset] = (div_x[idx] + force[0]) / density[idx];
                    ay_chunk[offset] = (div_y[idx] + force[1]) / density[idx];
                    az_chunk[offset] = (div_z[idx] + force[2]) / density[idx];
                    position.advance(ny, nz);
                }
            },
        );
        Ok(())
    }
}

fn fill_uniform_plane_acceleration(
    ax: &mut [f64],
    ay: &mut [f64],
    div_x: &[f64],
    div_y: &[f64],
    chunk_size: usize,
    inverse_density: f64,
) {
    for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
        ax,
        ay,
        chunk_size,
        |chunk_idx, ax_chunk, ay_chunk| {
            let start = chunk_idx * chunk_size;
            for offset in 0..ax_chunk.len() {
                let idx = start + offset;
                ax_chunk[offset] = div_x[idx] * inverse_density;
                ay_chunk[offset] = div_y[idx] * inverse_density;
            }
        },
    );
}

fn fill_variable_plane_acceleration(
    ax: &mut [f64],
    ay: &mut [f64],
    div_x: &[f64],
    div_y: &[f64],
    density: &[f64],
    chunk_size: usize,
) {
    for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
        ax,
        ay,
        chunk_size,
        |chunk_idx, ax_chunk, ay_chunk| {
            let start = chunk_idx * chunk_size;
            for offset in 0..ax_chunk.len() {
                let idx = start + offset;
                ax_chunk[offset] = div_x[idx] / density[idx];
                ay_chunk[offset] = div_y[idx] / density[idx];
            }
        },
    );
}
