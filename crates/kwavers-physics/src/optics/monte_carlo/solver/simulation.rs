//! Parallel Monte Carlo launch and result assembly.

use super::MonteCarloSolver;
use anyhow::Result;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use kwavers_domain::grid::{Grid3D, GridDimensions};
use crate::optics::monte_carlo::config::SimulationConfig;
use crate::optics::monte_carlo::interfaces::fresnel_reflectance;
use crate::optics::monte_carlo::result::MCResult;
use crate::optics::monte_carlo::source::PhotonSource;

impl MonteCarloSolver {
    /// Run Monte Carlo simulation.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn simulate(&self, source: &PhotonSource, config: &SimulationConfig) -> Result<MCResult> {
        let num_photons = config.num_photons;
        let chunk_size = (num_photons / rayon::current_num_threads()).max(1000);
        let total_voxels = total_voxels(&self.grid);
        let absorbed_energy = Arc::new(new_atomic_vec(total_voxels));
        let fluence = Arc::new(new_atomic_vec(total_voxels));
        let reflected_weight = Arc::new(AtomicU64::new(0));

        (0..num_photons)
            .into_par_iter()
            .chunks(chunk_size)
            .for_each(|chunk| {
                let mut rng = rand::thread_rng();

                for _ in chunk {
                    let mut photon = source.launch_photon(&mut rng);
                    if let Some(sp) = self.optical_map.get(0, 0, 0) {
                        let n = sp.refractive_index;
                        if n > 1.0 + 1e-9 {
                            photon.weight *= 1.0 - fresnel_reflectance(1.0, n, 1.0);
                        }
                    }
                    self.trace_photon(
                        photon,
                        config,
                        &absorbed_energy,
                        &fluence,
                        &reflected_weight,
                        &mut rng,
                    );
                }
            });

        let absorbed_energy = collect_atomic_vec(&absorbed_energy);
        let fluence = collect_atomic_vec(&fluence);
        let reflected = f64::from_bits(reflected_weight.load(Ordering::Relaxed));
        let diffuse_reflectance = reflected / num_photons as f64;

        Ok(MCResult {
            dimensions: GridDimensions::from_grid(&self.grid),
            absorbed_energy,
            fluence,
            num_photons,
            diffuse_reflectance,
        })
    }
}

fn total_voxels(grid: &Grid3D) -> usize {
    grid.nx * grid.ny * grid.nz
}

fn new_atomic_vec(len: usize) -> Vec<AtomicU64> {
    (0..len).map(|_| AtomicU64::new(0)).collect()
}

fn collect_atomic_vec(values: &[AtomicU64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| f64::from_bits(value.load(Ordering::Relaxed)))
        .collect()
}
