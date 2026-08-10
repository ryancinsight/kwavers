//! Parallel Monte Carlo launch and result assembly.

use super::MonteCarloSolver;
use anyhow::Result;
use moirai_parallel::{for_each_index_with, Adaptive};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use themis::CpuTopology;
use tyche_core::{
    sampling::{Counter, UserDomain},
    Seed, SplitMix64,
};

use crate::optics::monte_carlo::config::SimulationConfig;
use crate::optics::monte_carlo::interfaces::fresnel_reflectance;
use crate::optics::monte_carlo::result::MCResult;
use crate::optics::monte_carlo::source::PhotonSource;
use kwavers_grid::{Grid3D, GridDimensions};

impl MonteCarloSolver {
    /// Run Monte Carlo simulation.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn simulate(&self, source: &PhotonSource, config: &SimulationConfig) -> Result<MCResult> {
        let num_photons = config.num_photons;
        let lanes = CpuTopology::detect()
            .map(|topology| topology.logical_processors())
            .unwrap_or(1)
            .max(1);
        let chunk_size = (num_photons / lanes).max(1000);
        let chunk_count = num_photons.div_ceil(chunk_size);
        let total_voxels = total_voxels(&self.grid);
        let absorbed_energy = Arc::new(new_atomic_vec(total_voxels));
        let fluence = Arc::new(new_atomic_vec(total_voxels));
        let reflected_weight = Arc::new(AtomicU64::new(0));
        let seed = Seed::new(0x8B45_D2A7_61C3_1E9Fu64);

        for_each_index_with::<Adaptive, _>(chunk_count, |chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(num_photons);
            let chunk_address = u64::try_from(chunk_idx).expect("invariant: usize fits in u64");
            let chunk_seed =
                Counter::<UserDomain<0>, SplitMix64>::word(seed, chunk_address, 0);
            let mut rng = ChaCha8Rng::seed_from_u64(chunk_seed);

            for _ in start..end {
                let mut photon = source.launch_photon(&mut rng);
                if let Some(sp) = self.optical_map.get_properties(0, 0, 0) {
                    let n = sp.refractive_index();
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
