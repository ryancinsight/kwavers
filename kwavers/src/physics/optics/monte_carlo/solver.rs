use anyhow::Result;
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::domain::grid::{Grid3D, GridDimensions};
use crate::physics::optics::map_builder::OpticalPropertyMap;

use crate::physics::optics::monte_carlo::config::SimulationConfig;
use crate::physics::optics::monte_carlo::photon::Photon;
use crate::physics::optics::monte_carlo::result::MCResult;
use crate::physics::optics::monte_carlo::source::PhotonSource;
use crate::physics::optics::monte_carlo::utils::{atomic_add_f64, scatter_photon};

/// Monte Carlo photon transport solver
#[derive(Debug)]
pub struct MonteCarloSolver {
    grid: Grid3D,
    optical_map: OpticalPropertyMap,
}

impl MonteCarloSolver {
    /// Create new Monte Carlo solver
    pub fn new(grid: Grid3D, optical_map: OpticalPropertyMap) -> Self {
        Self { grid, optical_map }
    }

    /// Run Monte Carlo simulation
    pub fn simulate(&self, source: &PhotonSource, config: &SimulationConfig) -> Result<MCResult> {
        let num_photons = config.num_photons;
        let chunk_size = (num_photons / rayon::current_num_threads()).max(1000);

        // Accumulators for absorbed energy and fluence
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let total_voxels = nx * ny * nz;

        // Thread-safe accumulators
        let absorbed_energy = Arc::new(
            (0..total_voxels)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>(),
        );
        let fluence = Arc::new(
            (0..total_voxels)
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>(),
        );

        // Launch photons in parallel
        (0..num_photons)
            .into_par_iter()
            .chunks(chunk_size)
            .for_each(|chunk| {
                let mut rng = rand::thread_rng();

                for _ in chunk {
                    let photon = source.launch_photon(&mut rng);
                    self.trace_photon(photon, config, &absorbed_energy, &fluence, &mut rng);
                }
            });

        // Convert atomic accumulators to regular vectors
        let absorbed_energy: Vec<f64> = absorbed_energy
            .iter()
            .map(|a| f64::from_bits(a.load(Ordering::Relaxed)))
            .collect();

        let fluence: Vec<f64> = fluence
            .iter()
            .map(|a| f64::from_bits(a.load(Ordering::Relaxed)))
            .collect();

        Ok(MCResult {
            dimensions: GridDimensions::from_grid(&self.grid),
            absorbed_energy,
            fluence,
            num_photons,
        })
    }

    /// Trace photon through medium
    fn trace_photon<R: Rng>(
        &self,
        mut photon: Photon,
        config: &SimulationConfig,
        absorbed_energy: &[AtomicU64],
        fluence: &[AtomicU64],
        rng: &mut R,
    ) {
        let bounds = [
            self.grid.dx * self.grid.nx as f64,
            self.grid.dy * self.grid.ny as f64,
            self.grid.dz * self.grid.nz as f64,
        ];

        let mut step_count = 0;
        let max_steps = config.max_steps;

        while photon.alive && step_count < max_steps {
            // Get optical properties at current position
            let (i, j, k) = match self.position_to_voxel(photon.position) {
                Some(idx) => idx,
                None => break, // Photon exited domain
            };

            let props = match self.optical_map.get(i, j, k) {
                Some(p) => p,
                None => break,
            };

            // Sample step length
            let mu_t = props.total_attenuation();
            if mu_t < 1e-12 {
                // No attenuation, move to boundary
                #[allow(unused_assignments)]
                {
                    photon.alive = false;
                }
                break;
            }

            let step_length = -rng.gen::<f64>().ln() / mu_t;

            // Check for boundary intersection
            let (hit_boundary, actual_step) = self.check_boundary_intersection(
                photon.position,
                photon.direction,
                step_length,
                bounds,
            );

            // Move photon
            photon.position = [
                photon.position[0] + actual_step * photon.direction[0],
                photon.position[1] + actual_step * photon.direction[1],
                photon.position[2] + actual_step * photon.direction[2],
            ];

            if hit_boundary {
                // Handle boundary reflection/transmission
                if config.boundary_reflection {
                    self.handle_boundary(&mut photon, rng);
                } else {
                    #[allow(unused_assignments)]
                    {
                        photon.alive = false;
                    }
                    break;
                }
            }

            // Accumulate fluence (photon path length * weight)
            let voxel_idx = k * (self.grid.nx * self.grid.ny) + j * self.grid.nx + i;
            if voxel_idx < fluence.len() {
                let fluence_increment = photon.weight * actual_step;
                atomic_add_f64(&fluence[voxel_idx], fluence_increment);
            }

            // Absorption
            let albedo = props.albedo();
            let absorbed = photon.weight * (1.0 - albedo);
            photon.weight *= albedo;

            // Accumulate absorbed energy
            if voxel_idx < absorbed_energy.len() {
                atomic_add_f64(&absorbed_energy[voxel_idx], absorbed);
            }

            // Russian roulette for low-weight photons
            if photon.weight < config.russian_roulette_threshold {
                if rng.gen::<f64>() < config.russian_roulette_survival {
                    photon.weight /= config.russian_roulette_survival;
                } else {
                    #[allow(unused_assignments)]
                    {
                        photon.alive = false;
                    }
                    break;
                }
            }

            // Scattering
            if photon.weight > 0.0 {
                scatter_photon(&mut photon, props.anisotropy, rng);
            }

            step_count += 1;
        }
    }

    /// Convert position to voxel indices
    fn position_to_voxel(&self, pos: [f64; 3]) -> Option<(usize, usize, usize)> {
        if pos[0] < 0.0
            || pos[1] < 0.0
            || pos[2] < 0.0
            || pos[0] >= self.grid.dx * self.grid.nx as f64
            || pos[1] >= self.grid.dy * self.grid.ny as f64
            || pos[2] >= self.grid.dz * self.grid.nz as f64
        {
            return None;
        }

        let i = (pos[0] / self.grid.dx).floor() as usize;
        let j = (pos[1] / self.grid.dy).floor() as usize;
        let k = (pos[2] / self.grid.dz).floor() as usize;

        if i < self.grid.nx && j < self.grid.ny && k < self.grid.nz {
            Some((i, j, k))
        } else {
            None
        }
    }

    /// Check for boundary intersection
    fn check_boundary_intersection(
        &self,
        pos: [f64; 3],
        dir: [f64; 3],
        step_length: f64,
        bounds: [f64; 3],
    ) -> (bool, f64) {
        let new_pos = [
            pos[0] + step_length * dir[0],
            pos[1] + step_length * dir[1],
            pos[2] + step_length * dir[2],
        ];

        // Check each axis
        for (axis, bound) in bounds.iter().enumerate() {
            if new_pos[axis] < 0.0 || new_pos[axis] >= *bound {
                // Compute distance to boundary
                let t = if dir[axis] > 0.0 {
                    (*bound - pos[axis]) / dir[axis]
                } else if dir[axis] < 0.0 {
                    -pos[axis] / dir[axis]
                } else {
                    f64::INFINITY
                };

                return (true, t.min(step_length));
            }
        }

        (false, step_length)
    }

    /// Handle boundary reflection/transmission
    fn handle_boundary<R: Rng>(&self, photon: &mut Photon, _rng: &mut R) {
        // Simple specular reflection at boundaries
        // Find which boundary was hit and reverse corresponding direction component
        let bounds = [
            self.grid.dx * self.grid.nx as f64,
            self.grid.dy * self.grid.ny as f64,
            self.grid.dz * self.grid.nz as f64,
        ];

        for (axis, bound) in bounds.iter().enumerate() {
            if photon.position[axis] <= 0.0 {
                photon.direction[axis] = photon.direction[axis].abs();
            } else if photon.position[axis] >= *bound {
                photon.direction[axis] = -photon.direction[axis].abs();
            }
        }
    }
}
