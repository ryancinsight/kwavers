//! MCML photon tracing kernel.

use super::MonteCarloSolver;
use rand::Rng;
use std::sync::atomic::AtomicU64;

use crate::physics::optics::monte_carlo::config::SimulationConfig;
use crate::physics::optics::monte_carlo::interfaces::{apply_fresnel, fresnel_reflectance};
use crate::physics::optics::monte_carlo::photon::Photon;
use crate::physics::optics::monte_carlo::utils::{
    atomic_add, normalize, photon_step_to_boundary, scatter_photon,
};

impl MonteCarloSolver {
    /// Trace one photon through the medium using the MCML algorithm.
    pub(super) fn trace_photon<R: Rng>(
        &self,
        mut photon: Photon,
        config: &SimulationConfig,
        absorbed_energy: &[AtomicU64],
        fluence: &[AtomicU64],
        reflected_weight: &AtomicU64,
        rng: &mut R,
    ) {
        let mut step_count = 0;

        while step_count < config.max_steps {
            let (mut ci, mut cj, mut ck) = match self.position_to_voxel(photon.position) {
                Some(idx) => idx,
                None => {
                    if photon.position[2] < 0.0 {
                        atomic_add(reflected_weight, photon.weight);
                    }
                    break;
                }
            };

            let props = match self.optical_map.get(ci, cj, ck) {
                Some(p) => p,
                None => break,
            };

            let mu_t_init = props.total_attenuation();
            if mu_t_init < 1e-12 {
                photon.position[2] += self.grid.dz;
                step_count += 1;
                continue;
            }

            let mut s_remaining = -rng.gen::<f64>().ln() / mu_t_init;
            let mut mu_t_ref = mu_t_init;

            let scatter_occurred = loop {
                let (s_to_boundary, next_voxel) = photon_step_to_boundary(
                    photon.position,
                    photon.direction,
                    ci,
                    cj,
                    ck,
                    self.grid.dx,
                    self.grid.dy,
                    self.grid.dz,
                    self.grid.nx,
                    self.grid.ny,
                    self.grid.nz,
                );

                let voxel_idx = ck * (self.grid.nx * self.grid.ny) + cj * self.grid.nx + ci;

                if s_remaining <= s_to_boundary {
                    photon.position[0] += s_remaining * photon.direction[0];
                    photon.position[1] += s_remaining * photon.direction[1];
                    photon.position[2] += s_remaining * photon.direction[2];

                    if voxel_idx < fluence.len() {
                        atomic_add(&fluence[voxel_idx], photon.weight * s_remaining);
                    }

                    let albedo = self
                        .optical_map
                        .get(ci, cj, ck)
                        .map_or_else(|| props.albedo(), |p| p.albedo());
                    let absorbed = photon.weight * (1.0 - albedo);
                    photon.weight -= absorbed;
                    if voxel_idx < absorbed_energy.len() {
                        atomic_add(&absorbed_energy[voxel_idx], absorbed);
                    }

                    break true;
                }

                photon.position[0] += s_to_boundary * photon.direction[0];
                photon.position[1] += s_to_boundary * photon.direction[1];
                photon.position[2] += s_to_boundary * photon.direction[2];

                if voxel_idx < fluence.len() {
                    atomic_add(&fluence[voxel_idx], photon.weight * s_to_boundary);
                }

                s_remaining -= s_to_boundary;

                match next_voxel {
                    None => {
                        if photon.position[2] <= 0.0 && photon.direction[2] < 0.0 {
                            let n_tissue = self
                                .optical_map
                                .get(ci, cj, 0)
                                .map_or(1.4, |p| p.refractive_index);
                            let cos_i = (-photon.direction[2]).clamp(0.0, 1.0);
                            let r = fresnel_reflectance(n_tissue, 1.0, cos_i);
                            if rng.gen::<f64>() < r {
                                photon.position[2] = 1e-15;
                                photon.direction[2] = photon.direction[2].abs();
                                photon.direction = normalize(photon.direction);
                                if let Some((ni, nj, nk)) = self.position_to_voxel(photon.position)
                                {
                                    ci = ni;
                                    cj = nj;
                                    ck = nk;
                                }
                                continue;
                            }
                            atomic_add(reflected_weight, photon.weight);
                        }
                        if config.boundary_reflection
                            && !(photon.position[2] <= 0.0 && photon.direction[2] < 0.0)
                        {
                            self.handle_boundary(&mut photon);
                            if let Some((ni, nj, nk)) = self.position_to_voxel(photon.position) {
                                ci = ni;
                                cj = nj;
                                ck = nk;
                                if let Some(np) = self.optical_map.get(ci, cj, ck) {
                                    let new_mu_t = np.total_attenuation();
                                    if new_mu_t > 1e-12 && mu_t_ref > 1e-12 {
                                        s_remaining *= mu_t_ref / new_mu_t;
                                        mu_t_ref = new_mu_t;
                                    }
                                }
                                continue;
                            }
                        }
                        break false;
                    }
                    Some((ni, nj, nk)) => {
                        if let Some(np) = self.optical_map.get(ni, nj, nk) {
                            if (np.refractive_index - props.refractive_index).abs() > 1e-6 {
                                let n_hat = self.voxel_boundary_normal(
                                    photon.position,
                                    photon.direction,
                                    ci,
                                    cj,
                                    ck,
                                );
                                apply_fresnel(
                                    &mut photon.direction,
                                    n_hat,
                                    props.refractive_index,
                                    np.refractive_index,
                                    rng,
                                );
                                photon.direction = normalize(photon.direction);
                            }
                            let new_mu_t = np.total_attenuation();
                            if new_mu_t > 1e-12 && mu_t_ref > 1e-12 {
                                s_remaining *= mu_t_ref / new_mu_t;
                                mu_t_ref = new_mu_t;
                            }
                        }
                        ci = ni;
                        cj = nj;
                        ck = nk;
                    }
                }
            };

            if !scatter_occurred || photon.weight < 1e-30 {
                break;
            }

            if photon.weight < config.russian_roulette_threshold {
                if rng.gen::<f64>() < config.russian_roulette_survival {
                    photon.weight /= config.russian_roulette_survival;
                } else {
                    break;
                }
            }

            if let Some(p) = self.optical_map.get(ci, cj, ck) {
                scatter_photon(&mut photon, p.anisotropy, rng);
            }

            step_count += 1;
        }
    }
}
