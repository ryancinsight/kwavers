use anyhow::Result;
use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::domain::grid::{Grid3D, GridDimensions};
use crate::physics::optics::map_builder::OpticalPropertyMap;

use crate::physics::optics::monte_carlo::config::SimulationConfig;
use crate::physics::optics::monte_carlo::interfaces::{apply_fresnel, fresnel_reflectance};
use crate::physics::optics::monte_carlo::photon::Photon;
use crate::physics::optics::monte_carlo::result::MCResult;
use crate::physics::optics::monte_carlo::source::PhotonSource;
use crate::physics::optics::monte_carlo::utils::{
    atomic_add_f64, normalize, photon_step_to_boundary, scatter_photon,
};

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
        // Diffuse reflectance: accumulated reflected photon weight (z < 0 exits)
        let reflected_weight = Arc::new(AtomicU64::new(0));

        // Launch photons in parallel
        (0..num_photons)
            .into_par_iter()
            .chunks(chunk_size)
            .for_each(|chunk| {
                let mut rng = rand::thread_rng();

                for _ in chunk {
                    let mut photon = source.launch_photon(&mut rng);
                    // Specular entry weight reduction at air→tissue z=0 surface.
                    // **Theorem (Wang et al. 1995 §2.2):** A fraction R_spec of the
                    // incident power is specularly reflected at normal incidence:
                    //   R_spec = ((1 − n) / (1 + n))²
                    // so the initial photon weight is reduced to W₀ = 1 − R_spec.
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

        // Convert atomic accumulators to regular vectors
        let absorbed_energy: Vec<f64> = absorbed_energy
            .iter()
            .map(|a| f64::from_bits(a.load(Ordering::Relaxed)))
            .collect();

        let fluence: Vec<f64> = fluence
            .iter()
            .map(|a| f64::from_bits(a.load(Ordering::Relaxed)))
            .collect();

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

    /// Trace one photon through the medium using the MCML algorithm.
    ///
    /// ## Algorithm (Wang et al. 1995, §2)
    ///
    /// For each scatter event:
    /// 1. Sample total step `s = -ln(xi)/mu_t` for the current voxel's medium.
    /// 2. Sub-step voxel-by-voxel along `s` using `photon_step_to_boundary`:
    ///    a. Advance to the nearest voxel boundary or scatter point.
    ///    b. Accumulate fluence `Phi += W * ds` for each partial segment.
    ///    c. At each boundary with a refractive-index change, apply Fresnel R/T.
    ///    d. Rescale remaining step to the new medium's mu_t (Wang 1995 §2.3).
    /// 3. At the scatter point, deposit absorbed energy `dW = W * (1 - albedo)`
    ///    and scatter with Henyey-Greenstein.
    /// 4. Apply Russian roulette when `W < W_threshold` (Wang 1995 §2.6).
    /// 5. Track photons exiting source surface (z < 0) as diffuse reflectance.
    fn trace_photon<R: Rng>(
        &self,
        mut photon: Photon,
        config: &SimulationConfig,
        absorbed_energy: &[AtomicU64],
        fluence: &[AtomicU64],
        reflected_weight: &AtomicU64,
        rng: &mut R,
    ) {
        let mut step_count = 0;
        let max_steps = config.max_steps;

        while step_count < max_steps {
            // ── Get current voxel ─────────────────────────────────────────────
            let (mut ci, mut cj, mut ck) = match self.position_to_voxel(photon.position) {
                Some(idx) => idx,
                None => {
                    // Photon exited domain. Track reflectance from source surface (z < 0).
                    if photon.position[2] < 0.0 {
                        atomic_add_f64(reflected_weight, photon.weight);
                    }
                    break;
                }
            };

            let props = match self.optical_map.get(ci, cj, ck) {
                Some(p) => p,
                None => break,
            };

            // ── Sample total step length for this scatter event ───────────────
            let mu_t_init = props.total_attenuation();
            if mu_t_init < 1e-12 {
                // Transparent voxel — skip forward by one voxel width
                photon.position[2] += self.grid.dz;
                step_count += 1;
                continue;
            }

            let mut s_remaining = -rng.gen::<f64>().ln() / mu_t_init;
            // Track the mu_t of the medium where the step was sampled for rescaling
            let mut mu_t_ref = mu_t_init;

            // ── Sub-step voxel-by-voxel ───────────────────────────────────────
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
                    // Scatter point is inside the current voxel
                    photon.position[0] += s_remaining * photon.direction[0];
                    photon.position[1] += s_remaining * photon.direction[1];
                    photon.position[2] += s_remaining * photon.direction[2];

                    if voxel_idx < fluence.len() {
                        atomic_add_f64(&fluence[voxel_idx], photon.weight * s_remaining);
                    }

                    // Deposit absorbed energy (albedo splitting, Wang 1995 §2.5)
                    let albedo = if let Some(p) = self.optical_map.get(ci, cj, ck) {
                        p.albedo()
                    } else {
                        props.albedo()
                    };
                    let absorbed = photon.weight * (1.0 - albedo);
                    photon.weight -= absorbed;
                    if voxel_idx < absorbed_energy.len() {
                        atomic_add_f64(&absorbed_energy[voxel_idx], absorbed);
                    }

                    break true; // Scatter event occurred
                }

                // ── Boundary crossing ─────────────────────────────────────────
                photon.position[0] += s_to_boundary * photon.direction[0];
                photon.position[1] += s_to_boundary * photon.direction[1];
                photon.position[2] += s_to_boundary * photon.direction[2];

                if voxel_idx < fluence.len() {
                    atomic_add_f64(&fluence[voxel_idx], photon.weight * s_to_boundary);
                }

                s_remaining -= s_to_boundary;

                match next_voxel {
                    None => {
                        // ── z=0 tissue–air exit: Fresnel gate (Wang 1995 §2.7) ──────
                        //
                        // **Theorem:** At the tissue–air boundary (n₁ → 1.0) photons
                        // with angle θ_i > θ_c = arcsin(1/n₁) are totally internally
                        // reflected.  For θ_i < θ_c, the probability of transmission is
                        // T = 1 − R(θ_i), where R = (R_s + R_p)/2 (Fresnel, unpolarised).
                        // Only transmitted photons escape and are counted as Rd.
                        // Internally reflected photons re-enter the first voxel (k=0)
                        // and continue propagating, preserving weight exactly.
                        //
                        // **Reference:** Wang L, Jacques SL, Zheng L (1995) §2.7.
                        if photon.position[2] <= 0.0 && photon.direction[2] < 0.0 {
                            let n_tissue = self
                                .optical_map
                                .get(ci, cj, 0)
                                .map(|p| p.refractive_index)
                                .unwrap_or(1.4);
                            let cos_i = (-photon.direction[2]).clamp(0.0, 1.0);
                            let r = fresnel_reflectance(n_tissue, 1.0, cos_i);
                            if rng.gen::<f64>() < r {
                                // Internal reflection — push back just inside z=0 and flip
                                photon.position[2] = 1e-15;
                                photon.direction[2] = photon.direction[2].abs();
                                photon.direction = normalize(photon.direction);
                                if let Some((ni, nj, nk)) = self.position_to_voxel(photon.position)
                                {
                                    ci = ni;
                                    cj = nj;
                                    ck = nk;
                                }
                                continue; // resume voxel sub-step loop
                            } else {
                                // Transmitted through surface — diffuse reflectance
                                atomic_add_f64(reflected_weight, photon.weight);
                            }
                        }
                        if config.boundary_reflection
                            && !(photon.position[2] <= 0.0 && photon.direction[2] < 0.0)
                        {
                            self.handle_boundary(&mut photon, rng);
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
                        break false; // Photon escaped — no scatter
                    }
                    Some((ni, nj, nk)) => {
                        // Check Fresnel at refractive-index boundary
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
                            // Rescale remaining step to new medium's mu_t (Wang 1995 §2.3)
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

            // ── Russian roulette (Wang 1995 §2.6) ────────────────────────────
            // Variance-preserving: survive with prob m, boost weight by 1/m.
            if photon.weight < config.russian_roulette_threshold {
                if rng.gen::<f64>() < config.russian_roulette_survival {
                    photon.weight /= config.russian_roulette_survival;
                } else {
                    break;
                }
            }

            // ── Scattering (Henyey-Greenstein phase function) ─────────────────
            if let Some(p) = self.optical_map.get(ci, cj, ck) {
                scatter_photon(&mut photon, p.anisotropy, rng);
            }

            step_count += 1;
        }
    }

    /// Compute the inward surface normal at the voxel face the photon is
    /// about to cross (slab method, pointing into the incident medium).
    fn voxel_boundary_normal(
        &self,
        pos: [f64; 3],
        dir: [f64; 3],
        i: usize,
        j: usize,
        k: usize,
    ) -> [f64; 3] {
        let x0 = i as f64 * self.grid.dx;
        let y0 = j as f64 * self.grid.dy;
        let z0 = k as f64 * self.grid.dz;
        let x1 = x0 + self.grid.dx;
        let y1 = y0 + self.grid.dy;
        let z1 = z0 + self.grid.dz;

        let tx = if dir[0] > 1e-12 {
            (x1 - pos[0]) / dir[0]
        } else if dir[0] < -1e-12 {
            (x0 - pos[0]) / dir[0]
        } else {
            f64::INFINITY
        };
        let ty = if dir[1] > 1e-12 {
            (y1 - pos[1]) / dir[1]
        } else if dir[1] < -1e-12 {
            (y0 - pos[1]) / dir[1]
        } else {
            f64::INFINITY
        };
        let tz = if dir[2] > 1e-12 {
            (z1 - pos[2]) / dir[2]
        } else if dir[2] < -1e-12 {
            (z0 - pos[2]) / dir[2]
        } else {
            f64::INFINITY
        };

        if tx <= ty && tx <= tz {
            if dir[0] > 0.0 {
                [-1.0, 0.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            }
        } else if ty <= tz {
            if dir[1] > 0.0 {
                [0.0, -1.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            }
        } else {
            if dir[2] > 0.0 {
                [0.0, 0.0, -1.0]
            } else {
                [0.0, 0.0, 1.0]
            }
        }
    }

    /// Convert position to voxel indices
    pub(crate) fn position_to_voxel(&self, pos: [f64; 3]) -> Option<(usize, usize, usize)> {
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

    /// Handle boundary reflection/transmission (specular reflection)
    fn handle_boundary<R: Rng>(&self, photon: &mut Photon, _rng: &mut R) {
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
