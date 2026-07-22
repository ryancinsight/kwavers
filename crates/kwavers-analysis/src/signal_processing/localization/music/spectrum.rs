use crate::signal_processing::localization::SourceLocation;
use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use kwavers_math::linear_algebra::eigendecomposition::{EigenSolver, EigenSolverConfig};
use leto::Array2;

use super::super::model_order::{ModelOrderConfig, ModelOrderEstimator};
use super::{MUSICProcessor, MUSICResult};

impl MUSICProcessor {
    /// Compute MUSIC pseudospectrum over 3D search grid.
    ///
    /// For each grid point θ, computes P(θ) = 1 / (a^H E_n E_n^H a).
    /// Sources correspond to peaks of P(θ) where a(θ) ⊥ E_n.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_pseudospectrum(
        &self,
        noise_eigenvectors: &Array2<Complex64>,
    ) -> KwaversResult<(Vec<f64>, [usize; 3])> {
        let sensor_positions = &self.config.config.sensor_positions;
        let frequency = self.config.center_frequency;
        let speed_of_sound = self.config.config.sound_speed;
        let res = self.config.grid_resolution;

        let [xmin, xmax, ymin, ymax, zmin, zmax] = self.config.search_bounds;

        let nx = res;
        let ny = res;
        let nz = res;

        let dx = if nx > 1 {
            (xmax - xmin) / (nx - 1) as f64
        } else {
            0.0
        };
        let dy = if ny > 1 {
            (ymax - ymin) / (ny - 1) as f64
        } else {
            0.0
        };
        let dz = if nz > 1 {
            (zmax - zmin) / (nz - 1) as f64
        } else {
            0.0
        };

        let mut pseudospectrum = vec![0.0; nx * ny * nz];

        // Precompute E_n E_n^H for efficiency
        let num_sensors = sensor_positions.len();
        let noise_subspace_dim = noise_eigenvectors.shape()[1];
        let mut noise_projector = Array2::zeros((num_sensors, num_sensors));

        for i in 0..num_sensors {
            for j in 0..num_sensors {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..noise_subspace_dim {
                    sum += noise_eigenvectors[[i, k]] * noise_eigenvectors[[j, k]].conj();
                }
                noise_projector[[i, j]] = sum;
            }
        }

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let x = (ix as f64).mul_add(dx, xmin);
                    let y = (iy as f64).mul_add(dy, ymin);
                    let z = (iz as f64).mul_add(dz, zmin);

                    let steering = Self::steering_vector(
                        [x, y, z],
                        sensor_positions,
                        frequency,
                        speed_of_sound,
                    );

                    let mut denominator = 0.0;
                    for i in 0..num_sensors {
                        for j in 0..num_sensors {
                            let term = steering[i].conj() * noise_projector[[i, j]] * steering[j];
                            denominator += term.re;
                        }
                    }

                    let p_music = if denominator > 1e-12 {
                        1.0 / denominator
                    } else {
                        1e12
                    };

                    pseudospectrum[ix * ny * nz + iy * nz + iz] = p_music;
                }
            }
        }

        Ok((pseudospectrum, [nx, ny, nz]))
    }

    /// Find peaks in MUSIC pseudospectrum, filtered by minimum source separation.
    pub(super) fn find_peaks(
        &self,
        pseudospectrum: &[f64],
        grid_dims: [usize; 3],
        num_peaks: usize,
    ) -> Vec<SourceLocation> {
        let [nx, ny, nz] = grid_dims;
        let [xmin, xmax, ymin, ymax, zmin, zmax] = self.config.search_bounds;

        let mut candidates = Vec::new();

        for ix in 1..nx - 1 {
            for iy in 1..ny - 1 {
                for iz in 1..nz - 1 {
                    let idx = ix * ny * nz + iy * nz + iz;
                    let value = pseudospectrum[idx];

                    let mut is_local_max = true;
                    'neighbor_loop: for di in -1..=1 {
                        for dj in -1..=1 {
                            for dk in -1..=1 {
                                if di == 0 && dj == 0 && dk == 0 {
                                    continue;
                                }
                                let ni = (ix as i32 + di) as usize;
                                let nj = (iy as i32 + dj) as usize;
                                let nk = (iz as i32 + dk) as usize;
                                if pseudospectrum[ni * ny * nz + nj * nz + nk] > value {
                                    is_local_max = false;
                                    break 'neighbor_loop;
                                }
                            }
                        }
                    }

                    if is_local_max {
                        let x = xmin + ix as f64 * (xmax - xmin) / (nx - 1) as f64;
                        let y = ymin + iy as f64 * (ymax - ymin) / (ny - 1) as f64;
                        let z = zmin + iz as f64 * (zmax - zmin) / (nz - 1) as f64;
                        candidates.push((value, [x, y, z]));
                    }
                }
            }
        }

        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut sources: Vec<SourceLocation> = Vec::new();
        for (magnitude, position) in candidates {
            let mut too_close = false;
            for existing in &sources {
                let dx = position[0] - existing.position[0];
                let dy = position[1] - existing.position[1];
                let dz = position[2] - existing.position[2];
                if dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
                    < self.config.min_source_separation
                {
                    too_close = true;
                    break;
                }
            }

            if !too_close {
                sources.push(SourceLocation {
                    position,
                    confidence: magnitude.log10().clamp(0.0, 1.0),
                    uncertainty: 1.0 / magnitude.sqrt().max(1.0),
                });
                if sources.len() >= num_peaks {
                    break;
                }
            }
        }

        sources
    }

    /// Run complete MUSIC algorithm on complex sensor snapshots.
    ///
    /// Steps: covariance estimation → eigendecomposition → source number selection
    /// → noise subspace extraction → pseudospectrum → peak detection.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn run(&self, snapshots: &Array2<Complex64>) -> KwaversResult<MUSICResult> {
        let num_sensors = snapshots.shape()[0];
        let num_snapshots = snapshots.shape()[1];

        let covariance = self.estimate_covariance(snapshots)?;

        let (eigenvalues, eigenvectors) = {
            let r = EigenSolver::jacobi_hermitian(&covariance, EigenSolverConfig::default())?;
            (r.eigenvalues, r.eigenvectors)
        };

        let num_sources = if let Some(k) = self.config.num_sources {
            k
        } else {
            let model_config = ModelOrderConfig::new(num_sensors, num_snapshots)?
                .with_criterion(self.config.model_order_criterion);
            let estimator = ModelOrderEstimator::new(model_config)?;
            let real_eigenvalues: Vec<f64> = eigenvalues.iter().copied().collect();
            let result = estimator.estimate(&real_eigenvalues)?;
            result.num_sources
        };

        if num_sources == 0 {
            return Ok(MUSICResult {
                sources: Vec::new(),
                pseudospectrum: Vec::new(),
                grid_dims: [0, 0, 0],
                search_bounds: self.config.search_bounds,
                num_sources: 0,
                noise_subspace_dim: num_sensors,
            });
        }

        let noise_subspace_dim = num_sensors - num_sources;
        let mut noise_eigenvectors = Array2::zeros((num_sensors, noise_subspace_dim));

        for i in 0..num_sensors {
            for j in 0..noise_subspace_dim {
                noise_eigenvectors[[i, j]] = eigenvectors[[i, num_sources + j]];
            }
        }

        let (pseudospectrum, grid_dims) = self.compute_pseudospectrum(&noise_eigenvectors)?;
        let sources = self.find_peaks(&pseudospectrum, grid_dims, num_sources);

        Ok(MUSICResult {
            sources,
            pseudospectrum,
            grid_dims,
            search_bounds: self.config.search_bounds,
            num_sources,
            noise_subspace_dim,
        })
    }
}
