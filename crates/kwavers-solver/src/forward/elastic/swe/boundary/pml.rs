//! `ElasticSwePMLBoundary` — pre-computed PML attenuation field and damping application.

use super::config::SwePmlConfig;
use kwavers_core::utils::iterators::{for_each_indexed_mut, for_each_indexed_pair_mut};
use kwavers_grid::Grid;
use leto::{Array1, Array3};
use moirai_parallel::{for_each_chunk_triple_mut_enumerated_with, Adaptive};

const PML_CHUNK: usize = 4096;

/// PML boundary condition calculator.
///
/// Computes and stores spatially-varying attenuation coefficients for the six
/// faces of a 3D rectangular domain using a power-law profile:
///
/// `σ(d) = σ_max * (d / L_pml)^n`
///
/// Damping is applied as `v(t+Δt) = v(t) * exp(-σ * Δt)`.
///
/// ## References
///
/// - Berenger, J. P. (1994). J. Comp. Phys., 114(2), 185–200.
/// - Komatitsch, D., & Martin, R. (2007). Geophysics, 72(5), SM155–SM167.
/// - Collino, F., & Tsogka, C. (2001). Geophysics, 66(1), 294–307.
#[derive(Debug)]
pub struct ElasticSwePMLBoundary {
    /// Attenuation coefficient field σ(x,y,z) (Np/m).
    sigma: Array3<f64>,
    /// Configuration parameters.
    config: SwePmlConfig,
}

impl ElasticSwePMLBoundary {
    /// Create a new PML boundary with pre-computed attenuation field.
    #[must_use]
    pub fn new(grid: &Grid, config: SwePmlConfig) -> Self {
        let sigma = Self::compute_attenuation_field(grid, &config);
        Self { sigma, config }
    }

    /// Attenuation coefficient at grid point (i,j,k) in Np/m.
    #[must_use]
    pub fn attenuation(&self, i: usize, j: usize, k: usize) -> f64 {
        self.sigma[[i, j, k]]
    }

    /// Full attenuation field (read-only reference).
    #[must_use]
    pub fn attenuation_field(&self) -> &Array3<f64> {
        &self.sigma
    }

    /// Returns `true` if the point lies within the PML region (σ > 0).
    #[must_use]
    pub fn is_in_pml(&self, i: usize, j: usize, k: usize) -> bool {
        self.sigma[[i, j, k]] > 0.0
    }

    /// Apply exponential PML damping to velocity components (in-place).
    ///
    /// `v(t+Δt) = v(t) * exp(-σ * Δt)`
    ///
    /// ## Theorem (race-freedom)
    ///
    /// Each element `(i,j,k)` of `{vx,vy,vz}` is updated independently using
    /// only the collocated value `self.sigma[[i,j,k]]`.  No element reads a
    /// neighbour → parallel updates are race-free.
    ///
    /// ## Theorem (PML exponential stability)
    ///
    /// The multiplicative factor `exp(-σ·Δt) ≤ 1` for σ ≥ 0, so the update
    /// is unconditionally stable: no time-step restriction beyond the elastic
    /// CFL condition is imposed by the PML absorption.
    ///
    /// Reference: Collino & Tsogka (2001), Geophysics 66(1), 294–307.
    pub fn apply_damping(
        &self,
        vx: &mut Array3<f64>,
        vy: &mut Array3<f64>,
        vz: &mut Array3<f64>,
        dt: f64,
    ) {
        apply_velocity_damping(vx, vy, vz, &self.sigma, dt);
    }

    /// Per-axis σ profiles matching the scalar `compute_attenuation_field` profile.
    ///
    /// Returns `(sigma_x, sigma_y, sigma_z)` each of length `n_α`.  Interior
    /// points have `σ = 0`; absorbing-layer points follow
    /// `σ(d) = σ_max · (d / L_pml)^order` with `d` the boundary distance.
    ///
    /// Callers that need per-step damping compute `exp(−σ · dt)` themselves,
    /// because `dt` is not known at `ElasticSwePMLBoundary` construction time.
    #[must_use]
    pub fn axis_sigma_profiles(&self, grid: &Grid) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let (nx, ny, nz) = grid.dimensions();
        let thickness = self.config.thickness;
        let sigma_max = self.config.sigma_max;
        let order = self.config.profile_order as i32;

        let compute_axis = |n: usize, active: bool| {
            let mut sigma = Array1::<f64>::zeros(n);
            if !active || thickness == 0 || n < 2 {
                return sigma;
            }
            for i in 0..thickness {
                if i < n {
                    let dist = (thickness - i) as f64;
                    sigma[i] = sigma_max * (dist / thickness as f64).powi(order);
                }
            }
            if n > thickness {
                for i in (n - thickness)..n {
                    let dist = (i - (n - thickness) + 1) as f64;
                    sigma[i] = sigma_max * (dist / thickness as f64).powi(order);
                }
            }
            sigma
        };

        (
            compute_axis(nx, nx > 1),
            compute_axis(ny, ny > 1),
            compute_axis(nz, nz > 1),
        )
    }

    /// Theoretical reflection coefficient: `R ≈ exp(-2 * σ_max * L_pml / c_max)`.
    ///
    /// Formula from Collino & Tsogka (2001).
    #[must_use]
    pub fn theoretical_reflection(&self, c_max: f64, grid: &Grid) -> f64 {
        let l_pml = self.config.thickness as f64 * grid.dx.min(grid.dy).min(grid.dz);
        let exponent = -2.0 * self.config.sigma_max * l_pml / c_max;
        exponent.exp()
    }

    /// Compute σ_max to achieve a target reflection coefficient.
    ///
    /// `σ_max = -ln(R_target) * c_max / (2 * L_pml)`
    #[must_use]
    pub fn optimize_sigma_max(
        target_reflection: f64,
        c_max: f64,
        grid: &Grid,
        thickness: usize,
    ) -> f64 {
        let l_pml = thickness as f64 * grid.dx.min(grid.dy).min(grid.dz);
        -target_reflection.ln() * c_max / (2.0 * l_pml)
    }

    /// Binary mask: 1.0 in PML region, 0.0 in interior.
    ///
    /// Parallelised elementwise: each cell independently maps σ > 0 → 1.0.
    #[must_use]
    pub fn get_mask(&self) -> Array3<f64> {
        let [nx, ny, nz] = self.sigma.shape();
        let mut mask = Array3::<f64>::zeros((nx, ny, nz));
        for_each_indexed_pair_mut(mask.view_mut(), self.sigma.view(), |_idx, m, &s| {
            if s > 0.0 {
                *m = 1.0;
            }
        });
        mask
    }

    /// Fraction of domain occupied by the PML region.
    #[must_use]
    pub fn volume_fraction(&self) -> f64 {
        let total_points = self.sigma.len();
        let pml_points = self.sigma.iter().filter(|&&s| s > 0.0).count();
        pml_points as f64 / total_points as f64
    }

    /// Compute the spatially-varying attenuation field for all six domain faces.
    ///
    /// ## Theorem (race-freedom)
    ///
    /// Each element `σ[i,j,k]` is computed from the cell indices and global
    /// scalars `{thickness, sigma_max, order, nx, ny, nz, pml_x, pml_y, pml_z}`
    /// without reading any other element → pointwise writes are race-free.
    ///
    /// ## PML power-law profile
    ///
    /// `σ(d) = σ_max · (d / L_pml)^n` where `d` is the distance from the
    /// domain boundary into the PML layer and `L_pml = thickness · Δx`.
    /// At each cell, the maximum σ contribution across all six faces is taken
    /// (corner and edge cells absorb from multiple directions simultaneously).
    fn compute_attenuation_field(grid: &Grid, config: &SwePmlConfig) -> Array3<f64> {
        let (nx, ny, nz) = grid.dimensions();
        let mut sigma = Array3::<f64>::zeros((nx, ny, nz));

        let thickness = config.thickness;
        let sigma_max = config.sigma_max;
        let order = config.profile_order as i32;

        // Degenerate axes (size == 1): no propagation → no PML needed.
        // Mirrors fd1_y / fd1_z guard (`if ny <= 1 { return 0.0 }`).
        let pml_x = nx > 1;
        let pml_y = ny > 1;
        let pml_z = nz > 1;

        for_each_indexed_mut(sigma.view_mut(), |(i, j, k), s| {
            let mut max_sigma = 0.0_f64;

            // X-direction PML (left and right faces)
            if pml_x {
                if i < thickness {
                    let dist = (thickness - i) as f64;
                    max_sigma = max_sigma.max(sigma_max * (dist / thickness as f64).powi(order));
                } else if nx > thickness && i >= nx - thickness {
                    let dist = (i - (nx - thickness) + 1) as f64;
                    max_sigma = max_sigma.max(sigma_max * (dist / thickness as f64).powi(order));
                }
            }

            // Y-direction PML (front and back faces)
            if pml_y {
                if j < thickness {
                    let dist = (thickness - j) as f64;
                    max_sigma = max_sigma.max(sigma_max * (dist / thickness as f64).powi(order));
                } else if ny > thickness && j >= ny - thickness {
                    let dist = (j - (ny - thickness) + 1) as f64;
                    max_sigma = max_sigma.max(sigma_max * (dist / thickness as f64).powi(order));
                }
            }

            // Z-direction PML (top and bottom faces)
            if pml_z {
                if k < thickness {
                    let dist = (thickness - k) as f64;
                    max_sigma = max_sigma.max(sigma_max * (dist / thickness as f64).powi(order));
                } else if nz > thickness && k >= nz - thickness {
                    let dist = (k - (nz - thickness) + 1) as f64;
                    max_sigma = max_sigma.max(sigma_max * (dist / thickness as f64).powi(order));
                }
            }

            *s = max_sigma;
        });

        sigma
    }
}

fn apply_velocity_damping(
    vx: &mut Array3<f64>,
    vy: &mut Array3<f64>,
    vz: &mut Array3<f64>,
    sigma: &Array3<f64>,
    dt: f64,
) {
    assert_eq!(
        vx.shape(),
        sigma.shape(),
        "invariant: PML vx/sigma shape mismatch"
    );
    assert_eq!(
        vx.shape(),
        vy.shape(),
        "invariant: PML vx/vy shape mismatch"
    );
    assert_eq!(
        vx.shape(),
        vz.shape(),
        "invariant: PML vx/vz shape mismatch"
    );

    match (
        vx.as_slice_mut(),
        vy.as_slice_mut(),
        vz.as_slice_mut(),
        sigma.as_slice(),
    ) {
        (Some(vx), Some(vy), Some(vz), Some(sigma)) => {
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                vx,
                vy,
                vz,
                PML_CHUNK,
                |chunk_index, vx_chunk, vy_chunk, vz_chunk| {
                    let start = chunk_index * PML_CHUNK;
                    for offset in 0..(vx_chunk.len()) {
                        let sigma_value = sigma[start + offset];
                        if sigma_value > 0.0 {
                            let damping = (-sigma_value * dt).exp();
                            vx_chunk[offset] *= damping;
                            vy_chunk[offset] *= damping;
                            vz_chunk[offset] *= damping;
                        }
                    }
                },
            );
        }
        _ => {
            // Index-based PML velocity damping over three mutable views; correct
            // for arbitrarily strided sub-region views (leto indexes by [i,j,k]).
            let sigma_v = sigma.view();
            let [nx, ny, nz] = vx.shape();
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let sigma_value = sigma_v[[i, j, k]];
                        if sigma_value > 0.0 {
                            let damping = (-sigma_value * dt).exp();
                            vx[[i, j, k]] *= damping;
                            vy[[i, j, k]] *= damping;
                            vz[[i, j, k]] *= damping;
                        }
                    }
                }
            }
        }
    }
}
