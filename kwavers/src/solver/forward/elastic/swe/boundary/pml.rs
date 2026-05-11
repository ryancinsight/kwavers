//! `PMLBoundary` — pre-computed PML attenuation field and damping application.

use super::config::PMLConfig;
use crate::domain::grid::Grid;
use ndarray::{Array1, Array3};

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
pub struct PMLBoundary {
    /// Attenuation coefficient field σ(x,y,z) (Np/m).
    sigma: Array3<f64>,
    /// Configuration parameters.
    config: PMLConfig,
}

impl PMLBoundary {
    /// Create a new PML boundary with pre-computed attenuation field.
    #[must_use]
    pub fn new(grid: &Grid, config: PMLConfig) -> Self {
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
    pub fn apply_damping(
        &self,
        vx: &mut Array3<f64>,
        vy: &mut Array3<f64>,
        vz: &mut Array3<f64>,
        dt: f64,
    ) {
        let (nx, ny, nz) = vx.dim();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let sigma = self.sigma[[i, j, k]];
                    if sigma > 0.0 {
                        let damping_factor = (-sigma * dt).exp();
                        vx[[i, j, k]] *= damping_factor;
                        vy[[i, j, k]] *= damping_factor;
                        vz[[i, j, k]] *= damping_factor;
                    }
                }
            }
        }
    }

    /// Per-axis σ profiles matching the scalar `compute_attenuation_field` profile.
    ///
    /// Returns `(sigma_x, sigma_y, sigma_z)` each of length `n_α`.  Interior
    /// points have `σ = 0`; absorbing-layer points follow
    /// `σ(d) = σ_max · (d / L_pml)^order` with `d` the boundary distance.
    ///
    /// Callers that need per-step damping compute `exp(−σ · dt)` themselves,
    /// because `dt` is not known at `PMLBoundary` construction time.
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
    #[must_use]
    pub fn get_mask(&self) -> Array3<f64> {
        let (nx, ny, nz) = self.sigma.dim();
        let mut mask = Array3::zeros((nx, ny, nz));
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if self.sigma[[i, j, k]] > 0.0 {
                        mask[[i, j, k]] = 1.0;
                    }
                }
            }
        }
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
    fn compute_attenuation_field(grid: &Grid, config: &PMLConfig) -> Array3<f64> {
        let (nx, ny, nz) = grid.dimensions();
        let mut sigma = Array3::zeros((nx, ny, nz));

        let thickness = config.thickness;
        let sigma_max = config.sigma_max;
        let order = config.profile_order;

        // Degenerate axes (size == 1): no propagation → no PML needed.
        // Mirrors fd1_y / fd1_z guard (`if ny <= 1 { return 0.0 }`).
        let pml_x = nx > 1;
        let pml_y = ny > 1;
        let pml_z = nz > 1;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mut max_sigma = 0.0_f64;

                    // X-direction PML (left and right faces)
                    if pml_x {
                        if i < thickness {
                            let dist = (thickness - i) as f64;
                            let val = sigma_max * (dist / thickness as f64).powi(order as i32);
                            max_sigma = max_sigma.max(val);
                        } else if nx > thickness && i >= nx - thickness {
                            let dist = (i - (nx - thickness) + 1) as f64;
                            let val = sigma_max * (dist / thickness as f64).powi(order as i32);
                            max_sigma = max_sigma.max(val);
                        }
                    }

                    // Y-direction PML (front and back faces)
                    if pml_y {
                        if j < thickness {
                            let dist = (thickness - j) as f64;
                            let val = sigma_max * (dist / thickness as f64).powi(order as i32);
                            max_sigma = max_sigma.max(val);
                        } else if ny > thickness && j >= ny - thickness {
                            let dist = (j - (ny - thickness) + 1) as f64;
                            let val = sigma_max * (dist / thickness as f64).powi(order as i32);
                            max_sigma = max_sigma.max(val);
                        }
                    }

                    // Z-direction PML (top and bottom faces)
                    if pml_z {
                        if k < thickness {
                            let dist = (thickness - k) as f64;
                            let val = sigma_max * (dist / thickness as f64).powi(order as i32);
                            max_sigma = max_sigma.max(val);
                        } else if nz > thickness && k >= nz - thickness {
                            let dist = (k - (nz - thickness) + 1) as f64;
                            let val = sigma_max * (dist / thickness as f64).powi(order as i32);
                            max_sigma = max_sigma.max(val);
                        }
                    }

                    sigma[[i, j, k]] = max_sigma;
                }
            }
        }

        sigma
    }
}
