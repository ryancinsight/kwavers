use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::{Array1, Array3};

use super::WavefieldModeler;

impl WavefieldModeler {
    /// Apply PML boundary conditions (Berenger 1994).
    #[allow(dead_code)]
    pub(super) fn apply_pml(&self, wavefield: &mut Array3<f64>) {
        let (nx, ny, nz) = wavefield.dim();
        let width = self.pml_width;
        if width == 0 {
            return;
        }
        let limit = width.min(nx).min(ny).min(nz);

        let reflection_coeff: f64 = 1e-6;
        let pml_order = 2.0;
        let max_velocity = 4000.0;
        let max_damping = -(pml_order + 1.0) * max_velocity * reflection_coeff.ln()
            / (2.0 * width as f64 * 0.001);

        for i in 0..limit {
            let xi = (width - i) as f64 / width as f64;
            let damping = max_damping * xi.powf(pml_order);

            for j in 0..ny {
                for k in 0..nz {
                    wavefield[[i, j, k]] *= (-damping).exp();
                    wavefield[[nx - 1 - i, j, k]] *= (-damping).exp();
                }
            }
            for ii in 0..nx {
                for k in 0..nz {
                    wavefield[[ii, i, k]] *= (-damping).exp();
                    wavefield[[ii, ny - 1 - i, k]] *= (-damping).exp();
                }
            }
            for ii in 0..nx {
                for j in 0..ny {
                    wavefield[[ii, j, i]] *= (-damping).exp();
                    wavefield[[ii, j, nz - 1 - i]] *= (-damping).exp();
                }
            }
        }
    }

    /// Apply 4th-order finite difference stencil for wave equation.
    #[allow(dead_code)]
    pub(super) fn apply_fd_stencil(
        &self,
        current: &Array3<f64>,
        previous: &Array3<f64>,
        velocity: &Array3<f64>,
        dt: f64,
        dx: f64,
    ) -> Array3<f64> {
        let (nx, ny, nz) = current.dim();
        let mut next = Array3::zeros((nx, ny, nz));

        const C0: f64 = -5.0 / 2.0;
        const C1: f64 = 4.0 / 3.0;
        const C2: f64 = -1.0 / 12.0;

        for i in 2..nx.saturating_sub(2) {
            for j in 2..ny.saturating_sub(2) {
                for k in 2..nz.saturating_sub(2) {
                    let laplacian = C0 * current[[i, j, k]]
                        + C1 * (current[[i + 1, j, k]]
                            + current[[i - 1, j, k]]
                            + current[[i, j + 1, k]]
                            + current[[i, j - 1, k]]
                            + current[[i, j, k + 1]]
                            + current[[i, j, k - 1]])
                        + C2 * (current[[i + 2, j, k]]
                            + current[[i - 2, j, k]]
                            + current[[i, j + 2, k]]
                            + current[[i, j - 2, k]]
                            + current[[i, j, k + 2]]
                            + current[[i, j, k - 2]]);

                    let v2 = velocity[[i, j, k]] * velocity[[i, j, k]];
                    next[[i, j, k]] = 2.0 * current[[i, j, k]] - previous[[i, j, k]]
                        + dt * dt * v2 * laplacian / (dx * dx);
                }
            }
        }

        next
    }

    /// CFL-stable timestep: dt = CFL × dx / v_max.
    pub(super) fn calculate_stable_timestep(
        &self,
        velocity_model: &Array3<f64>,
    ) -> KwaversResult<f64> {
        let v_max = velocity_model.iter().copied().fold(0.0f64, f64::max);
        if !v_max.is_finite() || v_max <= 0.0 {
            return Err(KwaversError::Validation(
                ValidationError::ConstraintViolation {
                    message: "Velocity model must contain a finite, strictly positive maximum"
                        .to_string(),
                },
            ));
        }
        Ok(0.5 * self.config.dx / v_max)
    }

    /// Compute PML damping profile (polynomial grading).
    pub(super) fn compute_pml_profile(&self, thickness: usize, _max_dim: usize) -> Array1<f64> {
        if thickness == 0 {
            return Array1::zeros(0);
        }
        let mut profile = Array1::zeros(thickness);
        let reflection_coeff: f64 = 1e-6;
        let pml_order = 2.0;
        let max_velocity = 4000.0;

        let max_damping = -(pml_order + 1.0) * max_velocity * reflection_coeff.ln()
            / (2.0 * thickness as f64 * self.config.dx);

        for i in 0..thickness {
            let xi = (thickness - i) as f64 / thickness as f64;
            profile[i] = max_damping * xi.powf(pml_order);
        }

        profile
    }

    /// Apply PML absorbing boundaries to all six faces.
    pub(super) fn apply_pml_boundaries(
        &self,
        wavefield: &mut Array3<f64>,
        damping: &Array1<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
        thickness: usize,
    ) {
        let limit = thickness.min(damping.len()).min(nx).min(ny).min(nz);

        for i in 0..limit {
            let d = (-damping[i]).exp();

            for j in 0..ny {
                for k in 0..nz {
                    wavefield[[i, j, k]] *= d;
                    if nx > i {
                        wavefield[[nx - 1 - i, j, k]] *= d;
                    }
                }
            }
            for ii in 0..nx {
                for k in 0..nz {
                    wavefield[[ii, i, k]] *= d;
                    if ny > i {
                        wavefield[[ii, ny - 1 - i, k]] *= d;
                    }
                }
            }
            for ii in 0..nx {
                for j in 0..ny {
                    wavefield[[ii, j, i]] *= d;
                    if nz > i {
                        wavefield[[ii, j, nz - 1 - i]] *= d;
                    }
                }
            }
        }
    }

    /// 7-point stencil Laplacian: ∇²u ≈ (Σ_neighbors − 6u) / dx².
    pub(super) fn compute_laplacian_stencil_7pt(
        &self,
        field: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        dx2: f64,
    ) -> f64 {
        let center = field[[i, j, k]];
        let neighbors_sum = field[[i + 1, j, k]]
            + field[[i - 1, j, k]]
            + field[[i, j + 1, k]]
            + field[[i, j - 1, k]]
            + field[[i, j, k + 1]]
            + field[[i, j, k - 1]];

        (neighbors_sum - 6.0 * center) / dx2
    }

    /// Ricker wavelet: (1 − 2a²) exp(−a²) where a = π f_peak (t − 1/f_peak).
    pub(super) fn ricker_wavelet(&self, t: f64, f_peak: f64) -> f64 {
        let t0 = 1.0 / f_peak;
        let a = std::f64::consts::PI * f_peak * (t - t0);
        (1.0 - 2.0 * a * a) * (-a * a).exp()
    }
}
