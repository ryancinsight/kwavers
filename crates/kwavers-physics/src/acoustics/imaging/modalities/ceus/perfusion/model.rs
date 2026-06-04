//! `CeusPerfusionModel` — advection-diffusion-reaction transport for CEUS agents.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use ndarray::Array3;

use super::kinetics::FlowKinetics;

/// Perfusion model for contrast agent kinetics
#[derive(Debug)]
pub struct CeusPerfusionModel {
    /// Concentration field (bubbles/m³)
    concentration: Array3<f64>,
    /// Blood flow velocity field (m/s)
    velocity: Array3<(f64, f64, f64)>,
    /// Trans-capillary permeability (m/s) (typical value: 1e-6 m/s for lipid-shell agents)
    permeability: f64,
    /// Grid spacing in x, y, z (m)
    dx: f64,
    dy: f64,
    dz: f64,
}

impl CeusPerfusionModel {
    /// Create new perfusion model
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(grid: &Grid, _medium: &dyn Medium) -> KwaversResult<Self> {
        let (nx, ny, nz) = grid.dimensions();

        Ok(Self {
            concentration: Array3::zeros((nx, ny, nz)),
            velocity: Array3::from_elem((nx, ny, nz), (0.0, 0.0, 0.0)),
            permeability: 1e-6, // m/s (typical capillary permeability for lipid-shell CEUS agents)
            dx: grid.dx,
            dy: grid.dy,
            dz: grid.dz,
        })
    }

    /// Advance concentration field by one timestep `dt`.
    ///
    /// Implements first-order upwind advection + trans-capillary clearance.
    /// A scratch buffer holds the state at time n; the update writes into
    /// `self.concentration` so no aliasing occurs.
    ///
    /// # Arguments
    /// * `inflow_concentration` — Dirichlet BC value at the inflow face [bubbles m⁻³]
    /// * `dt`                   — time step (s); caller must satisfy CFL: |u|·dt/dx ≤ 1
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn update_concentration(
        &mut self,
        inflow_concentration: f64,
        dt: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = self.concentration.dim();

        // Scratch copy of C at time n (upwind reads must come from unmodified state)
        let c_old = self.concentration.clone();

        // Effective clearance rate: k_perf = permeability / dx [s⁻¹]
        let dx = self.dx;
        let dy = self.dy;
        let dz = self.dz;
        let k_perf = self.permeability / dx;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let (vx, vy, vz) = self.velocity[[i, j, k]];

                    // First-order upwind advection in x
                    let dc_dx = if vx >= 0.0 {
                        if i > 0 {
                            (c_old[[i, j, k]] - c_old[[i - 1, j, k]]) / dx
                        } else {
                            0.0
                        }
                    } else if i + 1 < nx {
                        (c_old[[i + 1, j, k]] - c_old[[i, j, k]]) / dx
                    } else {
                        0.0
                    };

                    // First-order upwind advection in y
                    let dc_dy = if vy >= 0.0 {
                        if j > 0 {
                            (c_old[[i, j, k]] - c_old[[i, j - 1, k]]) / dy
                        } else {
                            0.0
                        }
                    } else if j + 1 < ny {
                        (c_old[[i, j + 1, k]] - c_old[[i, j, k]]) / dy
                    } else {
                        0.0
                    };

                    // First-order upwind advection in z
                    let dc_dz = if vz >= 0.0 {
                        if k > 0 {
                            (c_old[[i, j, k]] - c_old[[i, j, k - 1]]) / dz
                        } else {
                            0.0
                        }
                    } else if k + 1 < nz {
                        (c_old[[i, j, k + 1]] - c_old[[i, j, k]]) / dz
                    } else {
                        0.0
                    };

                    // dC/dt = −u·∇C − k_perf·C
                    let dc_dt = k_perf.mul_add(
                        -c_old[[i, j, k]],
                        -vz.mul_add(dc_dz, vx.mul_add(dc_dx, vy * dc_dy)),
                    );

                    self.concentration[[i, j, k]] = (c_old[[i, j, k]] + dc_dt * dt).max(0.0);
                }
            }
        }

        // Dirichlet inflow BC at i=0 (applied after the interior update)
        for j in 0..ny {
            for k in 0..nz {
                let (vx, _, _) = self.velocity[[0, j, k]];
                if vx >= 0.0 {
                    self.concentration[[0, j, k]] = inflow_concentration;
                }
            }
        }

        Ok(())
    }

    /// Get concentration at specific location
    #[must_use]
    pub fn concentration(&self, i: usize, j: usize, k: usize) -> f64 {
        self.concentration[[i, j, k]]
    }

    /// Get concentration field reference
    #[must_use]
    pub fn concentration_field(&self) -> &Array3<f64> {
        &self.concentration
    }

    /// Create a default gamma variate flow kinetics model suitable for CEUS bolus injection.
    #[must_use]
    pub fn gamma_variate_model() -> FlowKinetics {
        let frame_rate = 10.0;
        let duration = 30.0;
        let n_frames = (frame_rate * duration) as usize;
        let dt = 1.0 / frame_rate;

        let alpha = 3.0;
        let beta = 1.5;
        let tau = 0.5;

        let mut arterial_input = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let t = i as f64 * dt;
            let val = if t > 0.0 {
                (t / tau).powf(alpha) * (-(t - tau) / beta).exp()
            } else {
                0.0
            };
            arterial_input.push(val.max(0.0));
        }

        let mean_transit_time = 10.0;
        let mut residue_function = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let t = i as f64 * dt;
            residue_function.push((-(t / mean_transit_time)).exp());
        }

        FlowKinetics {
            arterial_input,
            residue_function,
            mean_transit_time,
        }
    }
}
