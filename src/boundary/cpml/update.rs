//! CPML field update implementation

use super::config::CPMLConfig;
use super::memory::CPMLMemory;
use super::profiles::CPMLProfiles;
use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array3, Array4};

/// CPML field updater
#[derive(Clone, Debug))]
pub struct CPMLUpdater {
    // Pre-computed update coefficients can be stored here
}

impl CPMLUpdater {
    /// Create new updater
    pub fn new(_config: &CPMLConfig) -> Self {
        Self {}
    }

    /// Update fields with CPML boundary conditions
    pub fn update_fields(
        &self,
        fields: &mut Array4<f64>,
        memory: &mut CPMLMemory,
        profiles: &CPMLProfiles,
        dt: f64,
        grid: &Grid,
        config: &CPMLConfig,
    ) -> KwaversResult<()> {
        // Field indices
        const PRESSURE_IDX: usize = 0;
        const VX_IDX: usize = 1;
        const VY_IDX: usize = 2;
        const VZ_IDX: usize = 3;

        let thickness = config.thickness;

        // Update x-boundary regions
        self.update_x_boundaries(fields, memory, profiles, dt, grid, thickness)?;

        // Update y-boundary regions
        self.update_y_boundaries(fields, memory, profiles, dt, grid, thickness)?;

        // Update z-boundary regions
        self.update_z_boundaries(fields, memory, profiles, dt, grid, thickness)?;

        Ok(())
    }

    fn update_x_boundaries(
        &self,
        fields: &mut Array4<f64>,
        memory: &mut CPMLMemory,
        profiles: &CPMLProfiles,
        dt: f64,
        grid: &Grid,
        thickness: usize,
    ) -> KwaversResult<()> {
        // CPML update equations for x-boundaries (Roden & Gedney 2000, Eq. 7-8)
        // Left boundary (x = 0 to thickness)
        for i in 0..thickness {
            let b_x =
                (-dt * (profiles.sigma_x[i] / profiles.kappa_x[i] + profiles.alpha_x[i])).exp();
            let a_x = if profiles.sigma_x[i] > 0.0 {
                profiles.sigma_x[i]
                    / (profiles.sigma_x[i] * profiles.kappa_x[i]
                        + profiles.kappa_x[i] * profiles.kappa_x[i] * profiles.alpha_x[i])
                    * (b_x - 1.0)
            } else {
                0.0
            };

            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    // Update memory variable ψ_x
                    memory.psi_vx_x[[i, j, k] = b_x * memory.psi_vx_x[[i, j, k]
                        + a_x * (fields[[1, i + 1, j, k] - fields[[1, i, j, k]) / grid.dx;

                    // Apply to velocity field
                    fields[[1, i, j, k] += dt * memory.psi_vx_x[[i, j, k];
                }
            }
        }

        // Right boundary (x = nx - thickness to nx)
        for i in (grid.nx - thickness)..grid.nx {
            let idx = i - (grid.nx - thickness);
            let b_x =
                (-dt * (profiles.sigma_x[i] / profiles.kappa_x[i] + profiles.alpha_x[i])).exp();
            let a_x = if profiles.sigma_x[i] > 0.0 {
                profiles.sigma_x[i]
                    / (profiles.sigma_x[i] * profiles.kappa_x[i]
                        + profiles.kappa_x[i] * profiles.kappa_x[i] * profiles.alpha_x[i])
                    * (b_x - 1.0)
            } else {
                0.0
            };

            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    // Update memory variable
                    memory.psi_vx_x[[idx, j, k] = b_x * memory.psi_vx_x[[idx, j, k]
                        + a_x * (fields[[1, i, j, k] - fields[[1, i - 1, j, k]) / grid.dx;

                    // Apply to velocity field
                    fields[[1, i, j, k] += dt * memory.psi_vx_x[[idx, j, k];
                }
            }
        }

        Ok(())
    }

    fn update_y_boundaries(
        &self,
        fields: &mut Array4<f64>,
        memory: &mut CPMLMemory,
        profiles: &CPMLProfiles,
        dt: f64,
        grid: &Grid,
        thickness: usize,
    ) -> KwaversResult<()> {
        // CPML update equations for y-boundaries
        // Similar structure to x-boundaries but for y-direction
        for j in 0..thickness {
            let b_y =
                (-dt * (profiles.sigma_y[j] / profiles.kappa_y[j] + profiles.alpha_y[j])).exp();
            let a_y = if profiles.sigma_y[j] > 0.0 {
                profiles.sigma_y[j]
                    / (profiles.sigma_y[j] * profiles.kappa_y[j]
                        + profiles.kappa_y[j] * profiles.kappa_y[j] * profiles.alpha_y[j])
                    * (b_y - 1.0)
            } else {
                0.0
            };

            for i in 0..grid.nx {
                for k in 0..grid.nz {
                    memory.psi_vy_y[[i, j, k] = b_y * memory.psi_vy_y[[i, j, k]
                        + a_y * (fields[[2, i, j + 1, k] - fields[[2, i, j, k]) / grid.dy;
                    fields[[2, i, j, k] += dt * memory.psi_vy_y[[i, j, k];
                }
            }
        }

        for j in (grid.ny - thickness)..grid.ny {
            let idx = j - (grid.ny - thickness);
            let b_y =
                (-dt * (profiles.sigma_y[j] / profiles.kappa_y[j] + profiles.alpha_y[j])).exp();
            let a_y = if profiles.sigma_y[j] > 0.0 {
                profiles.sigma_y[j]
                    / (profiles.sigma_y[j] * profiles.kappa_y[j]
                        + profiles.kappa_y[j] * profiles.kappa_y[j] * profiles.alpha_y[j])
                    * (b_y - 1.0)
            } else {
                0.0
            };

            for i in 0..grid.nx {
                for k in 0..grid.nz {
                    memory.psi_vy_y[[i, idx, k] = b_y * memory.psi_vy_y[[i, idx, k]
                        + a_y * (fields[[2, i, j, k] - fields[[2, i, j - 1, k]) / grid.dy;
                    fields[[2, i, j, k] += dt * memory.psi_vy_y[[i, idx, k];
                }
            }
        }

        Ok(())
    }

    fn update_z_boundaries(
        &self,
        fields: &mut Array4<f64>,
        memory: &mut CPMLMemory,
        profiles: &CPMLProfiles,
        dt: f64,
        grid: &Grid,
        thickness: usize,
    ) -> KwaversResult<()> {
        // CPML update equations for z-boundaries
        for k in 0..thickness {
            let b_z =
                (-dt * (profiles.sigma_z[k] / profiles.kappa_z[k] + profiles.alpha_z[k])).exp();
            let a_z = if profiles.sigma_z[k] > 0.0 {
                profiles.sigma_z[k]
                    / (profiles.sigma_z[k] * profiles.kappa_z[k]
                        + profiles.kappa_z[k] * profiles.kappa_z[k] * profiles.alpha_z[k])
                    * (b_z - 1.0)
            } else {
                0.0
            };

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    memory.psi_vz_z[[i, j, k] = b_z * memory.psi_vz_z[[i, j, k]
                        + a_z * (fields[[3, i, j, k + 1] - fields[[3, i, j, k]) / grid.dz;
                    fields[[3, i, j, k] += dt * memory.psi_vz_z[[i, j, k];
                }
            }
        }

        for k in (grid.nz - thickness)..grid.nz {
            let idx = k - (grid.nz - thickness);
            let b_z =
                (-dt * (profiles.sigma_z[k] / profiles.kappa_z[k] + profiles.alpha_z[k])).exp();
            let a_z = if profiles.sigma_z[k] > 0.0 {
                profiles.sigma_z[k]
                    / (profiles.sigma_z[k] * profiles.kappa_z[k]
                        + profiles.kappa_z[k] * profiles.kappa_z[k] * profiles.alpha_z[k])
                    * (b_z - 1.0)
            } else {
                0.0
            };

            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    memory.psi_vz_z[[i, j, idx] = b_z * memory.psi_vz_z[[i, j, idx]
                        + a_z * (fields[[3, i, j, k] - fields[[3, i, j, k - 1]) / grid.dz;
                    fields[[3, i, j, k] += dt * memory.psi_vz_z[[i, j, idx];
                }
            }
        }

        Ok(())
    }

    /// Update memory component for acoustic gradients
    pub fn update_memory_component(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        component: usize,
        profiles: &CPMLProfiles,
    ) {
        match component {
            0 => self.update_x_memory(memory, gradient, profiles),
            1 => self.update_y_memory(memory, gradient, profiles),
            2 => self.update_z_memory(memory, gradient, profiles),
            _ => {}
        }
    }

    /// Apply gradient correction from CPML
    pub fn apply_gradient_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        component: usize,
        profiles: &CPMLProfiles,
    ) {
        match component {
            0 => self.apply_x_correction(gradient, memory, profiles),
            1 => self.apply_y_correction(gradient, memory, profiles),
            2 => self.apply_z_correction(gradient, memory, profiles),
            _ => {}
        }
    }

    fn update_x_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_x.dim().0;

        // Update memory for left boundary
        for i in 0..thickness.min(nx) {
            for j in 0..ny {
                for k in 0..nz {
                    memory.psi_p_x[[i, j, k] += gradient[[i, j, k] * profiles.sigma_x[i];
                }
            }
        }
    }

    fn update_y_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_y.dim().1;

        for i in 0..nx {
            for j in 0..thickness.min(ny) {
                for k in 0..nz {
                    memory.psi_p_y[[i, j, k] += gradient[[i, j, k] * profiles.sigma_y[j];
                }
            }
        }
    }

    fn update_z_memory(
        &self,
        memory: &mut CPMLMemory,
        gradient: &Array3<f64>,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_z.dim().2;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..thickness.min(nz) {
                    memory.psi_p_z[[i, j, k] += gradient[[i, j, k] * profiles.sigma_z[k];
                }
            }
        }
    }

    fn apply_x_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_x.dim().0;

        for i in 0..thickness.min(nx) {
            for j in 0..ny {
                for k in 0..nz {
                    gradient[[i, j, k] += memory.psi_p_x[[i, j, k] / profiles.kappa_x[i];
                }
            }
        }
    }

    fn apply_y_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_y.dim().1;

        for i in 0..nx {
            for j in 0..thickness.min(ny) {
                for k in 0..nz {
                    gradient[[i, j, k] += memory.psi_p_y[[i, j, k] / profiles.kappa_y[j];
                }
            }
        }
    }

    fn apply_z_correction(
        &self,
        gradient: &mut Array3<f64>,
        memory: &CPMLMemory,
        profiles: &CPMLProfiles,
    ) {
        let (nx, ny, nz) = gradient.dim();
        let thickness = memory.psi_p_z.dim().2;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..thickness.min(nz) {
                    gradient[[i, j, k] += memory.psi_p_z[[i, j, k] / profiles.kappa_z[k];
                }
            }
        }
    }
}
