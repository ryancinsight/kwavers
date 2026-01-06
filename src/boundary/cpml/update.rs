//! CPML field update implementation

use super::memory::CPMLMemory;
use super::profiles::CPMLProfiles;
use ndarray::Array3;

/// CPML field updater
#[derive(Debug, Clone)]
pub struct CPMLUpdater {
    // Pre-computed update coefficients can be stored here
}

impl CPMLUpdater {
    /// Create new updater
    #[must_use]
    pub fn new() -> Self {
        Self {}
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
                    memory.psi_p_x[[i, j, k]] += gradient[[i, j, k]] * profiles.sigma_x[i];
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
                    memory.psi_p_y[[i, j, k]] += gradient[[i, j, k]] * profiles.sigma_y[j];
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
                    memory.psi_p_z[[i, j, k]] += gradient[[i, j, k]] * profiles.sigma_z[k];
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
                    gradient[[i, j, k]] += memory.psi_p_x[[i, j, k]] / profiles.kappa_x[i];
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
                    gradient[[i, j, k]] += memory.psi_p_y[[i, j, k]] / profiles.kappa_y[j];
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
                    gradient[[i, j, k]] += memory.psi_p_z[[i, j, k]] / profiles.kappa_z[k];
                }
            }
        }
    }
}
